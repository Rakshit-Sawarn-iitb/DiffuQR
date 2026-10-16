"""QR codeword layer: module grid <-> codewords <-> Reed-Solomon <-> payload.

This is what lets us decode from a module grid we produced ourselves
(sampled, restored, or posterior-fused) instead of handing pixels to
pyzbar, and it is where erasure decoding lives:

    grid (N x N bits)
      -> unmask + zigzag read of the data region       read_codewords
      -> de-interleave into RS blocks                   split_blocks
      -> RS decode per block, optional erasures         rs_decode_blocks
      -> parse mode segments into the payload           parse_payload

Per block, RS with `nsym` EC codewords succeeds when
    2 * errors + erasures <= nsym.
An erasure (a codeword we flag as unreliable) costs half an error, which is
the lever soft information pulls (contribution C3).

Spec tables (block structure, mask functions, fixed regions) come from
segno so nothing is hand-copied from ISO 18004. Uses segno internals;
verified on segno 1.6.6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np
from reedsolo import ReedSolomonError, RSCodec
from segno import consts as segno_consts
from segno import encoder as segno_encoder

from diffuqr.qrgrid import _SEGNO_ERROR, fixed_template, modules_for_version


# -- Layout ------------------------------------------------------------------

@lru_cache(maxsize=None)
def data_positions(version: int) -> Tuple[Tuple[int, int], ...]:
    """(row, col) of every data-region module in codeword bit order:
    two-column strips from the right edge, skipping the vertical timing
    column 6, alternating upward / downward, right column first."""
    n = modules_for_version(version)
    fixed, _ = fixed_template(version, "L", 0)  # function region is EC/mask independent
    out = []
    upward = True
    col = n - 1
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(n - 1, -1, -1) if upward else range(n)
        for r in rows:
            for c in (col, col - 1):
                if not fixed[r, c]:
                    out.append((r, c))
        upward = not upward
        col -= 2
    return tuple(out)


@lru_cache(maxsize=None)
def mask_array(version: int, mask: int) -> np.ndarray:
    n = modules_for_version(version)
    fn = segno_encoder.get_data_mask_functions(False)[mask]
    return np.array([[fn(i, j) for j in range(n)] for i in range(n)], dtype=np.uint8)


@lru_cache(maxsize=None)
def block_layout(version: int, error: str) -> Tuple[Tuple[int, int], ...]:
    """((n_total, n_data), ...) per block, in block order."""
    out = []
    for ec in segno_consts.ECC[version][_SEGNO_ERROR[error]]:
        out += [(ec.num_total, ec.num_data)] * ec.num_blocks
    return tuple(out)


def n_codewords(version: int, error: str) -> int:
    return sum(t for t, _ in block_layout(version, error))


# -- Grid <-> codewords ------------------------------------------------------

def read_codeword_bits(bits: np.ndarray, version: int, mask: int) -> np.ndarray:
    """Unmasked data-region bits in read order (includes remainder bits)."""
    pos = data_positions(version)
    rr, cc = np.array(pos).T
    return (bits[rr, cc] ^ mask_array(version, mask)[rr, cc]).astype(np.uint8)


def read_codewords(bits: np.ndarray, version: int, error: str, mask: int) -> np.ndarray:
    stream = read_codeword_bits(bits, version, mask)
    k = n_codewords(version, error)
    return np.packbits(stream[: 8 * k]).astype(np.uint8)


def write_codewords(codewords: Sequence[int], version: int, error: str, mask: int) -> np.ndarray:
    """Inverse of `read_codewords`: build the full grid (function patterns,
    format/version info, masked data, zero remainder bits)."""
    fixed, vals = fixed_template(version, error, mask)
    grid = vals.copy()
    stream = np.unpackbits(np.asarray(codewords, dtype=np.uint8))
    pos = data_positions(version)
    full = np.zeros(len(pos), dtype=np.uint8)
    full[: len(stream)] = stream
    rr, cc = np.array(pos).T
    grid[rr, cc] = full ^ mask_array(version, mask)[rr, cc]
    return grid


def codeword_module_index(version: int, error: str) -> List[List[Tuple[int, int]]]:
    """For each interleaved codeword, the 8 (row, col) modules carrying it.
    Used to turn per-module confidence into per-codeword confidence."""
    pos = data_positions(version)
    k = n_codewords(version, error)
    return [list(pos[8 * i: 8 * i + 8]) for i in range(k)]


# -- Interleaving ------------------------------------------------------------

@lru_cache(maxsize=None)
def interleave_map(version: int, error: str) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """For each block, the interleaved-stream index of each of its
    codewords (data first, then EC) -> ((block_idx, pos_in_block), ...)."""
    layout = block_layout(version, error)
    blocks = [[] for _ in layout]
    idx = 0
    max_data = max(d for _, d in layout)
    for i in range(max_data):
        for b, (_, d) in enumerate(layout):
            if i < d:
                blocks[b].append(idx)
                idx += 1
    max_ec = max(t - d for t, d in layout)
    for i in range(max_ec):
        for b, (t, d) in enumerate(layout):
            if i < t - d:
                blocks[b].append(idx)
                idx += 1
    return tuple(tuple(x) for x in blocks)


def split_blocks(codewords: np.ndarray, version: int, error: str) -> List[np.ndarray]:
    return [codewords[list(ix)] for ix in interleave_map(version, error)]


# -- RS ----------------------------------------------------------------------

@lru_cache(maxsize=None)
def _rs(nsym: int) -> RSCodec:
    # QR: GF(256) with primitive 0x11d, generator 2, first consecutive root 0
    # (reedsolo defaults).
    return RSCodec(nsym)


@dataclass
class BlockResult:
    ok: bool
    data: Optional[bytes]
    n_corrected: int          # codewords changed by the decoder (errors + erasures filled)
    n_erasures: int


@dataclass
class DecodeResult:
    ok: bool
    payload: Optional[str]
    blocks: List[BlockResult] = field(default_factory=list)
    corrected_codewords: Optional[np.ndarray] = None  # interleaved, post-RS (only if ok)


def rs_decode_blocks(
    codewords: np.ndarray,
    version: int,
    error: str,
    erasures: Optional[Sequence[int]] = None,
) -> List[BlockResult]:
    """RS-decode each block. `erasures` are indices into the INTERLEAVED
    codeword stream; they are routed to their blocks automatically."""
    layout = block_layout(version, error)
    imap = interleave_map(version, error)
    er = set(erasures or [])
    out = []
    for (total, ndata), ix in zip(layout, imap):
        block = bytearray(int(codewords[i]) for i in ix)
        erase_pos = [p for p, i in enumerate(ix) if i in er]
        nsym = total - ndata
        try:
            msg, full, _ = _rs(nsym).decode(block, erase_pos=erase_pos or None)
            n_corr = sum(a != b for a, b in zip(block, full))
            out.append(BlockResult(True, bytes(msg), n_corr, len(erase_pos)))
        except (ReedSolomonError, ValueError, ZeroDivisionError, IndexError):
            out.append(BlockResult(False, None, 0, len(erase_pos)))
    return out


# -- Payload parsing ---------------------------------------------------------

def _cci_bits(version: int, mode: int) -> int:
    tier = 0 if version <= 9 else (1 if version <= 26 else 2)
    return {1: (10, 12, 14), 2: (9, 11, 13), 4: (8, 16, 16), 8: (8, 10, 12)}[mode][tier]


_ALNUM = segno_consts.ALPHANUMERIC_CHARS.decode("ascii")


def parse_payload(data: bytes, version: int) -> Optional[str]:
    """Parse numeric / alphanumeric / byte / ECI segments. Byte segments
    are decoded as UTF-8, falling back to ISO-8859-1. Returns None on a
    malformed stream."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    p = 0

    def take(k: int) -> int:
        nonlocal p
        if p + k > len(bits):
            raise ValueError("overrun")
        v = 0
        for b in bits[p: p + k]:
            v = (v << 1) | int(b)
        p += k
        return v

    out_bytes = bytearray()
    try:
        while len(bits) - p >= 4:
            mode = take(4)
            if mode == 0:
                break
            if mode == 7:  # ECI: designator only, assume UTF-8 / default handling
                first = take(8)
                if first & 0x80:
                    take(8 if (first & 0xC0) == 0x80 else 16)
                continue
            if mode not in (1, 2, 4):
                return None  # kanji / structured append / invalid: not expected (likely RS mis-correction)
            count = take(_cci_bits(version, mode))
            if mode == 1:
                s = []
                while count >= 3:
                    s.append(f"{take(10):03d}"); count -= 3
                if count == 2:
                    s.append(f"{take(7):02d}")
                elif count == 1:
                    s.append(f"{take(4):01d}")
                out_bytes += "".join(s).encode("ascii")
            elif mode == 2:
                s = []
                while count >= 2:
                    v = take(11); s.append(_ALNUM[v // 45] + _ALNUM[v % 45]); count -= 2
                if count:
                    s.append(_ALNUM[take(6)])
                out_bytes += "".join(s).encode("ascii")
            else:
                out_bytes += bytes(take(8) for _ in range(count))
    except (ValueError, IndexError):
        return None
    try:
        return out_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return out_bytes.decode("iso-8859-1")


# -- Top level ---------------------------------------------------------------

def decode_grid(
    bits: np.ndarray,
    version: int,
    error: str,
    mask: int,
    erasures: Optional[Sequence[int]] = None,
) -> DecodeResult:
    """Full decode of a module grid with known (version, EC, mask)."""
    cw = read_codewords(bits, version, error, mask)
    blocks = rs_decode_blocks(cw, version, error, erasures)
    if not all(b.ok for b in blocks):
        return DecodeResult(False, None, blocks)
    data = b"".join(b.data for b in blocks)
    payload = parse_payload(data, version)
    corrected = None
    if payload is not None:
        corrected = np.zeros_like(cw)
        layout = block_layout(version, error)
        for (total, ndata), ix, b in zip(layout, interleave_map(version, error), blocks):
            full = _rs(total - ndata).encode(bytearray(b.data))
            corrected[list(ix)] = np.frombuffer(bytes(full), dtype=np.uint8)
    return DecodeResult(payload is not None, payload, blocks, corrected)


def codeword_confidence(module_conf: np.ndarray, version: int, error: str) -> np.ndarray:
    """Per-codeword confidence = min over its 8 modules' confidence."""
    idx = codeword_module_index(version, error)
    return np.array([min(module_conf[r, c] for r, c in mods) for mods in idx], dtype=np.float32)


def decode_soft(
    bits: np.ndarray,
    module_conf: np.ndarray,
    version: int,
    error: str,
    mask: int,
    per_block_erasures: Optional[Sequence[int]] = None,
    reserve: int = 4,
) -> Tuple[DecodeResult, int]:
    """Try hard decoding, then erase the k least-confident codewords of
    EACH block (RS budgets are per block), for increasing k. Returns
    (first successful result or the last failure, k used). Larger k raises
    the mis-correction risk; callers must compare against GT where
    available and report wrong decodes."""
    conf = codeword_confidence(module_conf, version, error)
    imap = interleave_map(version, error)
    nsym = min(t - d for t, d in block_layout(version, error))
    # Keep >= `reserve` check symbols unspent: with k == nsym erasures RS can
    # ALWAYS fit a codeword, so "success" says nothing (observed on real
    # frames: garbage payloads that parse). The reserve is what detects a
    # failed correction.
    ks = per_block_erasures if per_block_erasures is not None else range(0, nsym - reserve + 1, 2)
    last, last_k = None, 0
    for k in ks:
        er = []
        if k:
            for ix in imap:
                ix = np.array(ix)
                er += ix[np.argsort(conf[ix])[:k]].tolist()
        res = decode_grid(bits, version, error, mask, erasures=er or None)
        if res.ok:
            return res, k
        last, last_k = res, k
    return last, last_k
