"""
sb_gpav.py

SB-GPAV implementation accepting inputs (X, Y, L, f) as described:
  X: Dataset vectors (list or array)
  Y: Target values (real numbers)
  L: Topological order (list of indices of X)
  f: Optional comparison function f(a, b) -> bool (True if a <= b)

If f is not provided, defaults to coordinate-wise dominance:
  a <= b iff a[k] <= b[k] for all k.

References:
  Sysoev, Burdakov, Grimvall (2011) "A segmentation based algorithm..."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Callable, Any, Tuple, Dict, Union, Sequence
import hashlib
import itertools
import os
import sys
import time

import numpy as np
import networkx as nx
from .gpav import gpav_seg
from .trend_following import _build_dag_incrementally
from .progress import Progress, _hms
ArrayLike = Union[np.ndarray, Sequence[float]]

# ---------------------------------------------------------------------
# Comparator and Defaults
# ---------------------------------------------------------------------

def default_comparator(a: Any, b: Any) -> bool:
    """
    Default comparator: Coordinate-wise dominance (a <= b).
    Assumes a, b are array-like or comparable.
    """
    if a is b:
        return True
    
    # Numpy-aware check
    oa = np.asanyarray(a)
    ob = np.asanyarray(b)
    
    return bool(np.all(oa <= ob))




# ---------------------------------------------------------------------
# Helper: Build Hasse Diagram from Data (Induced on a subset)
# ---------------------------------------------------------------------

def _build_induced_hasse(
    indices: List[int],
    X: Any, 
    f: Callable[[Any, Any], bool],
    assume_component_wise: bool = False,
    progress=None,
) -> nx.DiGraph:
    """
    Build local Hasse diagram for a segment using memory-efficient construction.
    """
    def check_precedence(i: int, j: int) -> bool:
        # i, j are indices in X
        # Strict precedence: f(X[i], X[j]) AND i != j
        # Assuming topo order means if i precedes j in list, check f(i, j).
        # We enforce strict inequality for graph edges if values identical but distinct indices?
        # Actually standard GPAV works on strict poset.
        # If X[i] == X[j], they are equivalent. 
        # For simplicity, we use strict f(X[i], X[j]) is True.
        # If f(a,b) means a <= b, we check a <= b.
        # But for Hasse diagram, we usually want strict < relation or non-equivalence.
        # We rely on user f(a,b) meaning a <= b.
        # We add edge if a <= b. 
        # Incremental builder handles transitive reduction, so equal elements might form loops?
        # No, because loop `range(idx_j - 1, -1, -1)` enforces index order.
        return f(X[i], X[j])

    return _build_dag_incrementally(indices, check_precedence, assume_component_wise, progress=progress)


# ---------------------------------------------------------------------
# Helper: Block Extrema
# ---------------------------------------------------------------------

def _get_block_extrema(
    block_indices: List[int],
    X: Any,
    f: Callable[[Any, Any], bool]
) -> Tuple[List[int], List[int]]:
    """
    Find MIN(B) and MAX(B) within the block B using array comparator.
    
    MIN(B) = { u in B | not exists v in B s.t. v < u }
    MAX(B) = { u in B | not exists v in B s.t. u < v }
    """
    mins = []
    maxs = []
    
    for u in block_indices:
        # Check if u is minimal
        is_minimal = True
        val_u = X[u]
        for v in block_indices:
            if u == v: 
                continue
            val_v = X[v]
            if f(val_v, val_u):
                # v <= u. Check strictness v < u
                if not f(val_u, val_v):
                    is_minimal = False
                    break
        if is_minimal:
            mins.append(u)

        # Check if u is maximal
        is_maximal = True
        for v in block_indices:
            if u == v:
                continue
            val_v = X[v]
            if f(val_u, val_v):
                # u <= v. Check strictness u < v
                if not f(val_v, val_u):
                    is_maximal = False
                    break
        if is_maximal:
            maxs.append(u)
            
    return mins, maxs


# ---------------------------------------------------------------------
# Helper: Block Precedence (Min-Max Condition)
# ---------------------------------------------------------------------

def _block_precedes(
    minB_indices: List[int],
    maxA_indices: List[int],
    X: Any,
    f: Callable[[Any, Any], bool]
) -> bool:
    """
    Returns True if Block B precedes Block A (B -> A).
    Condition: exists b in min(B), a in max(A) s.t. b <= a.
    """
    for b in minB_indices:
        val_b = X[b]
        for a in maxA_indices:
            val_a = X[a]
            if f(val_b, val_a):
                return True
    return False


# ---------------------------------------------------------------------
# SB_GPAV Main
# ---------------------------------------------------------------------

@dataclass
class SegmentBlock:
    global_id: int
    members: List[int] # Indices in X
    value: float
    weight: float
    # These are computed locally:
    mins: List[int] 
    maxs: List[int]


# ---------------------------------------------------------------------
# Segment-level checkpointing (stage 1) -- flat numpy arrays, no pickle
# ---------------------------------------------------------------------

CKPT_VERSION = 1
_CK_KEYS = ("members", "mins", "maxs")


def _fsync_dir(d: str) -> None:
    try:
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _ragged(lists) -> Tuple[np.ndarray, np.ndarray]:
    lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
    flat = np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int64,
                       count=int(lens.sum()))
    return flat, lens


def _flatten_blocks(blocks: "List[SegmentBlock]") -> Dict[str, np.ndarray]:
    """One chunk of the checkpoint: the blocks of one segment, as flat arrays."""
    c: Dict[str, np.ndarray] = {}
    for key in _CK_KEYS:
        c[key], c[key + "_len"] = _ragged([getattr(b, key) for b in blocks])
    c["value"] = np.array([b.value for b in blocks], dtype=np.float64)
    c["weight"] = np.array([b.weight for b in blocks], dtype=np.float64)
    c["global_id"] = np.array([b.global_id for b in blocks], dtype=np.int64)
    return c


def _chunk_from_flat(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    c = {k: d[k] for k in ("members", "mins", "maxs", "value", "weight", "global_id")}
    for key in _CK_KEYS:
        c[key + "_len"] = np.diff(d[key + "_off"])
    return c


def _blocks_from_flat(d: Dict[str, np.ndarray]) -> "List[SegmentBlock]":
    flat = {k: d[k].tolist() for k in _CK_KEYS}
    off = {k: d[k + "_off"].tolist() for k in _CK_KEYS}
    val, wt, gid = d["value"].tolist(), d["weight"].tolist(), d["global_id"].tolist()
    return [
        SegmentBlock(
            global_id=gid[b], value=val[b], weight=wt[b],
            members=flat["members"][off["members"][b]:off["members"][b + 1]],
            mins=flat["mins"][off["mins"][b]:off["mins"][b + 1]],
            maxs=flat["maxs"][off["maxs"][b]:off["maxs"][b + 1]],
        )
        for b in range(len(val))
    ]


def _save_checkpoint(path: str, stage: int, next_s_idx: int,
                     chunks: List[Dict[str, np.ndarray]],
                     global_block_counter: int, fingerprint: str) -> None:
    """Atomic: tmp file + fsync + os.replace; the previous checkpoint is kept as .prev."""
    def cat(key, dtype):
        return (np.concatenate([c[key] for c in chunks]) if chunks
                else np.empty(0, dtype=dtype))
    arrays = dict(
        version=np.int64(CKPT_VERSION), stage=np.int64(stage),
        next_s_idx=np.int64(next_s_idx),
        global_block_counter=np.int64(global_block_counter),
        fingerprint=np.array(fingerprint),
        value=cat("value", np.float64), weight=cat("weight", np.float64),
        global_id=cat("global_id", np.int64),
    )
    for key in _CK_KEYS:
        arrays[key] = cat(key, np.int64)
        off = np.zeros(sum(len(c[key + "_len"]) for c in chunks) + 1, dtype=np.int64)
        np.cumsum(cat(key + "_len", np.int64), out=off[1:])
        arrays[key + "_off"] = off
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    if os.path.exists(path):
        os.replace(path, path + ".prev")
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(os.path.abspath(path)))


def _load_checkpoint(path: str, fingerprint: str, P: Progress) -> Optional[Dict[str, np.ndarray]]:
    """Newest readable checkpoint (path, else path.prev). Raises on version/fingerprint mismatch."""
    for cand in (path, path + ".prev"):
        if not os.path.exists(cand):
            continue
        try:
            with np.load(cand, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
        except Exception as e:  # truncated / corrupt file -> try the older copy
            P.log(f"WARNING: cannot read checkpoint {cand}: {e!r}")
            continue
        if int(d["version"]) != CKPT_VERSION:
            raise RuntimeError(f"{cand}: checkpoint version {int(d['version'])} != {CKPT_VERSION}")
        if str(d["fingerprint"]) != fingerprint:
            raise RuntimeError(
                f"{cand}: fingerprint mismatch -- the checkpoint was made with a different input, "
                "parameters or software versions. Refusing to resume; fix the setup, "
                "or pass resume=False / --no-resume (this overwrites the checkpoint).")
        P.log(f"loaded checkpoint {cand}")
        return d
    return None


def _make_fingerprint(X, Y, W, L, f, n_segments, assume_component_wise, block_order,
                      input_fingerprint) -> str:
    h = hashlib.sha256()
    def upd(s: str) -> None:
        h.update(s.encode()); h.update(b"\0")
    upd(f"ckpt_v{CKPT_VERSION}")
    upd(f"N={len(Y)};n_segments={n_segments};acw={assume_component_wise};block_order={block_order}")
    upd(f"f={getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', type(f).__name__)}")
    upd(f"py={sys.version_info[:3]};numpy={np.__version__};networkx={nx.__version__}")
    for arr in (np.asarray(L, dtype=np.int64), np.asarray(Y, dtype=np.float64), W):
        h.update(np.ascontiguousarray(arr).tobytes())
    if input_fingerprint is not None:
        upd(f"input={input_fingerprint}")
    else:  # no file identity supplied: hash the data itself
        try:
            h.update(memoryview(np.ascontiguousarray(X)).cast("B"))
        except (TypeError, ValueError):
            upd(f"X-shape={getattr(X, 'shape', None)}")
    return h.hexdigest()


def sb_gpav(
    X: np.ndarray,
    Y: ArrayLike,
    L: List[int],
    f: Optional[Callable[[Any, Any], bool]] = None,
    *,
    weights: Optional[ArrayLike] = None,
    n_segments: int = 10,
    assume_component_wise: bool = False,
    block_order: str = "kahn",
    verbose: bool = False,
    debug: bool = False,
    progress: bool = True,
    progress_interval: float = 600.0,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
    input_fingerprint: Optional[str] = None
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Segmentation-Based GPAV for array inputs with memory-efficient construction.
    
    Parameters
    ----------
    X : np.ndarray
        Dataset array of shape (N, d) where N is number of elements.
    L : list of int
        Topological order of all N elements (Algorithm 4, step 1). Segments are
        consecutive slices of L, and GPAV inside a segment follows L. The paper
        recommends the trend-following order (Algorithm 5); see
        utils.trend_following.trend_following_order.
    block_order : {"kahn", "kahn_lex", "trend"}
        Topological order on the block graph in the assembly stage (Algorithm 4,
        step 3, "any topological order"). "kahn": nx.topological_sort (original
        behaviour). "kahn_lex": the same algorithm with ties broken by block id.
        "trend": LowerY on the block values, as OperadicGPAV uses in Stage 2
        (used by utils.review.sb_gpav_review).
    progress, progress_interval : one flushed stderr line per phase start/end, and
        position/rate lines every `progress_interval` seconds inside long loops.
    checkpoint_path : if given, stage 1 is checkpointed after every segment (flat numpy
        arrays, atomic write, previous copy kept as <path>.prev) and, if `resume`, an
        existing checkpoint is continued. Refuses to resume on a fingerprint mismatch.
    input_fingerprint : identity of the input file (path/size/mtime) for the fingerprint;
        if None the data arrays themselves are hashed.
    """
    if block_order not in ("kahn", "kahn_lex", "trend"):
        raise ValueError(f"block_order must be 'kahn', 'kahn_lex' or 'trend', got {block_order!r}")
    
    # 0. Setup
    N = len(Y)
    if weights is None:
        W = np.ones(N, dtype=float)
    else:
        W = np.asarray(weights, dtype=float)
        
    if f is None:
        f = default_comparator
    P = Progress(interval=progress_interval, enabled=progress)

    # 1. Segmentation
    if n_segments <= 0:
        n_segments = 1
    n_segments = min(n_segments, N)
    
    # Chunk L into s segments
    base = N // n_segments
    rem = N % n_segments
    
    segments = []
    start = 0
    for s_i in range(n_segments):
        size = base + (1 if s_i < rem else 0)
        seg_indices = L[start : start + size]
        segments.append(seg_indices)
        start += size
        
    # 2. Local GPAV per segment
    all_blocks: List[SegmentBlock] = []
    global_block_counter = 0

    d = X.shape[1] if getattr(X, "ndim", 1) > 1 else 1
    P.log(
        f"sb_gpav START N={N} d={d} n_segments={n_segments} (segment size ~{N / n_segments:.1f}) "
        f"method=sb_gpav block_order={block_order} assume_component_wise={assume_component_wise} "
        f"weights={'given' if weights is not None else 'ones'} "
        f"f={getattr(f, '__qualname__', type(f).__name__)} verbose={verbose} debug={debug} "
        f"progress_interval={progress_interval}s checkpoint={checkpoint_path} resume={resume} "
        f"slurm_job={os.environ.get('SLURM_JOB_ID')} python={sys.version.split()[0]} "
        f"numpy={np.__version__} networkx={nx.__version__}")

    fp = None
    ck_chunks: List[Dict[str, np.ndarray]] = []
    start_seg = 0
    if checkpoint_path is not None:
        fp = _make_fingerprint(X, Y, W, L, f, n_segments, assume_component_wise,
                               block_order, input_fingerprint)
        ck = _load_checkpoint(checkpoint_path, fp, P) if resume else None
        if not resume and os.path.exists(checkpoint_path):
            P.log(f"WARNING: {checkpoint_path} exists but resume=False: starting from scratch (will overwrite)")
        if ck is not None:
            all_blocks = _blocks_from_flat(ck)
            global_block_counter = int(ck["global_block_counter"])
            start_seg = int(ck["next_s_idx"])
            ck_chunks = [_chunk_from_flat(ck)]
            P.log(f"RESUME stage={int(ck['stage'])}: segments done {start_seg}/{n_segments}, "
                  f"{len(all_blocks)} blocks restored")
        elif resume:
            P.log("no usable checkpoint found: starting from scratch")
    _run_t0 = time.monotonic()
    _n_done = 0
    
    
    for s_idx, seg_indices in enumerate(segments):
        if s_idx < start_seg:
            continue
        if not seg_indices:
            continue
        P.ctx = f"seg {s_idx + 1}/{n_segments}"
        _t_seg = time.monotonic()
        _n_before = len(all_blocks)
            
        # Build from array elements
        _t = P.start(f"hasse build: m={len(seg_indices)}")
        H_seg = _build_induced_hasse(seg_indices, X, f, assume_component_wise=assume_component_wise, progress=P)
        P.end(f"hasse build: m={len(seg_indices)} edges={H_seg.number_of_edges()}", _t)
        
        Y_map = {idx: float(Y[idx]) for idx in seg_indices}
        W_map = {idx: float(W[idx]) for idx in seg_indices}
        
        _, seg_block_dicts, _ = gpav_seg(
            Y=Y_map,
            poset=H_seg,
            topo_order=seg_indices,
            weights=W_map,
            verbose=verbose,
            name=f"Seg{s_idx}",
            progress=P
        )
        
        # Convert dicts to SegmentBlock objects
        _t = P.start(f"block extrema: {len(seg_block_dicts)} blocks, "
                     f"max size {max((len(b['labels']) for b in seg_block_dicts), default=0)}, "
                     f"est. comparator calls sum|B|^2={sum(len(b['labels']) ** 2 for b in seg_block_dicts):.3e}")
        _tk = P.ticker("block extrema (blocks)", len(seg_block_dicts))
        for b_dict in seg_block_dicts:
            if _tk is not None:
                _tk.step()
            members = b_dict['labels'] # list of indices
            # Compute extrema using array comparator
            mins, maxs = _get_block_extrema(members, X, f)
            
            blk = SegmentBlock(
                global_id=global_block_counter,
                members=members,
                value=b_dict['value'],
                weight=b_dict['weight'],
                mins=mins,
                maxs=maxs
            )
            all_blocks.append(blk)
            global_block_counter += 1
        P.end("block extrema", _t)

        _n_done += 1
        if checkpoint_path is not None:
            _stage = 2 if s_idx + 1 == n_segments else 1
            ck_chunks.append(_flatten_blocks(all_blocks[_n_before:]))
            _t = P.start(f"checkpoint save: stage={_stage} next_s_idx={s_idx + 1}")
            _save_checkpoint(checkpoint_path, _stage, s_idx + 1, ck_chunks, global_block_counter, fp)
            P.end("checkpoint save", _t)
        P.log(f"segment {s_idx + 1}/{n_segments} DONE m={len(seg_indices)} "
              f"blocks={len(all_blocks) - _n_before} total_blocks={len(all_blocks)} "
              f"seg_time={_hms(time.monotonic() - _t_seg)} "
              f"stage-1 naive-ETA={_hms((time.monotonic() - _run_t0) / _n_done * (n_segments - s_idx - 1))}")
            
    # 3. Assembly: Build Block DAG (Memory Efficient)
    B_total = len(all_blocks)
    P.ctx = "stage2"
    P.log(f"STAGE 1 DONE: B_total={B_total} blocks from N={N} points (mean size {N / max(B_total, 1):.2f}); "
          f"stage-2 block-DAG build = {B_total * (B_total - 1):.3e} ordered comparator calls "
          f"(2 x B^2/2 = {B_total ** 2 / 2:.3e} unordered pairs; each is a min x max loop)")
    
    def check_block_precedence(i: int, j: int) -> bool:
        # i, j are global block IDs
        return _block_precedes(all_blocks[i].mins, all_blocks[j].maxs, X, f)
    
    # Use incremental builder for blocks too.
    # Assumes sequential block IDs are topological order (which they are by segment ordering).
    block_indices = list(range(B_total))
    _t = P.start(f"block-DAG build: B_total={B_total}")
    G_blocks = _build_dag_incrementally(block_indices, check_block_precedence, assume_component_wise, progress=P)
    P.end(f"block-DAG build: edges={G_blocks.number_of_edges()}", _t)
        
    # 4. Global GPAV on blocks
    Y_blocks = np.array([b.value for b in all_blocks])
    W_blocks = np.array([b.weight for b in all_blocks])
    
    _t = P.start(f"topological sort ({block_order})")
    if block_order == "trend":
        from .trend_following import trend_following_order
        G_blocks.add_nodes_from(range(B_total))
        block_topo = trend_following_order(
            G=G_blocks, Y={i: float(Y_blocks[i]) for i in range(B_total)}, sparse_data=False)
    elif block_order == "kahn_lex":
        block_topo = list(nx.lexicographical_topological_sort(G_blocks))
    else:
        block_topo = list(nx.topological_sort(G_blocks))
    P.end("topological sort", _t)
    
    u_blocks_fitted, _, _ = gpav_seg(
        Y=Y_blocks,
        poset=G_blocks,
        topo_order=block_topo,
        weights=W_blocks,
        verbose=verbose,
        name="GPAV_Blocks",
        progress=P
    )
    
    # 5. Propagate back
    _t = P.start("propagation")
    u_final = np.zeros(N, dtype=float)
    for i in range(B_total):
        val = u_blocks_fitted[i]
        for member_idx in all_blocks[i].members:
            u_final[member_idx] = val
    P.end("propagation", _t)
    P.log("sb_gpav DONE")
            
    if debug:
        return u_final, {
            "all_blocks": all_blocks,
            "G_blocks": G_blocks,
            "u_blocks": u_blocks_fitted
        }
        
    return u_final
