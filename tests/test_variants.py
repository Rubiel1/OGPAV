"""
Tests for OperadicGPAV(variant=...) and the review benchmark helpers.

Reference fits in tests/data/variants_reference.npz were produced from the
original code, not from this code:
  main              main @ 4f3a8b2, unchanged
  main_kahn_stage2  main @ 4f3a8b2, use_trend_following_blocks=False, with
                    Stage 2 ordered by nx.lexicographical_topological_sort
  fast              PatchY @ b8c2d09, fast_mode=True
on the first 60 instances of _instances() below.
"""
import os
import warnings

import networkx as nx
import numpy as np
import pytest

from OperadicGPAV import OperadicGPAV, VARIANTS
from utils.trend_following import _lower_y_naive, trend_following_order

REF = os.path.join(os.path.dirname(__file__), "data", "variants_reference.npz")


def _instances(seed=7, n=60):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        m = int(rng.integers(2, 7))
        Q = nx.DiGraph([(a, b) for a in range(m) for b in range(a + 1, m) if rng.random() < .45])
        Q.add_nodes_from(range(m))
        R = [rng.random((int(rng.integers(1, 40)), int(rng.integers(1, 4)))) for _ in range(m)]
        yield Q, R, rng.normal(size=sum(map(len, R)))


def _run(Q, R, Y, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return OperadicGPAV(Q=Q, R_datasets=R, Y=Y, assume_component_wise=True, max_workers=1, **kw)


def _reference(key):
    data = np.load(REF)
    return np.split(data[key], np.cumsum(data["lengths"])[:-1])


def _full_poset_edges(Q, R):
    off = np.cumsum([0] + [len(r) for r in R])
    for i, r in enumerate(R):
        for a in range(len(r)):
            for b in range(len(r)):
                if a != b and np.all(r[a] <= r[b]) and not np.all(r[a] == r[b]):
                    yield off[i] + a, off[i] + b
    for i, j in nx.transitive_closure_dag(Q).edges():
        for a in range(len(R[i])):
            for b in range(len(R[j])):
                yield off[i] + a, off[j] + b


def _violations(Q, R, u, tol=1e-9):
    return sum(u[a] > u[b] + tol for a, b in _full_poset_edges(Q, R))


# --- exact reproduction of the original code --------------------------------

@pytest.mark.parametrize("variant", ["default", "review"])
def test_default_and_review_reproduce_main(variant):
    for (Q, R, Y), ref in zip(_instances(), _reference("main")):
        np.testing.assert_array_equal(_run(Q, R, Y, variant=variant), ref)


def test_kahn_stage2_override_matches_main_with_kahn():
    """use_trend_following_blocks=False changes only the Stage-2 order."""
    for (Q, R, Y), ref in zip(_instances(), _reference("main_kahn_stage2")):
        np.testing.assert_array_equal(_run(Q, R, Y, use_trend_following_blocks=False), ref)


def test_fast_reproduces_patchy_fast_mode_with_gateway_everywhere():
    """fast differs from PatchY only by the per-edge gateway rule: with the gateway
    forced onto every Q-edge (PatchY's fast_mode) it reproduces PatchY exactly."""
    import OperadicGPAV as og
    saved = og.VARIANTS["fast"]["gateway"]
    try:
        og.VARIANTS["fast"]["gateway"] = True
        for (Q, R, Y), ref in zip(_instances(), _reference("fast")):
            np.testing.assert_array_equal(_run(Q, R, Y, variant="fast"), ref)
    finally:
        og.VARIANTS["fast"]["gateway"] = saved


# --- properties --------------------------------------------------------------

@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_is_feasible(variant):
    rng = np.random.default_rng(101)
    for _ in range(40):
        m = int(rng.integers(2, 6))
        Q = nx.DiGraph([(a, b) for a in range(m) for b in range(a + 1, m) if rng.random() < .5])
        Q.add_nodes_from(range(m))
        R = [rng.random((int(rng.integers(1, 15)), int(rng.integers(1, 4)))) for _ in range(m)]
        Y = rng.normal(size=sum(map(len, R)))
        assert _violations(Q, R, _run(Q, R, Y, variant=variant)) == 0


def test_unknown_variant_is_rejected():
    Q = nx.DiGraph([(0, 1)])
    R = [np.array([[0.0]]), np.array([[1.0]])]
    with pytest.raises(ValueError, match="variant"):
        _run(Q, R, np.array([1.0, 0.0]), variant="exact")


@pytest.mark.parametrize("override", [
    dict(segment_topo_orders=[None, None]),
    dict(use_trend_following_first=False),
    dict(use_trend_following_blocks=True),
    dict(use_trend_following_blocks=False),
])
def test_review_configuration_is_locked(override):
    Q = nx.DiGraph([(0, 1)])
    R = [np.array([[0.0], [1.0]]), np.array([[2.0]])]
    with pytest.raises(ValueError, match="review"):
        _run(Q, R, np.array([2.0, 1.0, 0.0]), variant="review", **override)


def test_wide_fiber_no_recursion_error():
    """Main raised RecursionError here (D4): LowerY recursed once per picked element."""
    x = np.arange(1100.0)
    R = [np.vstack([[-1e4, -1e4], np.c_[x, -x]]), np.array([[5000.0, 5000.0]])]
    Y = np.random.default_rng(0).normal(size=1102)
    for variant in ("default", "review"):
        u = _run(nx.DiGraph([(0, 1)]), R, Y, variant=variant)
        assert u.shape == Y.shape


def test_gateway_needs_trend_following():
    """Q = 0->1, R_1 two incomparable points, data already monotone (the fit is Y).
    A gateway with a Y-blind Stage-2 order pools the two points to (3, 3); this is
    why the gateway is switched off whenever Stage 2 is not trend-following."""
    import OperadicGPAV as og
    Q = nx.DiGraph([(0, 1)])
    R = [np.array([[0.0, 0.0]]), np.array([[1.0, 2.0], [2.0, 1.0]])]
    Y = np.array([0.0, 5.0, 1.0])
    kw = dict(Q=Q, R_datasets=R, Y=Y, assume_component_wise=True, max_workers=1,
              indices_list=[[0], [1, 2]])
    saved = og.VARIANTS["fast"]["gateway"]
    try:
        og.VARIANTS["fast"]["gateway"] = True          # force the gateway (mechanism)
        np.testing.assert_array_equal(OperadicGPAV(variant="fast", **kw), Y)
        np.testing.assert_array_equal(
            OperadicGPAV(variant="fast", use_trend_following_blocks=False, **kw), [0.0, 3.0, 3.0])
    finally:
        og.VARIANTS["fast"]["gateway"] = saved
    # as delivered: no gateway where it saves nothing, and none with a Y-blind order
    np.testing.assert_array_equal(OperadicGPAV(variant="fast", **kw), Y)
    np.testing.assert_array_equal(OperadicGPAV(variant="fast", use_trend_following_blocks=False, **kw), Y)


# --- LowerY ------------------------------------------------------------------

def _lower_y_recursive(P, G, Y_map, sort_key):
    """The literal recursive translation of Algorithm 5 that main shipped."""
    if not P:
        return []
    i = P[0]
    anc = nx.ancestors(G, i)
    P1 = [v for v in P if v in anc]
    rest = [v for v in P if v not in anc and v != i]
    return _lower_y_recursive(sorted(P1, key=sort_key), G, Y_map, sort_key) + [i] + \
        _lower_y_recursive(rest, G, Y_map, sort_key)


def test_iterative_lower_y_matches_recursive():
    rng = np.random.default_rng(3)
    for t in range(150):
        n = int(rng.integers(2, 60))
        G = nx.gnp_random_graph(n, float(rng.uniform(.02, .4)), seed=t, directed=True)
        G = nx.DiGraph([(a, b) for a, b in G.edges() if a < b])
        G.add_nodes_from(range(n))
        Y = {v: float(rng.normal()) for v in G}

        def key(v):
            return (Y[v], str(v))
        P = sorted(G, key=key)
        assert _lower_y_naive(P, G, Y, key) == _lower_y_recursive(P, G, Y, key)


def test_trend_following_default_is_lower_y():
    G = nx.DiGraph([("r", "p"), ("p", "i"), ("s", "q"), ("q", "i")])
    Y = {"i": 0, "s": 1, "p": 2, "q": 3, "r": 9}
    assert trend_following_order(G=G, Y=Y) == ["s", "r", "p", "q", "i"]
    assert trend_following_order(G=G, Y=Y, sparse_data=True) == ["r", "p", "s", "q", "i"]


# --- review helpers ----------------------------------------------------------

def test_sb_gpav_review_is_feasible_and_compare_runs():
    from utils.review import compare_review, sb_gpav_review
    rng = np.random.default_rng(17)
    for _ in range(15):
        m = int(rng.integers(2, 5))
        Q = nx.DiGraph([(a, b) for a in range(m) for b in range(a + 1, m) if rng.random() < .6])
        Q.add_nodes_from(range(m))
        R = [rng.random((int(rng.integers(2, 20)), 2)) for _ in range(m)]
        Y = rng.normal(size=sum(map(len, R)))
        u, t = sb_gpav_review(Q, R, Y)
        assert _violations(Q, R, u) == 0
        assert set(t) == {"hasse", "order", "sb", "total"}
    res = compare_review(Q, R, Y)
    assert _violations(Q, R, res["u_ogpav"]) == 0
    assert res["sse_ogpav"] >= 0 and res["sse_sb"] >= 0


def test_sb_gpav_review_incremental_builder_drops_no_constraint():
    """sb_gpav_review builds block DAGs with the incremental builder (block ids taken
    as a linear extension). Check it gives the same fit as the all-pairs builder."""
    from utils.review import lexicographic_sum
    from utils.sb_gpav import sb_gpav
    from utils.trend_following import _build_dag_incrementally
    rng = np.random.default_rng(23)
    for _ in range(25):
        m = int(rng.integers(2, 5))
        Q = nx.DiGraph([(a, b) for a in range(m) for b in range(a + 1, m) if rng.random() < .6])
        Q.add_nodes_from(range(m))
        R = [rng.random((int(rng.integers(2, 20)), 2)) for _ in range(m)]
        Y = rng.normal(size=sum(map(len, R)))
        X, f_P, lin = lexicographic_sum(Q, R)
        G = _build_dag_incrementally(lin, lambda a, b: f_P(X[a], X[b]), True)
        G.add_nodes_from(range(len(X)))
        L = trend_following_order(G=G, Y={k: float(Y[k]) for k in range(len(X))})
        a = sb_gpav(X, Y, L, f_P, n_segments=m, assume_component_wise=True, block_order="trend")
        b = sb_gpav(X, Y, L, f_P, n_segments=m, assume_component_wise=False, block_order="trend")
        np.testing.assert_array_equal(a, b)


# --- weights (issue 1) -------------------------------------------------------

@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_weights_none_equals_all_ones(variant):
    for k, (Q, R, Y) in enumerate(_instances(seed=11, n=40)):
        np.testing.assert_array_equal(_run(Q, R, Y, variant=variant),
                                      _run(Q, R, Y, variant=variant, weights=np.ones(len(Y))))


def test_weights_worked_example():
    """a=(0,0) with Y=6 lies below b=(1,1), observed 3 times with Y=1,2,3. Merged: b has
    Y=2 and weight 3. The exact fit of the original data is 3 everywhere; without the
    weight GPAV pools to 4."""
    R = [np.array([[0.0, 0.0], [1.0, 1.0]])]
    Q = nx.DiGraph()
    Q.add_node(0)
    np.testing.assert_allclose(_run(Q, R, np.array([6.0, 2.0])), [4.0, 4.0])
    np.testing.assert_allclose(_run(Q, R, np.array([6.0, 2.0]), weights=[1, 3]), [3.0, 3.0])
    Q2 = nx.DiGraph([(0, 1)])            # same constraint, now across fibers
    R2 = [np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]])]
    np.testing.assert_allclose(_run(Q2, R2, np.array([6.0, 2.0]), weights=[1, 3]), [3.0, 3.0])


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_weights_properties(variant):
    """Feasible; weighted mean preserved (each block value is the weighted mean of its
    members); invariant to rescaling all weights; parallel equals sequential."""
    rng = np.random.default_rng(29)
    for t in range(25):
        m = int(rng.integers(2, 6))
        Q = nx.DiGraph([(a, b) for a in range(m) for b in range(a + 1, m) if rng.random() < .5])
        Q.add_nodes_from(range(m))
        R = [rng.random((int(rng.integers(1, 15)), 2)) for _ in range(m)]
        Y = rng.normal(size=sum(map(len, R)))
        w = rng.uniform(0.1, 5.0, len(Y))
        u = _run(Q, R, Y, variant=variant, weights=w)
        assert _violations(Q, R, u) == 0
        assert np.sum(w * u) == pytest.approx(np.sum(w * Y), abs=1e-9)
        np.testing.assert_allclose(_run(Q, R, Y, variant=variant, weights=7.5 * w), u, atol=1e-12)
        if t < 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                u2 = OperadicGPAV(Q=Q, R_datasets=R, Y=Y, assume_component_wise=True, max_workers=2,
                                  variant=variant, weights=w)
            np.testing.assert_array_equal(u, u2)


@pytest.mark.parametrize("bad", [[1.0], [1.0, 0.0, 1.0], [1.0, -1.0, 1.0], [1.0, np.nan, 1.0], [[1.0, 1.0, 1.0]]])
def test_invalid_weights_rejected(bad):
    Q = nx.DiGraph([(0, 1)])
    R = [np.array([[0.0], [1.0]]), np.array([[2.0]])]
    with pytest.raises(ValueError, match="weights"):
        _run(Q, R, np.array([2.0, 1.0, 0.0]), weights=bad)


def test_tie_error_points_to_weights():
    Q = nx.DiGraph([(0, 1)])
    # two copies of (0,0) with increasing Y stay in separate blocks, so the block
    # relation becomes cyclic (if GPAV pools the copies, no error is needed)
    R = [np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[9.0, 9.0]])]
    with pytest.raises(ValueError, match="weights"):
        _run(Q, R, np.array([0.0, 5.0, 9.0]))


def test_merged_ties_with_weights_give_feasible_fit_of_original():
    """Workflow from the tie error message: merge equal rows (mean Y, count as weight),
    fit, expand. The expanded fit respects every constraint of the original data,
    including equality of tied rows."""
    rng = np.random.default_rng(31)
    for _ in range(20):
        m = int(rng.integers(2, 5))
        Q = nx.DiGraph([(a, b) for a in range(m) for b in range(a + 1, m) if rng.random() < .5])
        Q.add_nodes_from(range(m))
        R = [rng.integers(0, 3, (int(rng.integers(3, 15)), 2)).astype(float) for _ in range(m)]
        Y = rng.normal(size=sum(map(len, R)))
        off = np.cumsum([0] + [len(r) for r in R])
        Rm, Ym, Wm, back, pos = [], [], [], np.empty(len(Y), int), 0
        for i, r in enumerate(R):
            uniq, inv = np.unique(r, axis=0, return_inverse=True)
            inv = np.asarray(inv).ravel()
            cnt = np.bincount(inv, minlength=len(uniq))
            Rm.append(uniq)
            Ym.append(np.bincount(inv, weights=Y[off[i]:off[i + 1]], minlength=len(uniq)) / cnt)
            Wm.append(cnt.astype(float))
            back[off[i]:off[i + 1]] = pos + inv
            pos += len(uniq)
        u = _run(Q, Rm, np.concatenate(Ym), weights=np.concatenate(Wm))[back]
        for a, b in _full_poset_edges(Q, R):
            assert u[a] <= u[b] + 1e-9
        for i, r in enumerate(R):        # tied rows share a value
            for a in range(len(r)):
                for b in range(len(r)):
                    if np.all(r[a] == r[b]):
                        assert u[off[i] + a] == u[off[i] + b]


def test_review_helpers_accept_weights():
    from utils.review import compare_review
    rng = np.random.default_rng(37)
    Q = nx.DiGraph([(0, 1), (0, 2)])
    R = [rng.random((10, 2)) for _ in range(3)]
    Y = rng.normal(size=30)
    w = rng.uniform(0.5, 3.0, 30)
    res = compare_review(Q, R, Y, weights=w)
    assert _violations(Q, R, res["u_ogpav"]) == 0 and _violations(Q, R, res["u_sb"]) == 0
    assert res["sse_ogpav"] == pytest.approx(float(np.sum(w * (res["u_ogpav"] - Y) ** 2)))


# --- comparator list (issue 2) ----------------------------------------------

@pytest.mark.parametrize("n_entries", [1, 3])
@pytest.mark.parametrize("q_edges", [[(0, 1)], []])
def test_comparator_list_must_have_one_entry_per_fiber(n_entries, q_edges):
    from utils.trend_following import default_comparator
    Q = nx.DiGraph(q_edges)
    Q.add_nodes_from(range(2))
    R = [np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[3.0, 3.0], [4.0, 4.0]])]
    with pytest.raises(ValueError, match="2 fibers"):
        OperadicGPAV(Q=Q, R_datasets=R, Y=np.array([1.0, 0.0, 8.0, 6.0]), max_workers=1,
                     f=[default_comparator] * n_entries, indices_list=[[0, 1], [2, 3]])


def test_comparator_list_entries():
    from utils.trend_following import default_comparator
    Q = nx.DiGraph([(0, 1)])
    R = [np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), np.array([[3.0, 3.0], [4.0, 4.0]])]
    Y = np.array([9.0, 1.0, 2.0, 8.0, 6.0])
    kw = dict(Q=Q, R_datasets=R, Y=Y, max_workers=1, indices_list=[[0, 1, 2], [3, 4]])
    with pytest.raises(TypeError, match=r"f\[1\]"):
        OperadicGPAV(f=[default_comparator, 5], **kw)
    # None is an explicit, documented choice: fiber 1 is unordered
    np.testing.assert_allclose(OperadicGPAV(f=[default_comparator, None], **kw), [4, 4, 4, 8, 6])
    np.testing.assert_allclose(OperadicGPAV(f=[default_comparator, default_comparator], **kw), [4, 4, 4, 7, 7])


# --- GPAV core: a block is never its own predecessor ---------------------------

def test_gpav_block_never_absorbs_itself():
    """Found by the gateway exploration (tier B, seed 200812, fiber 11). Rounding of a
    weighted average made block 9 re-enter its own predecessor list through the
    diamond 6 < 7, 6 < 8 < 9; gpav_seg then merged 9 into itself and raised KeyError."""
    from utils.gpav import gpav_seg, gpav_op
    edges = [(0, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 9), (3, 9), (4, 9), (5, 7), (6, 7),
             (6, 8), (7, 9), (8, 9), (9, 10), (9, 11), (10, 12), (11, 12)]
    Y = [3.0, 3.0, 0.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 0.0, 2.0]
    W = [0.8114532600335109, 2.6634728606274565, 0.8885110157206616, 3.530180443667549,
         1.0641357527570083, 1.33335631623451, 3.8955754646876675, 3.8030553438267733,
         3.890250694087653, 2.719369948855542, 2.0523225122012407, 4.101519298194084,
         3.983085528069095]
    topo = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10, 12]
    # Build the graph as Stage 1 does: GPAV's path depends on node insertion order
    # (nx.DiGraph(edges) would insert 9 before 7 and 8 and not reproduce the crash).
    from utils.trend_following import _build_dag_incrementally
    C = nx.transitive_closure_dag(nx.DiGraph(edges))
    G = _build_dag_incrementally(list(range(13)), lambda a, b: a == b or C.has_edge(a, b))
    Ym = {j: Y[j] for j in range(13)}
    Wm = {j: W[j] for j in range(13)}
    for fn in (gpav_seg, gpav_op):
        out = fn(Y=Ym, poset=G, topo_order=topo, weights=Wm)
        u, blocks = out[0], out[1]
        members = sorted(e for b in blocks for e in b["labels"])
        assert members == list(range(13)), "an element was lost or duplicated"
        assert sum(b["weight"] for b in blocks) == pytest.approx(sum(W))
        uu = [u[j] for j in range(13)] if isinstance(u, dict) else list(u)
        assert all(uu[a] <= uu[b] + 1e-12 for a, b in nx.transitive_closure_dag(G).edges())


# --- per-edge gateway rule ("auto") -------------------------------------------

def _gateway_count(Q, R, Y, **kw):
    import contextlib, io, re
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        OperadicGPAV(Q=Q, R_datasets=R, Y=Y, assume_component_wise=True, max_workers=1,
                     verbose=True, **kw)
    m = re.search(r"gateway: (\d+) gateway nodes", buf.getvalue())
    return int(m.group(1)) if m else 0


def test_auto_gateway_only_where_it_saves_edges():
    x = np.arange(30, dtype=float)
    chain_Q = nx.DiGraph([(0, 1), (1, 2)])
    # chain fibers with increasing Y: one block each (a = b = 1) -> plain edges only
    R = [np.c_[x, x] + 100 * k for k in range(3)]
    assert _gateway_count(chain_Q, R, np.arange(90, dtype=float)) == 0
    # antichain fibers: every point its own block, a = b = 30 -> one gateway per Q-edge
    R = [np.c_[x, -x] + 1000 * k for k in range(3)]
    Y = np.random.default_rng(0).normal(size=90)
    assert _gateway_count(chain_Q, R, Y) == 2
    # the same input with Kahn's order in Stage 2: the gateway is switched off
    assert _gateway_count(chain_Q, R, Y, use_trend_following_blocks=False) == 0
    # fast follows the same rule
    assert _gateway_count(chain_Q, R, Y, variant="fast") == 2
    R1 = [np.c_[x, x] + 100 * k for k in range(3)]
    assert _gateway_count(chain_Q, R1, np.arange(90, dtype=float), variant="fast") == 0


@pytest.mark.parametrize("variant", ["default", "review"])
def test_auto_gateway_same_fit_as_plain_edges(variant):
    """Empirical property (not a theorem): with LowerY in Stage 2 the gateway leaves
    the fit unchanged. Checked here against the plain-edge form, with weights."""
    import OperadicGPAV as og
    rng = np.random.default_rng(41)
    for _ in range(30):
        m = int(rng.integers(2, 7))
        Q = nx.DiGraph([(a, b) for a in range(m) for b in range(a + 1, m) if rng.random() < .5])
        Q.add_nodes_from(range(m))
        R = [rng.random((int(rng.integers(1, 25)), 2)) for _ in range(m)]
        Y = rng.normal(size=sum(map(len, R)))
        w = rng.uniform(0.2, 4.0, len(Y))
        u = _run(Q, R, Y, variant=variant, weights=w)
        saved = og.VARIANTS[variant]["gateway"]
        try:
            og.VARIANTS[variant]["gateway"] = False
            u_plain = _run(Q, R, Y, variant=variant, weights=w)
        finally:
            og.VARIANTS[variant]["gateway"] = saved
        np.testing.assert_allclose(u, u_plain, rtol=1e-10, atol=1e-10)
