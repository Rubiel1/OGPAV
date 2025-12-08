from segmented_gpav import trend_following_order_lowery_fast
from operadic_gpav import factorized_gpav_fast_parallel
from gpav import gpav_op, gpav_seg
import networkx as nx
import numpy as np

class TestClass:
    def test_line_op(self):
        # a <  b < c < d, identity
        G = nx.DiGraph()
        G.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])
        nodes = list(G.nodes())
        Y_map = {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0}
        Y_aligned = np.array([Y_map[v] for v in nodes], dtype=float)
        # A topological order T
        T = trend_following_order_lowery_fast(G, Y_aligned)
        u, _, _ = gpav_op(Y_aligned, G, topo_order=T, verbose=True)
        v, _, _ = gpav_seg(Y_aligned, G, topo_order=T, verbose=True)
        assert set(u) == set([1.0,2.0,3.0,4.0]), "error of GPAV_op when the input is the trivial line already ordered "
        assert set(v) == set([1.0,2.0,3.0,4.0]), "error of GPAV_seg when the input is the trivial line already ordered "
    def test_c_op(self):
        # a<b<c<d<e, a<f<e
        # Example 1: Small Q with 4 components R_i
        poset = hasse.PoSet.from_chains([0, 1, 2], [0, 3, 2])
        R_subposets = [
            hasse.PoSet.from_chains([0],),
            hasse.PoSet.from_chains([0,1,2]),
            hasse.PoSet.from_chains([0]),
            hasse.PoSet.from_chains([0]),
        ]
        Y = [10,13,11,9,12,11]
        Y_map = {i: float(y) for i, y in enumerate(Y)}
        u = factorized_gpav_fast_parallel(
            Q=poset,
            R_subposets=R_subposets,
            A=Y_map,
            use_lowerY_first=True,
            use_lowerY_blocks=True,
            inputs_are_reduced=False,
            verbose=True,
        )
        assert set(u) == {np.float64(10.0), np.float64(11.0), np.float64(12.0)}, "error of factorized_gpav_fast_parallel"