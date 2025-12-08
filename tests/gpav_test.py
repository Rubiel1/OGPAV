from segmented_gpav import trend_following_order_lowery_fast
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
