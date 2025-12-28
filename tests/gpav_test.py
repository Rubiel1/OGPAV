from trend_following import trend_following_order
from operadic_gpav import OGPAV
from gpav import gpav_op, gpav_seg
import networkx as nx
import numpy as np
import hasse

class TestClass:
    def test_line_op(self):
        # a <  b < c < d, identity
        G = nx.DiGraph()
        G.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])
        nodes = list(G.nodes())
        Y_map = {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0}
        Y_aligned = np.array([Y_map[v] for v in nodes], dtype=float)
        # A topological order T
        T = trend_following_order(G, Y_aligned)
        u, _, _ = gpav_op(Y_aligned, G, topo_order=T, verbose=True)
        v, _, _ = gpav_seg(Y_aligned, G, topo_order=T, verbose=True)
        assert set(u) == set([1.0,2.0,3.0,4.0]), "error of GPAV_op when the input is the trivial line already ordered "
        assert set(v) == set([1.0,2.0,3.0,4.0]), "error of GPAV_seg when the input is the trivial line already ordered "
    def test_closed_op(self):
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
        u = OGPAV(
            Q=poset,
            R_subposets=R_subposets,
            A=Y_map,
            use_trend_following_first=True,
            use_trend_following_blocks=True,
            inputs_are_reduced=False,
            verbose=True,
        )
        assert set(u) == {np.float64(10.0), np.float64(11.0), np.float64(12.0)}, "error of OGPAV"
    def test_position_op(self):
        # a<b<c, compose on b by {x,y,z} < {X,Y,Z}
        # Example 1: Small Q with 3 components R_i
        poset = hasse.PoSet.from_chains([0,1,2])
        R_subposets = [
            hasse.PoSet.from_chains([0],),
            hasse.PoSet.from_chains([0,1],[2,3],[4,5],[0,3],[2,1],[2,5],[4,3]),
            hasse.PoSet.from_chains([0]),
        ]
        Y = [1,7,5,3,6,9,10,4]
        Y_map = {i: float(y) for i, y in enumerate(Y)}
        u = OGPAV(
            Q=poset,
            R_subposets=R_subposets,
            A=Y_map,
            use_trend_following_first=True,
            use_trend_following_blocks=True,
            inputs_are_reduced=False,
            verbose=True,
        )
        assert u[3] == Y[3], "error on the implementation of OGPAV, it is modifying nodes that are supposed to be untouched"
    def test_position2_op(self):
        # a<b<c, compose on b by {x,y,z} < {X,Y,Z}
        # Example 1: Small Q with 2 components R_i
        poset = hasse.PoSet.from_chains([0,1])
        R_subposets = [
            hasse.PoSet.from_chains([0, 1], [0, 2], [0, 3]),
            hasse.PoSet.from_chains([0, 1],[2, 1], [3, 1]),
        ]
        Y = [1, 7, 3, 9, 5, 6, 10, 4]
        Y_map = {i: float(y) for i, y in enumerate(Y)}
        u = OGPAV(
            Q=poset,
            R_subposets=R_subposets,
            A=Y_map,
            use_trend_following_first=True,
            use_trend_following_blocks=True,
            inputs_are_reduced=False,
            verbose=True,
        )
        assert u[2] == Y[2], "error on the implementation of OGPAV, it is modifying nodes that are supposed to be untouched"
    def test_secondstage_op(self):
        # a,b<c, compose on a by {x,y} 
        # Example 1: Small Q with 3 components R_i
        poset = hasse.PoSet.from_chains([0,1],[2,1])
        R_subposets = [
            hasse.PoSet.from_chains([0],[1]),#orden global 0,1
            hasse.PoSet.from_chains([0]),#orden global 2
            hasse.PoSet.from_chains([0]),#orden global 3
        ]
        Y = [10,9,8,12]
        Y_map = {i: float(y) for i, y in enumerate(Y)}
        u = OGPAV(
            Q=poset,
            R_subposets=R_subposets,
            A=Y_map,
            use_trend_following_first=True,
            use_trend_following_blocks=True,
            inputs_are_reduced=False,
            verbose=True,
        )#[]
        assert np.array_equal(u, np.array([10.0,9.0,10.0,10.0])), "error on the implementation of OGPAV, it is modifying nodes that are supposed to be untouched"
    def test_order_input_op(self):
        poset = hasse.PoSet.from_chains([0,1,2])
        R_subposets = [
            hasse.PoSet.from_chains([0]),#orden global 0
            hasse.PoSet.from_chains([0,3],[1,4],[2,5],[0,4],[1,5],[2,3],[0,5],[1,3],[2,4]),#orden global 1,2,3
            hasse.PoSet.from_chains([0]),#orden global 7
        ]
        Y = [1,7,3,9,5,6,10,4]
        Y_map = {i: float(y) for i, y in enumerate(Y)}
        u = OGPAV(
            Q=poset,
            R_subposets=R_subposets,
            A=Y_map,
            use_trend_following_first=True,
            use_trend_following_blocks=True,
            inputs_are_reduced=False,
            verbose=True,
        )#[]
        poset = hasse.PoSet.from_chains([0,1,2])
        R_subposets = [
            hasse.PoSet.from_chains([0]),#orden global 0
            hasse.PoSet.from_chains([0],[1],[0,3],[1,4],[2,5],[0,4],[1,5],[2,3],[0,5],[1,3],[2,4]),#orden global 1,2,3
            hasse.PoSet.from_chains([0]),#orden global 7
        ]
        Y_map = {i: float(y) for i, y in enumerate(Y)}
        v = OGPAV(
            Q=poset,
            R_subposets=R_subposets,
            A=Y_map,
            use_trend_following_first=True,
            use_trend_following_blocks=True,
            inputs_are_reduced=False,
            verbose=True,
        )#[]
        assert  np.array_equal(u,v), "error, the orden of the inputs when creating the list Y or the poset afects the internal order, independently of the label"
    def test_linearization_op(self):
        poset = hasse.PoSet.from_chains([0,1,2])
        R_subposets = [
            hasse.PoSet.from_chains([0]),#orden global 0
            hasse.PoSet.from_chains([0,3],[1,4],[2,5],[0,4],[1,5],[2,3],[0,5],[1,3],[2,4]),#orden global 1,2,3
            hasse.PoSet.from_chains([0]),#orden global 7
        ]
        Y = [1,7,3,9,5,6,10,4]
        Y_map = {i: float(y) for i, y in enumerate(Y)}
        u = OGPAV(
            Q=poset,
            R_subposets=R_subposets,
            A=Y_map,
            use_trend_following_first=True,
            use_trend_following_blocks=True,
            inputs_are_reduced=False,
            verbose=True,
            segment_topo_orders=[[0],[2,1,0,3,4,5],[0]],
        )#[]
        v = OGPAV(
            Q=poset,
            R_subposets=R_subposets,
            A=Y_map,
            use_trend_following_first=True,
            use_trend_following_blocks=True,
            inputs_are_reduced=False,
            verbose=True,
            segment_topo_orders=[[0],[2,1,0,4,3,5],[0]],
        )#[]
        assert  np.array_equal(u,v), "error with user provided linearizations"