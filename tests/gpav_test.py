from trend_following import trend_following_order
from operadic_gpav import OGPAV, construct_lexicographic_sum_dag
from gpav import gpav_op, gpav_seg
import networkx as nx
import numpy as np
import hasse
from sb_gpav_paper import sb_gpav

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
        u, _, _, _ = gpav_op(Y_aligned, G, topo_order=T, verbose=True)
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
    def test_gpavs_op(self):
        posetcito = hasse.PoSet.from_chains([0,1,2,4],[1,3])
        Y = [0,10,11,0,-2]
        Y_map = {i: float(y) for i, y in enumerate(Y)}
        ordentopo= [0,1,2,3,4]
        u,_,_,_ = gpav_op(Y=Y_map,
                    poset=posetcito,
                    topo_order= ordentopo,
                    )#[]
        v,_,_ = gpav_seg(Y=Y_map,
                     poset=posetcito,
                     topo_order = ordentopo
        )#[]
        assert  (u[0]<=u[1])&(u[1]<=u[2])&(u[2]<=u[4])&(u[1]<=u[3]), "error in GPAV operadic code"
    def test_local_input_op(self):
        Q = hasse.PoSet.from_chains([0, 1], [0, 2])
        R_subposets = [
            hasse.PoSet.from_chains([0, 1]),        # R_0
            hasse.PoSet.from_chains([0], [1]),      # R_1
            hasse.PoSet.from_chains([0, 1, 2]),     # R_2
            ]
        A_list = [
            {0: 3.0, 1: 1.0},           # data on R_0
            {0: 2.0, 1: 4.0},           # data on R_1
            {0: 1.0, 1: 2.0, 2: 3.0},   # data on R_2
            ]
        u_list = OGPAV(
        Q=Q,
        R_subposets=R_subposets,
        A=None,                     # ignored
        A_list=A_list,
        return_by_local_index=True,       # if you added this option
        verbose=False,
        )
        assert (u_list[0][1] <= u_list[2][0])&(u_list[0][1] <= u_list[1][0])&(u_list[0][1] <= u_list[1][1]), "error in GPAV when list of Ys is given"
    def test_sb_chain_identity(self):
        G = nx.DiGraph()
        G.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])
        Y_map = {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0}
        u = sb_gpav(
            G, Y_map,
            n_segments=2,
            use_trend_following_first=True,
            use_trend_following_blocks=True,
            verbose=False,
            return_by_label=False,
        )
        assert set(u) == {1.0, 2.0, 3.0, 4.0}, "SB-GPAV changed an already isotone chain"
    def test_sb_order_invariance(self):

        # Same abstract poset, but build using different chain input orders
        poset1 = hasse.PoSet.from_chains([0,1,2], [0,3,2])
        poset2 = hasse.PoSet.from_chains([0,3,2], [0,1,2])  # swapped chain order

        Y = [10, 13, 11, 9]
        Y_map = {i: float(y) for i, y in enumerate(Y)}

        # Compare by label using return_by_label=True to avoid internal node-order issues
        m1 = sb_gpav(poset1, Y_map, n_segments=2)
        m2 = sb_gpav(poset2, Y_map, n_segments=2)

        assert m1 == m2, "SB-GPAV depends on construction order (label mismatch)"
    def test_sb_isotone_output(self):

        poset = hasse.PoSet.from_chains([0,1,2], [0,3,2])  # 0<1<2 and 0<3<2
        Y = [10, 13, 11, 9]
        Y_map = {i: float(y) for i, y in enumerate(Y)}

        u_map = sb_gpav(
            poset, Y_map,
            n_segments=2,
            verbose=False,
        )

        # Check isotonicity on all Hasse edges (covers):
        H = poset.hasse  # assuming hasse.PoSet exposes .hasse as a DiGraph
        
        for a, b in H.edges():
            assert u_map[a] <= u_map[b], f"SB-GPAV violated isotonicity on edge {a}<{b}"
    def test_sb_red(self):

        G = hasse.PoSet.from_chains([0,1,2], [0,3,2])  # 0<1<2 and 0<3<2
        Y = [10, 13, 11, 9]
        Y_map = {i: float(y) for i, y in enumerate(Y)}
        u1 = sb_gpav(G, Y_map, inputs_are_reduced=True)
        u2 = sb_gpav(G, Y_map, inputs_are_reduced=False)
        assert u1 == u2, "the reduction flag has a problem"
    def test_R_op(self):
        H_Q = nx.DiGraph()
        H_Q.add_edge(0, 1)

        # Local block DAG for R1: two blocks 0 < 1
        G1 = nx.DiGraph()
        G1.add_edge(0, 1)

        # Local block DAG for R2: three blocks 0 < 1 < 2
        G2 = nx.DiGraph()
        G2.add_edge(0, 1)
        G2.add_edge(1, 2)

        G_loc_list = [G1, G2]

        # Min/max blocks (local labels, overlapping on purpose)
        group_min = [[0], [0]]
        group_max = [[1], [2]]

        # Call with auto-relabeling enabled
        G_B, maps = construct_lexicographic_sum_dag(
          H_Q,
          G_loc_list,
          group_min_global=group_min,
          group_max_global=group_max,
          relabel_if_needed=True,
          return_relabel_maps=True,
          verbose=False,
        )

        assert set(maps[0].values())& set(maps[1].values()) ==set(), "R_i's still have same labels"
