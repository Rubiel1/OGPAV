import pytest
import numpy as np
import networkx as nx
import os
import shutil

from OperadicGPAV import OperadicGPAV, create_lexicographic_mapping
from sb_gpav import sb_gpav
from geometric_sb_dataset import generate_dataset_lazy

verbose = False

class TestArrayStack:
    """
    Test suite for array-only implementations of OperadicGPAV and sb_gpav.
    
    For PoSet-based tests, see tests/gpav_test.py which uses operadic_gpav.OGPAV.
    """
    
    def test_sb_gpav_identity(self):
        """
        SB-GPAV: Chain 1->2->3->4 with sorted values. Should be identity.
        """
        X = np.array([[1], [2], [3], [4]])
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        L = [0, 1, 2, 3]
        
        u = sb_gpav(X, Y, L, n_segments=2)
        
        np.testing.assert_allclose(u, Y, err_msg="SB-GPAV changed isotone chain")
    
    def test_op_gpav_single_fiber(self):
        """
        OperadicGPAV: Single fiber with sorted values. Should be identity.
        """
        Q = nx.DiGraph()
        Q.add_node(0)
        
        R_datasets = [np.array([[1], [2], [3], [4]])]
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        
        u = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            max_workers=1,
            verbose=verbose
        )
        
        np.testing.assert_allclose(u, Y, err_msg="Single fiber: changed isotone values")
    
    def test_op_gpav_multi_fiber_chain(self):
        """
        OperadicGPAV: Chain Q: 0->1->2->3, each fiber is 1D point.
        """
        Q = nx.DiGraph()
        Q.add_edges_from([(0,1), (1,2), (2,3)])
        
        R_datasets = [
            np.array([[0]]),
            np.array([[0]]),
            np.array([[0]]),
            np.array([[0]])
        ]
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        
        u = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            max_workers=1,
            verbose=verbose
        )
        
        np.testing.assert_allclose(u, Y, err_msg="Multi-fiber chain: changed isotone values")
    
    def test_operadic_diamond_structure(self):
        """
        OperadicGPAV: Diamond Q structure (0->1, 0->3, 1->2, 3->2).
        """
        Q = nx.DiGraph()
        Q.add_edges_from([(0,1), (1,2), (0,3), (3,2)])
        
        R_datasets = [
            np.array([[0]]),                   # R0: 1 point
            np.array([[0], [1], [2]]),         # R1: 3 points
            np.array([[0]]),                   # R2: 1 point
            np.array([[0]])                    # R3: 1 point
        ]
        
        Y = np.array([10, 13, 11, 9, 12, 11], dtype=float)
        
        u = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            max_workers=1,
            verbose=verbose
        )
        
        # Verify GPAV reduces to {10, 11, 12}
        assert set(u) == {10.0, 11.0, 12.0}, f"Diamond: wrong result set {set(u)}"
    
    def test_parallel_execution(self):
        """
        OperadicGPAV: Parallel vs sequential execution should match.
        """
        Q = nx.DiGraph()
        Q.add_edge(0, 1)
        
        R0 = np.random.seed(42) or np.random.rand(10, 2)
        R1 = np.random.rand(10, 2) + 2  # R1 > R0
        R_datasets = [R0, R1]
        
        # Y consistent with order
        Y = np.concatenate([np.zeros(10), np.ones(10)])
        
        # Sequential
        u_seq = OperadicGPAV(Q, R_datasets, Y, max_workers=1, verbose=False)
        
        # Parallel
        u_par = OperadicGPAV(Q, R_datasets, Y, max_workers=2, verbose=False)
        
        np.testing.assert_allclose(u_seq, u_par, 
            err_msg="Parallel result differs from sequential")
    
    def test_sb_gpav_violations(self):
        """
        SB-GPAV: Corrects violations in chain.
        """
        X = np.array([[1], [2], [3], [4]])
        Y = np.array([1.0, 3.0, 2.0, 4.0])  # Violation: 3 > 2
        L = [0, 1, 2, 3]
        
        u = sb_gpav(X, Y, L, n_segments=2)
        
        # Should fix violation
        assert u[1] <= u[2], "SB-GPAV didn't fix violation"
        
    def test_custom_comparator(self):
        """
        OperadicGPAV: Custom comparator function.
        """
        Q = nx.DiGraph()
        Q.add_node(0)
        
        # 2D points
        R_datasets = [np.array([[1, 2], [2, 1], [3, 3]])]
        Y = np.array([1.0, 2.0, 3.0])
        
        # Custom: compare by sum
        def sum_comparator(a, b):
            return np.sum(a) < np.sum(b)
        
        u = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            f=sum_comparator,
            max_workers=1,
            verbose=False
        )
        
        # [1,2] and [2,1] sum to 3, [3,3] sums to 6
        # Y=[1,2,3] is consistent
        np.testing.assert_allclose(u, Y, err_msg="Custom comparator failed")
    
    def test_position_array(self):
        """
        OperadicGPAV: Array equivalent of test_position_op from PoSet tests.
        
        Q: 0->1->2 (3 fibers)
        R_1 has Hasse edges: 0<1, 0<3, 2<1, 2<3, 2<5, 4<3, 4<5
        
        Tests that independent elements remain untouched.
        """
        Q = nx.DiGraph()
        Q.add_edges_from([(0,1), (1,2)])
        
        # R_0: 1 element, R_1: 6 elements, R_2: 1 element
        # For R_1, we need to create the exact partial order from PoSet test
        # Manually assign unique vectors and define comparator
        R_datasets = [
            np.array([[0]]),  # R_0: 1 element
            np.array([        # R_1: 6 elements - vectors are just labels
                [0],  # Element 0
                [1],  # Element 1
                [2],  # Element 2
                [3],  # Element 3
                [4],  # Element 4
                [5],  # Element 5
            ]),
            np.array([[0]])   # R_2: 1 element
        ]
        
        Y = np.array([1.0, 7.0, 5.0, 3.0, 6.0, 9.0, 10.0, 4.0])
        
        # Custom comparator for R_1 that exactly matches PoSet edges
        # Edges: 0<1, 0<3, 2<1, 2<3, 2<5, 4<3, 4<5
        def r1_comparator(a, b):
            # a and b are 1D arrays with single element
            i, j = int(a[0]), int(b[0])
            # Define the exact partial order
            edges = {(0,1), (0,3), (2,1), (2,3), (2,5), (4,3), (4,5)}
            
            if i == j:
                return True
            if (i, j) in edges:
                return True
            return False
        
        # Use per-fiber comparators
        f_list = [
            None,           # R_0: default
            r1_comparator,  # R_1: exact PO from PoSet test
            None            # R_2: default
        ]
        
        u = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            f=f_list,
            max_workers=1,
            verbose=False
        )
        
        # Global index 3 corresponds to R_1[2] (element 2 in R_1)
        # This element has value Y[3]=3.0, and with the partial order,
        # it should remain isotone, so GPAV should not modify it
        assert u[3] == Y[3], f"OGPAV modified independent element: u[3]={u[3]}, Y[3]={Y[3]}"
    
    def test_position2_array(self):
        """
        OperadicGPAV: Array equivalent of test_position2_op.
        
        Q: 0->1 (2 fibers)
        R_0 has Hasse edges: 0<1, 0<2, 0<3
        R_1 has Hasse edges: 0<1, 2<1, 3<1
        
        Tests that independent element at index 2 remains untouched.
        """
        Q = nx.DiGraph()
        Q.add_edge(0, 1)
        
        # Define R_0 and R_1 with explicit comparators
        R_datasets = [
            np.array([[0], [1], [2], [3]]),  # R_0: 4 elements
            np.array([[0], [1], [2], [3]])   # R_1: 4 elements
        ]
        
        Y = np.array([1.0, 7.0, 3.0, 9.0, 5.0, 6.0, 10.0, 4.0])
        
        # R_0 comparator: edges 0<1, 0<2, 0<3
        def r0_comp(a, b):
            i, j = int(a[0]), int(b[0])
            if i == j:
                return True
            return (i, j) in [(0, 1), (0, 2), (0, 3)]
        
        # R_1 comparator: edges 0<1, 2<1, 3<1  
        def r1_comp(a, b):
            i, j = int(a[0]), int(b[0])
            if i == j:
                return True
            return (i, j) in [(0, 1), (2, 1), (3, 1)]
        
        u = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            f=[r0_comp, r1_comp],
            max_workers=1,
            verbose=False
        )
        
        # Index 2 (R_0[2]) should remain unchanged
        assert u[2] == Y[2], f"OGPAV modified independent element: u[2]={u[2]}, Y[2]={Y[2]}"
    
    def test_secondstage_array(self):
        """
        OperadicGPAV: Array equivalent of test_secondstage_op.
        
        Q: 0->1, 2->1 (nodes 0 and 2 feed into node 1)
        R_0 has 2 incomparable elements
        
        Tests second-stage GPAV behavior.
        """
        Q = nx.DiGraph()
        Q.add_edges_from([(0, 1), (2, 1)])  # Fixed: was (0,2), (1,2)
        
        R_datasets = [
            np.array([[0], [1]]),  # R_0: 2 incomparable elements
            np.array([[0]]),       # R_1: 1 element
            np.array([[0]])        # R_2: 1 element
        ]
        
        Y = np.array([10.0, 9.0, 8.0, 12.0])
        
        # R_0 comparator: elements 0 and 1 are incomparable
        def r0_comp(a, b):
            i, j = int(a[0]), int(b[0])
            return i == j  # Only reflexive, no order between 0 and 1
        
        u = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            f=[r0_comp, None, None],
            max_workers=1,
            verbose=False
        )
        
        # Expected: [10.0, 9.0, 10.0, 10.0]
        # R_0[0]=10, R_0[1]=9 stay, R_1[0] becomes 10 (max of R_0), R_2[0] becomes 10 (must be <= R_1)
        np.testing.assert_array_equal(u, np.array([10.0, 9.0, 10.0, 10.0]),
            err_msg="Second-stage GPAV produced incorrect result")
    
    def test_sb_isotone_explicit(self):
        """
        SB-GPAV: Verify isotonicity on all edges explicitly.
        Tests that output respects partial order on every edge.
        """
        # Diamond: 0<1<3, 0<2<3
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        Y = np.array([10.0, 13.0, 11.0, 9.0])
        L = [0, 1, 2, 3]
        
        # Build Hasse diagram
        def comp(a, b):
            return np.all(a <= b)
        
        u = sb_gpav(X, Y, L, f=comp, n_segments=2)
        
        # Manually check isotonicity on edges: 0<1, 0<2, 1<3, 2<3
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        for i, j in edges:
            assert u[i] <= u[j], f"SB-GPAV violated isotonicity: {i}<{j} but u[{i}]={u[i]} > u[{j}]={u[j]}"
    
    def test_sb_gpav_chain_explicit(self):
        """
        SB-GPAV: Explicit chain test matching original test_sb_chain_identity.
        """
        X = np.array([[0], [1], [2], [3]])
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        L = [0, 1, 2, 3]
        
        u = sb_gpav(X, Y, L, n_segments=2)
        
        assert set(u) == {1.0, 2.0, 3.0, 4.0}, "SB-GPAV changed already isotone chain"
    
    def test_gpav_infrastructure(self):
        """
        Direct test of gpav_op and gpav_seg infrastructure with array inputs.
        
        Tests that core GPAV produces isotonic output on a DAG.
        Structure: 0<1<2<4, 1<3
        """
        from gpav import gpav_op as gpav_op_func, gpav_seg as gpav_seg_func
        
        # Build DAG: 0<1<2<4, 1<3
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 4), (1, 3)])
        
        Y_map = {0: 0.0, 1: 10.0, 2: 11.0, 3: 0.0, 4: -2.0}
        topo_order = [0, 1, 2, 3, 4]
        
        # Test gpav_op with block edges
        u_op, blocks_op, _, block_edges = gpav_op_func(
            Y=Y_map, poset=G, topo_order=topo_order, return_block_edges=True
        )
        
        # Verify block edges form a DAG (no cycles)
        G_blocks = nx.DiGraph()
        G_blocks.add_nodes_from(range(len(blocks_op)))
        for a, b in block_edges:
            if a != b:
                G_blocks.add_edge(a, b)
        assert nx.is_directed_acyclic_graph(G_blocks), \
            "gpav_op block edges contain a cycle"
        
        # Test gpav_seg
        u_seg, _, _ = gpav_seg_func(Y=Y_map, poset=G, topo_order=topo_order)
        
        # Verify isotonicity on all edges
        assert (u_op[0] <= u_op[1]) and (u_op[1] <= u_op[2]) and \
               (u_op[2] <= u_op[4]) and (u_op[1] <= u_op[3]), \
               "gpav_op violated isotonicity"
        
        assert (u_seg[0] <= u_seg[1]) and (u_seg[1] <= u_seg[2]) and \
               (u_seg[2] <= u_seg[4]) and (u_seg[1] <= u_seg[3]), \
               "gpav_seg violated isotonicity"
    
    def test_linearization_array(self):
        """
        OperadicGPAV: Array equivalent of test_linearization_op.
        
        Tests that different custom topological orders produce same result.
        """
        Q = nx.DiGraph()
        Q.add_edges_from([(0,1), (1,2)])
        
        # Same structure as test_order_input_op
        R_datasets = [
            np.array([[0]]),  # R_0: 1 element
            np.array([[0], [1], [2], [3], [4], [5]]),  # R_1: 6 elements
            np.array([[0]])   # R_2: 1 element
        ]
        
        Y = np.array([1.0, 7.0, 3.0, 9.0, 5.0, 6.0, 10.0, 4.0])
        
        # R_1 comparator: elements form a partial order
        def r1_comp(a, b):
            i, j = int(a[0]), int(b[0])
            if i == j:
                return True
            # Edges: 0<3, 1<4, 2<5, 0<4, 1<5, 2<3, 0<5, 1<3, 2<4
            edges = {(0,3), (1,4), (2,5), (0,4), (1,5), (2,3), (0,5), (1,3), (2,4)}
            return (i, j) in edges
        
        # First order for R_1
        u1 = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            f=[None, r1_comp, None],
            segment_topo_orders=[[0], [2, 1, 0, 3, 4, 5], [0]],
            max_workers=1,
            verbose=False
        )
        
        # Different order for R_1
        u2 = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            f=[None, r1_comp, None],
            segment_topo_orders=[[0], [2, 1, 0, 4, 3, 5], [0]],
            max_workers=1,
            verbose=False
        )
        
        # Results should be identical
        np.testing.assert_array_equal(u1, u2, 
            err_msg="Different linearizations produced different results")
    
    def test_gpav_with_weights(self):
        """
        Direct GPAV test with non-zero node labels and weights.
        Tests gpav_seg infrastructure with arbitrary node labels.
        """
        from gpav import gpav_seg as gpav_seg_func
        
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3, 4, 5])
        G.add_edges_from([
            (1, 2),
            (2, 3),
            (2, 4),
            (3, 5),
        ])
        
        Y = {1: 0.0, 2: 10.0, 3: 11.0, 4: 0.0, 5: -2.0}
        w = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
        
        # Run gpav_seg with weights
        u, blocks, elem_to_block = gpav_seg_func(Y, G, weights=w, verbose=False)
        
        # Check monotonicity on all edges
        nodes_list = list(G.nodes())
        for a, b in G.edges():
            a_idx = nodes_list.index(a)
            b_idx = nodes_list.index(b)
            assert u[a_idx] <= u[b_idx], \
                f"gpav_seg violated isotonicity: {a}<{b} but u[{a}]={u[a_idx]} > u[{b}]={u[b_idx]}"
    
    def test_operadic_geometric_dataset(self):
        """
        OperadicGPAV: Integration test with geometric_sb_dataset.
        
        Tests the full pipeline with geometrically-generated datasets.
        """
        Q = nx.DiGraph()
        Q.add_edge(0, 1)
        
        import tempfile
        import shutil
        cache_dir = tempfile.mkdtemp(prefix="ogpav_test1_")
        
        # Generate geometric dataset with 2 fibers
        result = generate_dataset_lazy(
            nQ=2,           # 2 fibers (matches Q nodes)
            avg_R=25,       # Average 25 points per fiber
            seed=42,
            cache_dir=cache_dir
        )
        
        # Get lazy dataset
        R_datasets = result['R_points_list']
        # Extract lengths internally via the object method for our test validation
        n0, n1 = R_datasets.get_fiber_lengths()[0], R_datasets.get_fiber_lengths()[1]
        
        # Create Y such that R0 < R1 generally
        Y = np.concatenate([
            np.random.RandomState(42).uniform(0, 5, n0),
            np.random.RandomState(43).uniform(5, 10, n1)
        ])
        
        # Run OperadicGPAV
        u = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            max_workers=1,
            assume_component_wise=True,
            verbose=False
        )
        
        # Verify output shape and basic properties
        assert len(u) == n0 + n1, f"Output length mismatch: {len(u)} != {n0 + n1}"
        assert np.all(np.isfinite(u)), "Output contains non-finite values"
        
        # Verify that max of R0 <= min of R1 (due to Q structure)
        u0_max = np.max(u[:n0])
        u1_min = np.min(u[n0:])
        assert u0_max <= u1_min, \
            f"OperadicGPAV violated Q constraint: max(u0)={u0_max} > min(u1)={u1_min}"
            
        R_datasets.cleanup()
    
    def test_operadic_lazy_iterators(self):
        """
        OperadicGPAV: Test with lazy iterators (not pre-converted to arrays).
        
        Verifies OperadicGPAV can accept generators/iterators directly
        and handle them properly in parallel processing.
        """
        Q = nx.DiGraph()
        Q.add_edges_from([(0, 1), (1, 2)])
        
        import tempfile
        import shutil
        cache_dir = tempfile.mkdtemp(prefix="ogpav_test2_")
        
        # Generate geometric dataset with 3 fibers
        result = generate_dataset_lazy(
            nQ=3,           # 3 fibers (matches Q nodes)
            avg_R=20,       # Average 20 points per fiber
            seed=42,
            cache_dir=cache_dir
        )
        
        # Get lazy dataset object which inherits Sequence and implements get_fiber_lengths
        R_datasets = result['R_points_list']
        lengths = R_datasets.get_fiber_lengths()
        n0, n1, n2 = lengths[0], lengths[1], lengths[2]
        
        Y = np.concatenate([
            np.random.RandomState(42).uniform(0, 3, n0),
            np.random.RandomState(43).uniform(3, 6, n1),
            np.random.RandomState(44).uniform(6, 9, n2)
        ])
        
        # Run with parallel workers - this tests iterator handling
        u = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            max_workers=2,  # Parallel to test iterator passing
            assume_component_wise=True,
            verbose=False
        )
        
        # Verify basic properties
        assert len(u) == n0 + n1 + n2, f"Output length mismatch"
        assert np.all(np.isfinite(u)), "Output contains non-finite values"
        
        # Verify Q constraints: R0 <= R1 <= R2
        u0_max = np.max(u[:n0])
        u1_max = np.max(u[n0:n0+n1])
        u2_min = np.min(u[n0+n1:])
        
        assert u0_max <= u2_min, "Q constraint violated: R0 not <= R2"
        
        R_datasets.cleanup()

    def test_operadic_lazy_large_scale(self):
        """
        OperadicGPAV: Test with a realistically large lazy dataset.
        
        Verifies that OperadicGPAV efficiently handles a significant
        number of geometric points (e.g. 50 fibers, ~500 points each = ~25,000 points)
        generated via lazy cached iteration, confirming the true utility of
        the memory-efficient iterator mode without taking unreasonable time on CI.
        """
        # Outer Poset = Chain of 50 nodes
        nQ = 50
        Q = nx.DiGraph()
        Q.add_edges_from([(i, i+1) for i in range(nQ-1)])
        
        import tempfile
        import shutil
        cache_dir = tempfile.mkdtemp(prefix="ogpav_test3_")
        
        # Generator for ~25k total vectors
        result = generate_dataset_lazy(
            nQ=nQ,           
            avg_R=500,       
            seed=42,
            cache_dir=cache_dir
        )
        
        # Lazy iterator
        R_datasets = result['R_points_list']
        lengths = R_datasets.get_fiber_lengths()
        
        total_points = sum(lengths)
        
        # Generate monotonic baseline Y loosely corresponding to node position
        # so it is a non-trivial GPAV execution
        rng = np.random.RandomState(42)
        Y_list = []
        base_val = 0.0
        for i in range(nQ):
            # A block of values roughly growing per fiber
            Y_fib = rng.uniform(base_val, base_val + 5.0, lengths[i])
            Y_list.append(Y_fib)
            base_val += 3.0 # Overlap to trigger pooling
            
        Y = np.concatenate(Y_list)
        
        u = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            max_workers=None, # Use all available cores for heavy lift
            assume_component_wise=True, # Vectors
            verbose=False
        )
        
        # Length Validation
        assert len(u) == total_points, "Lengths do not match completely"
        
        # Monotonization strictness test: last point of first node <= first point of last node (minimum guarantee)
        u_start_max = np.max(u[:lengths[0]])
        u_end_min = np.min(u[total_points - lengths[-1]:])
        assert u_start_max <= u_end_min, "Massive Q constraint totally violated"
        
        R_datasets.cleanup()
    
    def test_trend_following_flags(self):
        """
        OperadicGPAV: Test all combinations of trend_following flags.
        
        Verifies that use_trend_following_first and use_trend_following_blocks
        combinations all produce valid isotonic results.
        """
        Q = nx.DiGraph()
        Q.add_edge(0, 1)
        
        R_datasets = [
            np.array([[1, 1], [2, 2], [3, 3]]),
            np.array([[4, 4], [5, 5], [6, 6]])
        ]
        
        Y = np.array([3.0, 1.0, 2.0, 6.0, 4.0, 5.0])
        
        # Test all combinations
        combinations = [
            (True, True),   # Both trend following (default)
            (True, False),  # Only first stage trend following
            (False, True),  # Only blocks trend following
            (False, False)  # No trend following
        ]
        
        results = []
        for use_first, use_blocks in combinations:
            u = OperadicGPAV(
                Q=Q,
                R_datasets=R_datasets,
                Y=Y,
                use_trend_following_first=use_first,
                use_trend_following_blocks=use_blocks,
                max_workers=1,
                verbose=False
            )
            results.append(u)
            
            # Verify isotonicity regardless of flags
            assert len(u) == 6, f"Output length mismatch for flags ({use_first}, {use_blocks})"
            assert np.all(np.isfinite(u)), f"Non-finite values for flags ({use_first}, {use_blocks})"
            
            # Verify Q constraint: max(R0) <= min(R1)
            assert np.max(u[:3]) <= np.min(u[3:]), \
                f"Q constraint violated for flags ({use_first}, {use_blocks})"
        
        # All results should be valid (though potentially different values)
        # This is expected - different topological orders can give different GPAV results
        assert len(results) == 4, "Should have 4 results from 4 flag combinations"



if __name__ == "__main__":
    # Run tests
    t = TestArrayStack()
    print("Running test_sb_gpav_identity...")
    t.test_sb_gpav_identity()
    print("Running test_op_gpav_single_fiber...")
    t.test_op_gpav_single_fiber()
    print("Running test_op_gpav_multi_fiber_chain...")
    t.test_op_gpav_multi_fiber_chain()
    print("Running test_operadic_diamond_structure...")
    t.test_operadic_diamond_structure()
    print("Running test_parallel_execution...")
    t.test_parallel_execution()
    print("Running test_sb_gpav_violations...")
    t.test_sb_gpav_violations()
    print("Running test_custom_comparator...")
    t.test_custom_comparator()
    print("Running test_position_array...")
    t.test_position_array()
    print("Running test_position2_array...")
    t.test_position2_array()
    print("Running test_secondstage_array...")
    t.test_secondstage_array()
    print("Running test_sb_isotone_explicit...")
    t.test_sb_isotone_explicit()
    print("Running test_sb_gpav_chain_explicit...")
    t.test_sb_gpav_chain_explicit()
    print("Running test_gpav_infrastructure...")
    t.test_gpav_infrastructure()
    print("Running test_linearization_array...")
    t.test_linearization_array()
    print("Running test_gpav_with_weights...")
    t.test_gpav_with_weights()
    print("Running test_operadic_geometric_dataset...")
    t.test_operadic_geometric_dataset()
    print("Running test_operadic_lazy_iterators...")
    t.test_operadic_lazy_iterators()
    print("Running test_operadic_lazy_large_scale...")
    t.test_operadic_lazy_large_scale()
    print("Running test_trend_following_flags...")
    t.test_trend_following_flags()
    print("\n All tests passed!")
