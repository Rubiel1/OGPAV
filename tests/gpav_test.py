import pytest
import numpy as np
import networkx as nx
import os
import shutil
import sys
import time
import warnings

from OperadicGPAV import OperadicGPAV, create_lexicographic_mapping
from utils.sb_gpav import sb_gpav
from utils.geometric_sb_dataset import generate_dataset_lazy
from utils.trend_following import (
    _build_dag_incrementally,
    default_comparator,
    trend_following_order,
)

verbose = False


 
# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
  
def _chain_Q(m):
    Q = nx.DiGraph()
    Q.add_nodes_from(range(m))
    Q.add_edges_from((i, i + 1) for i in range(m - 1))
    return Q
 
 
def _full_poset(Q, R, f=default_comparator):
    """The lexicographic sum Q(R_0,...,R_{m-1}) on global indices, strict edges only."""
    offs, c = [0], 0
    for r in R:
        c += len(r)
        offs.append(c)
    G = nx.DiGraph()
    G.add_nodes_from(range(c))
    for i, r in enumerate(R):
        for a in range(len(r)):
            for b in range(len(r)):
                if a != b and f(r[a], r[b]) and not f(r[b], r[a]):
                    G.add_edge(offs[i] + a, offs[i] + b)
    if Q.number_of_edges():
        TC = nx.transitive_closure_dag(nx.transitive_reduction(Q))
        for i, j in TC.edges():
            for a in range(len(R[i])):
                for b in range(len(R[j])):
                    G.add_edge(offs[i] + a, offs[j] + b)
    return G
 
 
def _violations(G, u, tol=1e-9):
    return [(a, b) for a, b in G.edges() if u[a] > u[b] + tol]

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
            assume_component_wise=True,
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
            assume_component_wise=True,
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
            assume_component_wise=True,
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
        u_seq = OperadicGPAV(Q, R_datasets, Y, assume_component_wise=True, max_workers=1, verbose=False)
        
        # Parallel
        u_par = OperadicGPAV(Q, R_datasets, Y, assume_component_wise=True, max_workers=2, verbose=False)
        
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
        
        from utils.trend_following import default_comparator
        # Use per-fiber comparators
        f_list = [
            default_comparator,  # R_0: default
            r1_comparator,       # R_1: exact PO from PoSet test
            default_comparator   # R_2: default
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
        
        from utils.trend_following import default_comparator
        u = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            f=[r0_comp, default_comparator, default_comparator],
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
        from utils.gpav import gpav_op as gpav_op_func, gpav_seg as gpav_seg_func
        
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
        
        from utils.trend_following import default_comparator
        # First order for R_1
        u1 = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            f=[default_comparator, r1_comp, default_comparator],
            segment_topo_orders=[[0], [2, 1, 0, 3, 4, 5], [0]],
            max_workers=1,
            verbose=False
        )
        
        # Different order for R_1
        u2 = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=Y,
            f=[default_comparator, r1_comp, default_comparator],
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
        from utils.gpav import gpav_seg as gpav_seg_func
        
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
        
        assert len(results) == 4, "Should have 4 results from 4 flag combinations"

    def test_operadic_vs_sb_geometric_match(self):
        """
        Integration: OperadicGPAV and SB-GPAV should produce extremely similar results
        on a geometric dataset where the global poset Q precisely matches the 
        component-wise dominance order that SB-GPAV uses globally.
        """
        import tempfile
        import math
        
        # 1. Generate a small but non-trivial geometric dataset
        nQ = 20
        avg_R = 15
        
        cache_dir = tempfile.mkdtemp(prefix="ogpav_test_match_")
        data = generate_dataset_lazy(
            nQ=nQ,
            avg_R=avg_R,
            seed=123,
            cache_dir=cache_dir,
            square_max = 100,
        )
        
        R_datasets = data['R_points_list']
        Q = data['Q_hasse']
        
        # Ensure Q matches dataset length
        m = len(R_datasets.get_fiber_lengths())
        if Q.number_of_nodes() != m:
            mapping = {old: new for new, old in enumerate(sorted(Q.nodes()))}
            Q = nx.relabel_nodes(Q, mapping)
        
        # Build global X array
        lengths = R_datasets.get_fiber_lengths()
        total_n = sum(lengths)
        
        parts = []
        for i in range(m):
            parts.append(np.asarray(R_datasets[i]))
        X = np.concatenate(parts, axis=0)
        
        # Generate monotonic observations with moderate noise
        from utils.geometric_sb_dataset import make_observations
        y_noisy = make_observations(X, model="nonlinear", noise="normal", noise_scale=1.0, seed=42)
        
        # 2. Run OperadicGPAV
        u_ogpav = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=y_noisy,
            assume_component_wise=True,
            max_workers=1,
            verbose=False
        )
        
        # 3. Run SB-GPAV
        # Topo order by sum of coordinates
        L = list(np.argsort(np.sum(X, axis=1)))
        
        u_sb = sb_gpav(
            X=X,
            Y=y_noisy,
            L=L,
            n_segments=max(1, math.ceil(total_n / 50)),
            assume_component_wise=True,
            verbose=False
        )
        
        # 4. Compare their results 
        # Using MSE to ensure they are extremely close
        mse = np.mean((u_ogpav - u_sb) ** 2)
        
        assert mse < 1e-5, f"OGPAV and SB-GPAV diverge significantly: MSE = {mse}"
        
        R_datasets.cleanup()



    def test_dataset_loaders(self):
        """
        Tests the behavior of dataset.py CustomFiberDataset:
        - Alphabetic directory parsing
        - Zip archive explicit mapping (including length swapping)
        """
        from utils.dataset import CustomFiberDataset
        import zipfile
        import tempfile

        cache_dir = tempfile.mkdtemp(prefix="ogpav_test_loaders_")
        dump_dir = os.path.join(cache_dir, 'dummy_data')
        os.makedirs(dump_dir, exist_ok=True)
        
        # Fiber 1 = 10 rows. Fiber 0 = 5 rows.
        np.save(os.path.join(dump_dir, 'fiber_1.npy'), np.ones((10, 2)))
        np.save(os.path.join(dump_dir, 'fiber_0.npy'), np.zeros((5, 2)))

        zip_path = os.path.join(cache_dir, 'dummy.zip')
        with zipfile.ZipFile(zip_path, 'w') as z:
            z.write(os.path.join(dump_dir, 'fiber_0.npy'), 'fiber_0.npy')
            z.write(os.path.join(dump_dir, 'fiber_1.npy'), 'fiber_1.npy')

        # Test 1: Alphabetic Sorting (Expects warning, sorts 0 then 1)
        with pytest.warns(UserWarning, match="Defaulting to sorting files alphabetically"):
            ds1 = CustomFiberDataset(dump_dir)
            lengths1 = ds1.get_fiber_lengths()
            assert lengths1 == [5, 10], f"Alphabetic sort failed. Got lengths: {lengths1}"

        # Test 2: Mapped Zip Archive (Node 0 -> fiber 1, Node 1 -> fiber 0)
        ds2 = CustomFiberDataset(zip_path, node_to_file={0: 'fiber_1.npy', 1: 'fiber_0.npy'})
        lengths2 = ds2.get_fiber_lengths()
        
        assert lengths2 == [10, 5], f"Mapped sort failed. Got mapped lengths: {lengths2}"
        assert ds2[0].shape == (10, 2), "Fiber 0 did not map correctly"
        assert ds2[1].shape == (5, 2), "Fiber 1 did not map correctly"

        shutil.rmtree(cache_dir)

    def test_border_cases(self):
        """
        Tests the border case optimizations in OperadicGPAV:
        1. Q with no edges (m > 1) completely bypassing Stage 2.
        2. R_i Local Antichain where graph has strictly 0 edges.
        3. Multiprocessing lambda picklability fallback.
        """
        # Test 1: Q without edges (Stage 2 bypass)
        Q_no_edges = nx.DiGraph()
        Q_no_edges.add_nodes_from([0, 1, 2])
        
        R_no_edges = [
            np.array([[0], [1]]),
            np.array([[2], [3]]),
            np.array([[4], [5]])
        ]
        # Y values map exactly to nodes 0 and 1 since there's no interactions
        # Local GPAV forces isotonicity on coordinate values locally
        # Since we use default geometric on 1D representations:
        # Fiber 0: [0] <= [1] -> Y[0] <= Y[1] (4.0 > 1.0 -> pools to 2.5)
        # Fiber 1: [2] <= [3] -> Y[2] <= Y[3] (5.0 > 2.0 -> pools to 3.5)
        # Fiber 2: [4] <= [5] -> Y[4] <= Y[5] (6.0 <= 7.0 -> is already isotonic, no pooling!)
        Y_no_edges = np.array([4.0, 1.0, 5.0, 2.0, 6.0, 7.0])
        u1 = OperadicGPAV(
            Q=Q_no_edges, 
            R_datasets=R_no_edges, 
            Y=Y_no_edges, 
            assume_component_wise=True, # Critical: Forces geometric construction so it actually pools 4.0 and 1.0!
            max_workers=1, 
            verbose=False
        )
        np.testing.assert_allclose(u1, [2.5, 2.5, 3.5, 3.5, 6.0, 7.0], err_msg="Q without edges fast-path failed")

        # Test 2: Local Antichain with Lambda Fallback
        Q_chain = nx.DiGraph([(0, 1)]) # Forces global polling
        
        # A lambda function strictly forces NO local edges (False).
        # It also triggers the max_workers picklability fallback because it's a lambda!
        f_antichain = lambda a, b: False
        
        # Fiber 0: [5.0, 3.0] are disjoint.
        # Fiber 1: [4.0, 1.0] are disjoint.
        # Global polling: Fiber 0 must be <= Fiber 1.
        # Max of Fiber 0 = 5.0. Min of Fiber 1 = 1.0. 
        # They will instantly pool globally.
        Y_antichain = np.array([5.0, 3.0, 4.0, 1.0])
        R_antichain = [np.array([[0],[0]]), np.array([[0],[0]])]
        
        with pytest.warns(UserWarning, match="cannot be pickled"):
            u2 = OperadicGPAV(
                Q=Q_chain, 
                R_datasets=R_antichain, 
                Y=Y_antichain, 
                f=f_antichain, 
                max_workers=4, # Request parallel to test downgrade
                verbose=False
            )
            
        # Due to 5 block interacting with 1 block over the Q-edge, they merge.
        assert np.max(u2[:2]) <= np.min(u2[2:]), "Global Q constraint violated during antichain processing"



# ---------------------------------------------------------------------
# Defect 1 -- ties are reported, not raised as an opaque NetworkXError
# ---------------------------------------------------------------------
 
class TestTiesAreDiagnosed:
    """`f` must be antisymmetric on distinct elements.  Both ways it can fail
    should produce a message naming the cause, not a bare transitive_reduction
    error from deep inside networkx."""
 
    def test_duplicate_rows_name_the_fiber(self):
        # Elements 0 and 1 are the SAME point -> R_0 is not a set.
        R = [np.array([[2., 5.], [2., 5.], [4., 7.]]), np.array([[9., 9.]])]
        Y = np.array([1., 5., 9., 20.])          # keeps the copies in separate blocks
        with pytest.raises(ValueError, match=r"repeated elements"):
            OperadicGPAV(Q=nx.DiGraph([(0, 1)]), R_datasets=R, Y=Y,
                         assume_component_wise=True, max_workers=1)
 
    def test_duplicate_rows_are_caught_regardless_of_Y(self):
        # Same malformed input, but Y would have let GPAV pool the copies.
        # Whether the old code crashed or returned a number was an accident of Y;
        # if this ever stops raising, the pooled answer must at least be consistent.
        R = [np.array([[2., 5.], [2., 5.], [4., 7.]]), np.array([[9., 9.]])]
        Y = np.array([5., 1., 9., 20.])
        try:
            u = OperadicGPAV(Q=nx.DiGraph([(0, 1)]), R_datasets=R, Y=Y,
                             assume_component_wise=True, max_workers=1)
        except ValueError:
            return                                 # rejected up front: also fine
        assert u[0] == pytest.approx(u[1]), "identical rows must get identical fits"
 
    def test_non_antisymmetric_comparator_is_named(self):
        # (3,1) and (1,3) are DISTINCT rows, but by_total ties them.
        # The input data is fine; the comparator is not a partial order.
        def by_total(a, b):
            return float(np.sum(a)) <= float(np.sum(b))
 
        R = [np.array([[3., 1.], [1., 3.], [5., 5.]]), np.array([[9., 9.]])]
        Y = np.array([1., 5., 9., 20.])
        with pytest.raises((ValueError, nx.NetworkXError), match=r"antisymmetric"):
            OperadicGPAV(Q=nx.DiGraph([(0, 1)]), R_datasets=R, Y=Y,
                         f=by_total, max_workers=1)
 
    @pytest.mark.parametrize("R0,kw", [
        (np.array([[2., 5.], [2., 5.0001], [4., 7.]]), dict(assume_component_wise=True)),
        (np.array([[3., 2.], [1., 3.], [5., 5.]]),
         dict(f=lambda a, b: float(np.sum(a)) <= float(np.sum(b)))),
    ])
    def test_controls_still_run(self, R0, kw):
        """Same shapes with the tie removed must be unaffected."""
        u = OperadicGPAV(Q=nx.DiGraph([(0, 1)]),
                         R_datasets=[R0, np.array([[9., 9.]])],
                         Y=np.array([1., 5., 9., 20.]), max_workers=1, **kw)
        assert np.all(np.isfinite(u))
 
    def test_binned_data_no_longer_crashes_opaquely(self):
        """Integer-coded coordinates used to raise NetworkXError ~75% of the time."""
        rng = np.random.default_rng(0)
        opaque = 0
        for _ in range(25):
            R = [rng.integers(0, 10, size=(40, 2)).astype(float), np.array([[99., 99.]])]
            Y = np.concatenate([rng.normal(size=40), [100.]])
            try:
                OperadicGPAV(Q=nx.DiGraph([(0, 1)]), R_datasets=R, Y=Y,
                             assume_component_wise=True, max_workers=1)
            except (ValueError, nx.NetworkXError) as e:
                if "transitive_reduction" in str(e) and "antisymmetric" not in str(e):
                    opaque += 1
        assert opaque == 0, "a bare transitive_reduction error still reaches the user"
 
 
# ---------------------------------------------------------------------
# Defect 2 -- f=None means ANTICHAIN.  Pin the semantic that was chosen.
# ---------------------------------------------------------------------
 
class TestFNoneIsAnAntichain:
    """The audit found the docstring and the code disagreed.  The code won:
    f=None asserts NO order.  Nothing in the original suite pinned this, so a
    refactor could silently flip it.  These tests pin it."""
 
    CHAIN = [np.array([[0., 0.], [1., 1.], [2., 2.]]), np.array([[3., 3.]])]
    Y = np.array([9., 1., 2., 5.])
 
    def test_f_none_does_not_order_the_fiber(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = OperadicGPAV(Q=nx.DiGraph([(0, 1)]), R_datasets=self.CHAIN,
                             Y=self.Y, max_workers=1)
        # elements 1 and 2 are untouched: no intra-fiber order was asserted
        assert u[1] == pytest.approx(1.0)
        assert u[2] == pytest.approx(2.0)
        # ...and that is NOT monotone w.r.t. coordinate-wise dominance, by design
        G = _full_poset(nx.DiGraph([(0, 1)]), self.CHAIN)
        assert _violations(G, u), "f=None should not be enforcing the geometric order"
 
    def test_assume_component_wise_does_order_the_fiber(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = OperadicGPAV(Q=nx.DiGraph([(0, 1)]), R_datasets=self.CHAIN,
                             Y=self.Y, assume_component_wise=True, max_workers=1)
        G = _full_poset(nx.DiGraph([(0, 1)]), self.CHAIN)
        assert not _violations(G, u)
        np.testing.assert_allclose(u, [4., 4., 4., 5.])
 
    def test_none_inside_a_list_is_also_an_antichain(self):
        """README's f=[None, custom] pattern: position 0 asserts no order."""
        R = [np.array([[0.], [1.], [2.]]), np.array([[0., 0.], [1., 1.]])]
        Y = np.array([3., 1., 2., 4., 5.])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = OperadicGPAV(Q=nx.DiGraph([(0, 1)]), R_datasets=R, Y=Y,
                             f=[None, lambda a, b: float(np.sum(np.abs(a))) <
                                                    float(np.sum(np.abs(b)))],
                             max_workers=1)
        np.testing.assert_allclose(u[:3], [3., 1., 2.])
 
 
# ---------------------------------------------------------------------
# Defect 3 -- empty fibers are rejected (Schroder, Def. 7.1)
# ---------------------------------------------------------------------
 
class TestEmptyFibersRejected:
 
    def test_empty_fiber_in_the_middle_of_a_chain(self):
        R = [np.array([[0., 0.]]), np.zeros((0, 2)), np.array([[2., 2.]])]
        with pytest.raises(ValueError, match=r"empty"):
            OperadicGPAV(Q=_chain_Q(3), R_datasets=R, Y=np.array([9., 1.]),
                         assume_component_wise=True, max_workers=1)
 
    def test_empty_fiber_on_the_no_edge_fast_path(self):
        """The Q-has-no-edges branch returns early; the guard must precede it."""
        Q = nx.DiGraph()
        Q.add_nodes_from([0, 1])
        with pytest.raises(ValueError, match=r"empty"):
            OperadicGPAV(Q=Q, R_datasets=[np.array([[0., 0.]]), np.zeros((0, 2))],
                         Y=np.array([9.]), assume_component_wise=True, max_workers=1)
 
    def test_nonempty_chain_is_unaffected(self):
        R = [np.array([[0., 0.]]), np.array([[1., 1.]]), np.array([[2., 2.]])]
        u = OperadicGPAV(Q=_chain_Q(3), R_datasets=R, Y=np.array([9., 5., 1.]),
                         assume_component_wise=True, max_workers=1)
        np.testing.assert_allclose(u, [5., 5., 5.])
 
 
class TestYLengthValidated:
 
    R = [np.array([[0., 0.], [1., 1.]]), np.array([[2., 2.]])]   # N = 3
 
    @pytest.mark.parametrize("Y", [np.array([5., 1.]),                    # short
                                   np.array([5., 1., 9., 100., 200.])])   # long
    def test_wrong_length_is_rejected(self, Y):
        with pytest.raises(ValueError, match=r"length"):
            OperadicGPAV(Q=nx.DiGraph([(0, 1)]), R_datasets=self.R, Y=Y,
                         assume_component_wise=True, max_workers=1)
 
    def test_too_long_no_longer_returns_padded_zeros(self):
        """The dangerous case: the old code returned [3,3,9,0,0] with no error."""
        with pytest.raises(ValueError):
            OperadicGPAV(Q=nx.DiGraph([(0, 1)]), R_datasets=self.R,
                         Y=np.arange(20, dtype=float),
                         assume_component_wise=True, max_workers=1)
 
 
# ---------------------------------------------------------------------
# Defect 4 -- no RecursionError on wide inputs, and no global limit mutation
# ---------------------------------------------------------------------
 
class TestNoRecursionLimit:
 
    @staticmethod
    def _wide_fiber(n):
        return np.stack([np.arange(n, dtype=float), n - np.arange(n, dtype=float)], axis=1)
 
    def test_stage2_beyond_1000_blocks(self):
        """1,200 mutually incomparable blocks used to raise RecursionError."""
        n = 1200
        R = [self._wide_fiber(n), np.array([[1e6, 1e6]])]
        Y = np.concatenate([np.random.default_rng(0).normal(size=n), [1e3]])
        u = OperadicGPAV(Q=nx.DiGraph([(0, 1)]), R_datasets=R, Y=Y,
                         assume_component_wise=True, max_workers=1)
        assert len(u) == n + 1 and np.all(np.isfinite(u))
 
    def test_stage1_beyond_1000_elements(self):
        """A near-antichain fiber of 1,100 points used to raise RecursionError."""
        n = 1100
        pts = self._wide_fiber(n)
        pts[0] = [0., 0.]                       # one element below everything
        Y = np.concatenate([np.random.default_rng(0).normal(size=n), [1e3]])
        u = OperadicGPAV(Q=nx.DiGraph([(0, 1)]),
                         R_datasets=[pts, np.array([[1e6, 1e6]])], Y=Y,
                         assume_component_wise=True, max_workers=1)
        assert len(u) == n + 1 and np.all(np.isfinite(u))
 
    def test_library_does_not_mutate_the_recursion_limit(self):
        before = sys.getrecursionlimit()
        G = nx.DiGraph((i, i + 1) for i in range(3000))
        trend_following_order(G=G, Y={i: float(3000 - i) for i in G.nodes()})
        assert sys.getrecursionlimit() == before
 
    def test_deep_chain_at_the_stock_limit(self):
        """Depth is the longest chain; the iterative DFS must not care."""
        n = 20000
        G = nx.DiGraph((i, i + 1) for i in range(n - 1))
        order = trend_following_order(G=G, Y={i: float(n - i) for i in G.nodes()})
        assert len(order) == n
 
 
# ---------------------------------------------------------------------
# Optimisation -- the DAG builder must be equivalent AND fast
# ---------------------------------------------------------------------
 
class TestDagBuilderOptimisation:
 
    def test_incremental_matches_all_pairs(self):
        """The two branches must agree edge-for-edge on tie-free data."""
        rng = np.random.default_rng(0)
        for _ in range(120):
            n = int(rng.integers(3, 25))
            X = rng.random((n, int(rng.choice([1, 2, 3]))))
            fast = _build_dag_incrementally(
                np.argsort(X.sum(1)).tolist(),
                lambda a, b: default_comparator(X[a], X[b]), True)
            safe = _build_dag_incrementally(
                list(range(n)),
                lambda a, b: default_comparator(X[a], X[b]), False)
            assert set(fast.edges()) == set(safe.edges())
 
    def test_chain_build_is_not_cubic(self):
        """nx.has_path per candidate took ~45 s for an 800-element chain."""
        X = np.arange(800, dtype=float).reshape(-1, 1)
        t0 = time.time()
        G = _build_dag_incrementally(
            list(range(800)), lambda a, b: default_comparator(X[a], X[b]), True)
        elapsed = time.time() - t0
        assert G.number_of_edges() == 799
        assert elapsed < 5.0, f"chain build took {elapsed:.1f}s (was ~45s with has_path)"
 
 
# ---------------------------------------------------------------------
# End-to-end -- the property the whole algorithm exists to guarantee
# ---------------------------------------------------------------------
 
class TestFeasibility:
 
    def test_random_instances_are_monotone(self):
        """The fit must respect the lexicographic-sum order on every instance."""
        rng = np.random.default_rng(11)
        checked = 0
        for _ in range(40):
            m = int(rng.integers(2, 5))
            Q = nx.DiGraph()
            Q.add_nodes_from(range(m))
            for i in range(m):
                for j in range(i + 1, m):
                    if rng.random() < 0.5:
                        Q.add_edge(i, j)
            if Q.number_of_edges() == 0:
                continue
            R = [rng.random((int(rng.integers(2, 9)), 2)) for _ in range(m)]
            Y = rng.normal(size=sum(len(r) for r in R))
            u = OperadicGPAV(Q=Q, R_datasets=R, Y=Y,
                             assume_component_wise=True, max_workers=1)
            assert not _violations(_full_poset(Q, R), u)
            checked += 1
        assert checked >= 20


 
class TestFastMode:
    """fast_mode routes each Q-edge through one weight-0 gateway node instead of
    the complete bipartite graph max(i) x min(j).  Same constraint set, smaller
    graph, different greedy path.
 
    NOTE on what is and is not asserted here.  There is NO structural class of Q
    on which the two modes provably agree -- chains, trees and diamonds all
    produce occasional disagreements (measured: ~4% of random instances).  So
    these tests pin the properties that ARE invariant (default off, feasibility,
    weight-0, memory) and bound the deviation rather than forbidding it."""
 
    @staticmethod
    def _random_Q(rng, m, style):
        Q = nx.DiGraph(); Q.add_nodes_from(range(m))
        if style == "chain":
            Q.add_edges_from((i, i + 1) for i in range(m - 1))
        elif style == "diamond" and m >= 4:
            Q.add_edges_from([(0, 1), (0, 2), (1, m - 1), (2, m - 1)])
        elif style == "tree":
            for j in range(1, m):
                Q.add_edge(int(rng.integers(0, j)), j)
        elif style == "fanin":
            Q.add_edges_from((i, m - 1) for i in range(m - 1))
        elif style == "fanout":
            Q.add_edges_from((0, i) for i in range(1, m))
        else:
            for i in range(m):
                for j in range(i + 1, m):
                    if rng.random() < 0.5:
                        Q.add_edge(i, j)
        return Q
 
    def test_default_is_off(self):
        """A caller who never mentions fast_mode must get the exact form."""
        rng = np.random.default_rng(0)
        R = [rng.random((12, 2)) + 3.0 * k for k in range(4)]
        Q = self._random_Q(rng, 4, "chain")
        Y = rng.normal(size=48)
        a = OperadicGPAV(Q=Q, R_datasets=R, Y=Y, assume_component_wise=True, max_workers=1)
        b = OperadicGPAV(Q=Q, R_datasets=R, Y=Y, assume_component_wise=True,
                         max_workers=1, fast_mode=False)
        np.testing.assert_array_equal(a, b)
 
    def test_fast_mode_is_feasible(self):
        """The gateway must preserve the constraint set on every Q shape.
        This is the strong invariant: it holds unconditionally."""
        rng = np.random.default_rng(5)
        for style in ("chain", "diamond", "tree", "fanin", "fanout", "random"):
            for _ in range(6):
                m = int(rng.integers(3, 7))
                Q = self._random_Q(rng, m, style)
                if Q.number_of_edges() == 0:
                    continue
                R = [rng.random((int(rng.integers(2, 9)), 2)) for _ in range(m)]
                Y = rng.normal(size=sum(len(r) for r in R))
                u = OperadicGPAV(Q=Q, R_datasets=R, Y=Y, assume_component_wise=True,
                                 max_workers=1, fast_mode=True)
                assert not _violations(_full_poset(Q, R), u), \
                    f"fast_mode produced an infeasible fit on a {style} Q"
 
    def test_gateway_carries_no_mass(self):
        """weight=0 means the gateway never enters an average.  If it did, the
        total fitted mass would drift away from sum(Y)."""
        rng = np.random.default_rng(9)
        for style in ("chain", "diamond", "random"):
            m = 5
            Q = self._random_Q(rng, m, style)
            if Q.number_of_edges() == 0:
                continue
            R = [rng.random((int(rng.integers(3, 10)), 2)) for _ in range(m)]
            Y = rng.normal(size=sum(len(r) for r in R))
            u = OperadicGPAV(Q=Q, R_datasets=R, Y=Y, assume_component_wise=True,
                             max_workers=1, fast_mode=True)
            assert u.sum() == pytest.approx(Y.sum(), abs=1e-9), \
                "gateway leaked mass into the fit"
 
    def test_deviation_from_exact_is_bounded(self):
        """fast_mode may reach a different feasible optimum, but not a wild one.
        Observed worst over 565 random instances: 1.6% of the total sum of
        squares.  The bound below leaves ~6x headroom."""
        rng = np.random.default_rng(31337)
        worst = 0.0
        for t in range(120):
            m = int(rng.integers(2, 8))
            Q = self._random_Q(rng, m, ("chain", "diamond", "tree", "random")[t % 4])
            if Q.number_of_edges() == 0:
                continue
            R = [rng.random((int(rng.integers(1, 12)), 2)) for _ in range(m)]
            Y = rng.normal(size=sum(len(r) for r in R))
            tss = float(np.sum((Y - Y.mean()) ** 2))
            if tss < 1e-9:
                continue
            kw = dict(Q=Q, R_datasets=R, Y=Y, assume_component_wise=True, max_workers=1)
            a, b = OperadicGPAV(**kw), OperadicGPAV(fast_mode=True, **kw)
            worst = max(worst, abs(sse(b, Y) - sse(a, Y)) / tss)
        assert worst < 0.10, f"fast_mode deviated by {worst:.4f} of TSS from the exact form"
 
    def test_fast_mode_uses_less_memory(self):
        """On an uncompressible fiber the saving must actually materialise."""
        ni, m = 40, 3
        R = [np.stack([np.arange(ni, dtype=float),
                       ni - np.arange(ni, dtype=float)], axis=1) + 0.001 * k
             for k in range(m)]
        Q = nx.DiGraph(); Q.add_nodes_from(range(m))
        Q.add_edges_from((i, i + 1) for i in range(m - 1))
        Y = np.random.default_rng(0).normal(size=m * ni)
        import tracemalloc
        peaks = []
        for fm in (False, True):
            tracemalloc.start()
            OperadicGPAV(Q=Q, R_datasets=R, Y=Y, assume_component_wise=True,
                         max_workers=1, fast_mode=fm)
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
        assert peaks[1] < peaks[0] / 2, f"fast_mode did not reduce memory: {peaks}"

if __name__ == "__main__":
    # Run tests
    sys.exit(pytest.main([__file__, "-q"]))
