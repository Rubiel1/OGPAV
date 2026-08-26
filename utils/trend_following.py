from __future__ import annotations
from typing import Dict, Hashable, Iterable, List, Tuple, Union, Any, Optional, Callable, Sequence
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------
# Helper: Incremental Non-Redundant DAG Construction (Memory Efficient)
# ---------------------------------------------------------------------

def _build_dag_incrementally(
    indices: Sequence[int],
    precedes_func: Callable[[int, int], bool],
    assume_component_wise: bool = False
) -> nx.DiGraph:
    """
    Constructs a DAG on `indices` by adding only non-redundant edges.
    
    If assume_component_wise is True:
        The data are vectors and the order is v<y if every coordinate of v is less equal to every
        coordinate of y
        Assumes `indices` is topologically sorted (or at least ordered)
        and builds the DAG incrementally (memory efficient).
    If assume_component_wise is False:
        Checks all possible pairs (O(N^2)) to build a full valid DAG, then
        applies `nx.transitive_reduction` to remove redundant edges.
        
    Algorithm similar to Sysoev et al. Algorithm 3.
    """
    if assume_component_wise:
        G = nx.DiGraph()
        G.add_nodes_from(indices)
        
        N_subset = len(indices)
        
        for idx_j in range(N_subset):
            j = indices[idx_j]
            # `blocked` = nodes already known to reach a chosen parent of j, so an
            # edge from them to j would be transitively redundant. Maintaining it
            # incrementally replaces the per-candidate nx.has_path() traversal.
            blocked = set()
            
            # Scan backwards to find immediate predecessors
            for idx_i in range(idx_j - 1, -1, -1):
                i = indices[idx_i]
                
                if i in blocked:
                    continue
                
                if precedes_func(i, j):
                    G.add_edge(i, j)
                    # every ancestor of i also reaches j through i
                    stack = [i]
                    while stack:
                        x = stack.pop()
                        for w in G.predecessors(x):
                            if w not in blocked:
                                blocked.add(w)
                                stack.append(w)
        return G
    else:
        G = nx.DiGraph()
        G.add_nodes_from(indices)
        
        N_subset = len(indices)
        for idx_i in range(N_subset):
            for idx_j in range(N_subset):
                if idx_i == idx_j:
                    continue
                i = indices[idx_i]
                j = indices[idx_j]
                if precedes_func(i, j):
                    G.add_edge(i, j)
                    
        try:
            return nx.transitive_reduction(G)
        except nx.NetworkXError:
            # Diagnose only on failure.  transitive_reduction refuses a cyclic graph,
            # and the only way this loop can produce a cycle is a relation that is not
            # antisymmetric: two DISTINCT items u != v with precedes(u,v) and
            # precedes(v,u).  That is a preorder, not a partial order.
            _u = _v = None
            for a, b in G.edges():
                if G.has_edge(b, a):
                    _u, _v = a, b
                    break
            raise nx.NetworkXError(
                f"The supplied relation is not antisymmetric: items {_u} and {_v} are "
                f"distinct but satisfy precedes({_u},{_v}) and precedes({_v},{_u}), so the "
                "graph is cyclic and has no transitive reduction. A partial order requires "
                "that only equal items compare both ways. Either the two items are in fact "
                "duplicates (deduplicate the fiber) or the comparator ranks by a key -- a "
                "sum, norm, score or index -- that ties them; add a tie-break to make it strict."
            ) from None

def _lower_y_naive(
    P: List[Hashable],
    G: nx.DiGraph,
    Y_map: Dict[Hashable, float],
    sort_key: Callable
) -> List[Hashable]:
    """
    Naive O(N²) implementation of LowerY procedure from Algorithm 5.
    Literal translation of the paper's pseudocode.
    
    Algorithm 5 LowerY procedure:
    1. Set T = ∅
    2. While P ≠ ∅:
       2.1: Set i = P(1) and P' = Pred(i, P)
       2.2: Compute P'' = LowerY(P')
       2.3: Set T = [T, P'', i]
       2.4: Update P by removing both i and all k ∈ P'
    3. Return order = T
    
    Parameters
    ----------
    P : List[Hashable]
        Sequence of nodes sorted by Y (represents current remaining set)
    G : nx.DiGraph
        DAG encoding partial order (edge u→v means u ≺ v)
    Y_map : Dict[Hashable, float]
        Mapping from node to Y value
    sort_key : Callable
        Function to sort nodes (for stable ordering)
        
    Returns
    -------
    List[Hashable]
        Topological order produced by LowerY
        
    Complexity
    ----------
    Time: O(|P|²) where |P| is the length of input sequence
    Space: O(|P|) for recursion stack
    """
    # Base case: empty set
    if not P:
        return []
    
    # Step 2.1: i = P(1) (first element, has minimal Y)
    i = P[0]
    
    # Find P' = Pred(i, P): predecessors of i that are in P (TRANSITIVE predecessors)
    # Since G might be a Hasse diagram (transitively reduced), we must find ancestors.
    ancestors_i = nx.ancestors(G, i)
    P_prime = [node for node in P if node in ancestors_i]
    
    # Step 2.2: Recursively compute P'' = LowerY(P')
    # P' needs to be sorted by Y for the recursive call
    P_prime_sorted = sorted(P_prime, key=sort_key)
    P_double_prime = _lower_y_naive(P_prime_sorted, G, Y_map, sort_key)
    
    # Step 2.3: T = [T_previous, P'', i]
    # We're building T incrementally through recursion
    
    # Step 2.4: Remove i and all k ∈ P' from P for next iteration
    to_remove = set(P_prime) | {i}
    P_remaining = [node for node in P if node not in to_remove]
    
    # Recursively process remaining P
    T_rest = _lower_y_naive(P_remaining, G, Y_map, sort_key)
    
    # Return [P'', i, T_rest]
    return P_double_prime + [i] + T_rest


def default_comparator(a: Any, b: Any) -> bool:
    """
    Default comparator: Coordinate-wise dominance (a <= b).
    Handles both scalars and array-like objects.
    """
    if a is b:
        return True
    
    # Numpy-aware check
    oa = np.asanyarray(a)
    ob = np.asanyarray(b)
    
    # Handle scalar case
    if oa.ndim == 0 and ob.ndim == 0:
        return bool(oa <= ob)
    
    # Handle array case (must have same shape)
    if oa.shape != ob.shape:
        raise ValueError(f"Cannot compare arrays of different shapes: {oa.shape} vs {ob.shape}")
    
    return bool(np.all(oa <= ob))

# ---------------------------------------------------------------------


def _lower_y_dfs(
    nodes: List[Hashable],
    G: nx.DiGraph,
    Y_map: Dict[Hashable, float],
    sort_key: Callable
) -> List[Hashable]:
    """
    Optimized O((N+E) log N) DFS implementation of LowerY.
    
    This implementation uses DFS with memoization to avoid redundant work,
    making it more efficient for sparse graphs.
    
    Parameters
    ----------
    nodes : List[Hashable]
        All nodes to process (should be pre-sorted by Y)
    G : nx.DiGraph
        DAG encoding partial order (edge u→v means u ≺ v)
    Y_map : Dict[Hashable, float]
        Mapping from node to Y value
    sort_key : Callable
        Function to sort nodes (for stable ordering)
        
    Returns
    -------
    List[Hashable]
        Topological order produced by LowerY
        
    Complexity
    ----------
    Time: O((N+E) log N) where E is number of edges
    Space: O(N+E) for graph structures and tracking sets
    """
    # Build Reverse Graph with Sorted Adjacency (Parents)
    # We need to access parents of v to implement LowerY(Ancestors(u))
    rev_adj = {v: [] for v in nodes}
    for u, v in G.edges():
        rev_adj[v].append(u)  # u → v means u precedes v
        
    for v in nodes:
        if rev_adj[v]:
            rev_adj[v].sort(key=sort_key)  # Sort by Y
    
    # DFS Implementation of LowerY
    yielded = set()
    result = []

    for root in nodes:
        if root in yielded:
            continue
        stack = [(root, iter(rev_adj[root]))]
        while stack:
            u, it = stack[-1]
            for p in it:
                if p not in yielded:
                    stack.append((p, iter(rev_adj[p])))
                    break
            else:
                stack.pop()
                if u not in yielded:
                    yielded.add(u)
                    result.append(u)

    return result


# ---------------------------------------------------------------------
# Trend Following Order
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def trend_following_order(
    X: Optional[Any] = None,
    Y: Union[Dict[Hashable, float], np.ndarray, Sequence[float]] = None,
    f: Optional[Callable[[Any, Any], bool]] = None,
    G: Optional[nx.DiGraph] = None,
    *,
    stable_tiebreak: bool = True,
    sparse_data: bool = True,
) -> List[Hashable]:
    """
    Faithful implementation of the SB paper's trend-following topological order:
    Algorithm 5: LowerY(P), where P0 is the sequence of observations sorted by Y.

    Inputs
    ------
    X : Any (optional)
        Dataset vectors/elements. Required if G is None.
    Y : dict or array-like
        Observed response values for nodes. 
        If G is provided, Y can be a dict mapping node -> val.
        If X is provided (and G built internally), Y must be array-like aligned with X.
    f : Callable (optional)
        Comparator f(a, b) -> bool (a <= b). Used to build G if G is None.
        Defaults to coordinate-wise dominance.
    G : nx.DiGraph (optional)
        DAG encoding the partial order (edge u->v means u ≺ v).
        If None, the DAG is built from X using f and an incremental construction.
    stable_tiebreak : bool
        If True, ties are broken deterministically using the (Y, rank) order induced
        by sorting nodes by (Y, node_as_str).
    sparse_data : bool (default=True)
        If True, uses optimized DFS-based implementation: O((N+E) log N) time, O(N+E) space.
        If False, uses naive implementation from paper: O(N²) time, O(N) space.
        

    Output
    ------
    A list T (topological order) produced by the published LowerY procedure.
    
    Complexity
    ----------
    - sparse_data=False: O(N²) time, O(N) space
    - sparse_data=True: O((N+E) log N) time, O(N+E) space
    """

    # --- 1) Resolve Inputs and Build G if needed
    if G is not None:
        # Use provided graph
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("G must be a DAG for trend following order.")
        nodes = list(G.nodes())
        m = len(nodes)
        
        # Resolve Y for provided G
        # If Y is array, we must map it to nodes.
        if not isinstance(Y, dict):
            Y_arr = np.asarray(Y, dtype=float)
            if Y_arr.shape[0] != m:
                raise ValueError(f"Y array length {Y_arr.shape[0]} != num nodes {m}")
            # If G nodes are 0..m-1 integers, we assume alignment.
            # If not, this is ambiguous, but we'll try zip(nodes, Y).
            # Prefer dict input if G has non-integer nodes.
            Y = {nodes[i]: float(Y_arr[i]) for i in range(m)}
            
    else:
        # Build G from X
        if X is None:
            raise ValueError("Must provide either G (graph) or X (dataset) to trend_following_order.")
        
        m = len(X)
        if m == 0:
            return []
            
        nodes = list(range(m))
        
        if f is None:
            f = default_comparator
            
        # Build graph incrementally (memory efficient)
        # Algorithm 5 requires: "sort the observations in D by the value of Y_i"
        # So we must sort by Y values, not by sum of X coordinates
        try:
             # Sort by Y values as specified in Algorithm 5
             Y_arr = np.asarray(Y, dtype=float)
             if Y_arr.shape[0] != m:
                 raise ValueError(f"Y array length {Y_arr.shape[0]} != X length {m}")
             topo_indices = np.argsort(Y_arr).tolist()
        except:
             # Fallback: assume Y is just a list, use original order
             topo_indices = list(range(m))

        def check_precedence(i, j):
            # i, j are indices. Check f(X[i], X[j])
            return f(X[i], X[j])

        # This builds the graph edges (Memory Efficient)
        G = _build_dag_incrementally(topo_indices, check_precedence)
        
        # Resolve Y for built G (nodes are 0..m-1)
        if not isinstance(Y, dict):
            Y_arr = np.asarray(Y, dtype=float)
            if Y_arr.shape[0] != m:
                raise ValueError(f"Y array length {Y_arr.shape[0]} != X length {m}")
            # Align Y with indices 0..m-1
            Y = {i: float(Y_arr[i]) for i in range(m)}


    # Ensure all nodes have a Y-value
    missing = [v for v in nodes if v not in Y]
    if missing:
        raise KeyError(f"Missing Y-values for {len(missing)} nodes, e.g. {missing[:5]}")

    # --- 2) Prepare Sorted Orders
    if stable_tiebreak:
        def sort_key(v): return (float(Y[v]), str(v))
    else:
        def sort_key(v): return float(Y[v])

    # Global priority order (Algorithm 5 outer loop: select minimal Y from P)
    global_order = sorted(nodes, key=sort_key)

    # --- 3) Dispatch to Appropriate Implementation
    # Choose between naive O(N²) and optimized DFS O((N+E) log N)
    if sparse_data:
        # Use optimized DFS implementation for sparse graphs
        return _lower_y_dfs(global_order, G, Y, sort_key)
    else:
        # Use naive O(N²) implementation from paper
        return _lower_y_naive(global_order, G, Y, sort_key)

