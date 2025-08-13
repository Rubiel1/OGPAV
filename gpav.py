import hasse
import networkx as nx
import numpy as np

#It assumes the input was passed to make_reduced_poset_from_graph
def gpav(Y, poset, topo_order, weights=None):
    """
    GPAV corrected: uses head_of[] to track current head for each original node.
    This ensures when block j is absorbed into k we treat k as predecessor of future nodes.
    """
    n = len(Y)
    Y = np.asarray(Y, dtype=float)
    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    # blocks: head_node -> {'elements':[...], 'value':..., 'weight':...}
    blocks = {i: {'elements': [i], 'value': float(Y[i]), 'weight': float(weights[i])} for i in range(n)}

    # map every original node -> current head node (initially itself)
    head_of = list(range(n))

    # DAG (assumes nodes are 0..n-1)
    G = poset.hasse
    predecessors = {i: set(G.predecessors(i)) for i in range(n)}

    # Process in topo order (topo_order must be node labels)
    for k in topo_order:
        # if k is not a head any more skip
        if k not in blocks:
            continue

        while True:
            # Build set of current-head predecessors of k
            pred_heads = set()
            for p in predecessors.get(k, ()):
                h = head_of[p]
                # only consider current heads (blocks may have been absorbed)
                if h in blocks:
                    pred_heads.add(h)

            # violating preds are those heads with value > blocks[k]['value']
            violating_heads = [h for h in pred_heads if blocks[h]['value'] > blocks[k]['value']]
            if not violating_heads:
                break

            # pick the violator with maximum value (most severe)
            j = max(violating_heads, key=lambda h: blocks[h]['value'])
            # Merge j into k
            w_k = blocks[k]['weight']
            w_j = blocks[j]['weight']
            new_w = w_k + w_j
            new_val = (blocks[k]['value'] * w_k + blocks[j]['value'] * w_j) / new_w
            new_elems = blocks[k]['elements'] + blocks[j]['elements']

            # update head_of for elements coming from j
            for node in blocks[j]['elements']:
                head_of[node] = k

            # update block k
            blocks[k]['elements'] = new_elems
            blocks[k]['weight'] = new_w
            blocks[k]['value'] = new_val

            # remove j
            del blocks[j]

            # (optional) merge predecessor lists for faster future lookup:
            predecessors[k].update(predecessors.get(j, set()))
            predecessors[k].discard(j)

    # produce u
    u = np.empty(n, dtype=float)
    for head, block in blocks.items():
        for idx in block['elements']:
            u[idx] = block['value']
    return u



if __name__ == "__main__":
    
    # Define the poset using chains
    # For example, 0 < 1 < 3 and 0 < 2 < 3
    poset = hasse.PoSet.from_chains(
        [0, 3],[1, 3],[ 1, 2]
    )
    
    # Observed data
    Y = [2.0, 8.0, 2.1, 3.0]
    
    # Optional weights (default is equal weights)
    weights = None # [1.0, 1.0, 1.0, 1.0]
    #print topological order
    linearization = list(nx.topological_sort(poset.hasse))
    print(f"input data {Y}\n with poset { [ (x,y)  for x in range(poset.__len__()) for y in list(poset.hasse.successors(x)) ] }")
    print(f"linearization {linearization}")
    # Apply GPAV
    u = gpav(Y, poset, linearization, weights)
    
    print("Adjusted values:", u)
    
    