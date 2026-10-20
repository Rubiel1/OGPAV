"""Reproducible adjacency-storage metrics for GPAV-family algorithms.

Sysoev, Burdakov, and Grimvall report the maximum KBytes used to store the
adjacency matrix (GPAV) or matrices (SB-GPAV), rather than whole-process RAM.
Their paper does not state the sparse-matrix byte layout.  We therefore report
two related quantities without materialising a second matrix:

``paper_style_kib``
    Maximum number of stored adjacency nonzeros divided by 1024, i.e. one byte
    per Boolean nonzero.  This is the closest representation-independent
    reconstruction of the paper's adjacency-matrix measure.

``csr32_kib``
    A fully specified CSR estimate with one Boolean data byte and one 32-bit
    column index per nonzero, plus one 32-bit row pointer per row (and the final
    pointer): ``5 * E + 4 * (V + 1)`` bytes.

Both metrics deliberately exclude observations, responses, Python/NetworkX
object overhead, worker processes, and temporary files.  Slurm MaxRSS remains
the appropriate measurement of actual whole-process memory.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


def csr32_bytes(num_nodes: int, num_edges: int) -> int:
    """Return bytes for a directed Boolean adjacency matrix in CSR32 form."""
    num_nodes = int(num_nodes)
    num_edges = int(num_edges)
    if num_nodes < 0 or num_edges < 0:
        raise ValueError("Graph node and edge counts must be nonnegative.")
    return 5 * num_edges + 4 * (num_nodes + 1)


def graph_counts(graph: Any) -> Dict[str, int]:
    """Return the node/edge counts needed by the storage metrics."""
    return {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
    }


def empty_group() -> Dict[str, int]:
    """Create an accumulator for a family of graphs used one at a time."""
    return {
        "graphs_seen": 0,
        "max_nodes": 0,
        "max_edges": 0,
        "max_paper_style_bytes": 0,
        "max_csr32_bytes": 0,
    }


def update_group(group: Dict[str, int], counts: Mapping[str, int]) -> None:
    """Update an accumulator with one graph, retaining only maxima."""
    nodes = int(counts["nodes"])
    edges = int(counts["edges"])
    group["graphs_seen"] += 1
    group["max_nodes"] = max(group["max_nodes"], nodes)
    group["max_edges"] = max(group["max_edges"], edges)
    group["max_paper_style_bytes"] = max(group["max_paper_style_bytes"], edges)
    group["max_csr32_bytes"] = max(group["max_csr32_bytes"], csr32_bytes(nodes, edges))


def build_profile(groups: Mapping[str, Mapping[str, int]]) -> Dict[str, Any]:
    """Build the final peak profile from named, already-aggregated graph groups."""
    normalised = {name: dict(values) for name, values in groups.items()}
    if not normalised:
        normalised = {"none": empty_group()}

    paper_name, paper_group = max(
        normalised.items(), key=lambda item: int(item[1]["max_paper_style_bytes"])
    )
    csr_name, csr_group = max(
        normalised.items(), key=lambda item: int(item[1]["max_csr32_bytes"])
    )
    paper_bytes = int(paper_group["max_paper_style_bytes"])
    csr_bytes_value = int(csr_group["max_csr32_bytes"])

    return {
        "definition": (
            "Peak adjacency storage across graph phases. paper_style assumes one "
            "byte per stored Boolean nonzero; csr32 uses 5*E + 4*(V+1) bytes."
        ),
        "paper_style_peak_graph": paper_name,
        "paper_style_peak_edges": int(paper_group["max_edges"]),
        "paper_style_bytes": paper_bytes,
        "paper_style_kib": paper_bytes / 1024.0,
        "csr32_peak_graph": csr_name,
        "csr32_bytes": csr_bytes_value,
        "csr32_kib": csr_bytes_value / 1024.0,
        "csr32_mib": csr_bytes_value / (1024.0 * 1024.0),
        "groups": normalised,
        "scope": (
            "Adjacency structures only; excludes X, Y, fitted values, blocks, "
            "NetworkX/Python overhead, multiprocessing, and temporary files."
        ),
    }
