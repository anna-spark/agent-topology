"""
topologies.py
-------------
Constructs communication graphs for multi-agent topology experiments.

All graphs use agent IDs as node labels (e.g. "agent_00", "agent_01", ...).

Implemented topologies:
  - chain       : linear path graph
  - tree        : balanced binary tree
  - random      : Erdos-Renyi random graph (connected)
  - small_world : Watts-Strogatz small-world graph

Each function returns a networkx Graph.
"""

import networkx as nx
import random


def get_agent_ids(n: int) -> list[str]:
    return [f"agent_{i:02d}" for i in range(n)]


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

def make_chain(n: int) -> nx.Graph:
    """
    Linear path: agent_00 - agent_01 - ... - agent_(n-1)
    Information must travel hop by hop end to end.
    Diameter = n-1, average path length = n/3
    """
    ids = get_agent_ids(n)
    G = nx.path_graph(ids)
    return G


# ---------------------------------------------------------------------------
# Tree (balanced binary tree)
# ---------------------------------------------------------------------------

def make_tree(n: int) -> nx.Graph:
    """
    Balanced binary tree. Root = agent_00, children assigned BFS-order.
    Supports hierarchical decomposition and aggregation.
    Diameter = O(log n), but leaf-to-leaf paths go through root.
    """
    ids = get_agent_ids(n)
    # Build integer binary tree then relabel
    G_int = nx.balanced_tree(r=2, h=4)  # r=2 (binary), h=4 gives 31 nodes
    # Trim to exactly n nodes (BFS order)
    nodes_bfs = list(nx.bfs_tree(G_int, 0).nodes())[:n]
    G_trimmed = G_int.subgraph(nodes_bfs).copy()
    # Relabel to agent IDs
    mapping = {old: ids[i] for i, old in enumerate(nodes_bfs)}
    G = nx.relabel_nodes(G_trimmed, mapping)
    return G


# ---------------------------------------------------------------------------
# Random graph (Erdos-Renyi, guaranteed connected)
# ---------------------------------------------------------------------------

def make_random(n: int, seed: int = 42) -> nx.Graph:
    """
    Erdos-Renyi G(n, p) random graph, guaranteed connected.
    p chosen so expected degree ~ log(n) (connectivity threshold).
    Serves as neutral baseline.
    """
    rng = random.Random(seed)
    ids = get_agent_ids(n)
    p = 2.5 * (1 / n) * (n ** 0.5)   # slightly above connectivity threshold
    p = min(p, 0.3)                    # cap density

    # Retry until connected
    for attempt in range(100):
        G_int = nx.erdos_renyi_graph(n, p, seed=seed + attempt)
        if nx.is_connected(G_int):
            mapping = {i: ids[i] for i in range(n)}
            return nx.relabel_nodes(G_int, mapping)

    # Fallback: add spanning tree edges to connect components
    G_int = nx.erdos_renyi_graph(n, p, seed=seed)
    components = list(nx.connected_components(G_int))
    for i in range(len(components) - 1):
        u = min(components[i])
        v = min(components[i + 1])
        G_int.add_edge(u, v)
    mapping = {i: ids[i] for i in range(n)}
    return nx.relabel_nodes(G_int, mapping)


# ---------------------------------------------------------------------------
# Small-world (Watts-Strogatz)
# ---------------------------------------------------------------------------

def make_small_world(n: int, k: int = 4, p: float = 0.2, seed: int = 42) -> nx.Graph:
    """
    Watts-Strogatz small-world graph.
    k = each node connects to k nearest neighbors in ring
    p = probability of rewiring each edge (adds long-range shortcuts)
    Combines local clustering with short global path lengths.
    """
    ids = get_agent_ids(n)
    G_int = nx.watts_strogatz_graph(n, k=k, p=p, seed=seed)
    mapping = {i: ids[i] for i in range(n)}
    return nx.relabel_nodes(G_int, mapping)


# ---------------------------------------------------------------------------
# Registry — easy lookup by name
# ---------------------------------------------------------------------------

TOPOLOGY_BUILDERS = {
    "chain":       make_chain,
    "tree":        make_tree,
    "random":      make_random,
    "small_world": make_small_world,
}


def get_topology(name: str, n: int, seed: int = 42) -> nx.Graph:
    if name not in TOPOLOGY_BUILDERS:
        raise ValueError(f"Unknown topology '{name}'. Choose from: {list(TOPOLOGY_BUILDERS)}")
    if name in ("random", "small_world"):
        return TOPOLOGY_BUILDERS[name](n, seed=seed)
    return TOPOLOGY_BUILDERS[name](n)


# ---------------------------------------------------------------------------
# Graph statistics (used in analysis.py)
# ---------------------------------------------------------------------------

def compute_graph_stats(G: nx.Graph, name: str) -> dict:
    """Compute key network-theoretic statistics for a graph."""
    stats = {"topology": name, "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()}

    # Edge density
    n = G.number_of_nodes()
    stats["edge_density"] = nx.density(G)

    # Connected components
    stats["n_components"] = nx.number_connected_components(G)

    if nx.is_connected(G):
        stats["diameter"] = nx.diameter(G)
        stats["avg_shortest_path"] = nx.average_shortest_path_length(G)
    else:
        # Use largest component
        largest = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest)
        stats["diameter"] = nx.diameter(H)
        stats["avg_shortest_path"] = nx.average_shortest_path_length(H)

    stats["avg_clustering"] = nx.average_clustering(G)

    degrees = [d for _, d in G.degree()]
    stats["avg_degree"] = sum(degrees) / len(degrees)
    stats["max_degree"] = max(degrees)
    stats["min_degree"] = min(degrees)

    # Betweenness centrality (max — identifies bottleneck nodes)
    bc = nx.betweenness_centrality(G)
    stats["max_betweenness"] = max(bc.values())
    stats["avg_betweenness"] = sum(bc.values()) / len(bc)
    stats["bottleneck_agent"] = max(bc, key=bc.get)

    return stats


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    N = 20
    print(f"Building all topologies for N={N} agents...\n")

    rows = []
    for name in TOPOLOGY_BUILDERS:
        G = get_topology(name, N)
        stats = compute_graph_stats(G, name)
        rows.append(stats)

        print(f"--- {name.upper()} ---")
        print(f"  Nodes: {stats['n_nodes']}  Edges: {stats['n_edges']}")
        print(f"  Density: {stats['edge_density']:.3f}")
        print(f"  Diameter: {stats['diameter']}")
        print(f"  Avg shortest path: {stats['avg_shortest_path']:.2f}")
        print(f"  Avg clustering: {stats['avg_clustering']:.3f}")
        print(f"  Avg degree: {stats['avg_degree']:.2f}")
        print(f"  Max betweenness: {stats['max_betweenness']:.3f}  (bottleneck: {stats['bottleneck_agent']})")
        print()

    df = pd.DataFrame(rows).set_index("topology")
    print("\nSummary table:")
    print(df[[
        "n_edges", "edge_density", "diameter",
        "avg_shortest_path", "avg_clustering", "max_betweenness"
    ]].to_string())