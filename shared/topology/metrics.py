"""
Graph topology metrics for characterizing sparse network structure.

Computes clustering coefficient, average path length, small-world coefficient,
modularity, degree distribution statistics, and weight distribution statistics.
These metrics map directly to brain connectivity variables from the
neuroscience literature (clustering coefficient, small-worldedness,
number of clusters, communicability).
"""

import numpy as np
import networkx as nx
from scipy import stats
from typing import Optional
import warnings


def mask_to_graph(
    mask: np.ndarray,
    weights: Optional[np.ndarray] = None,
    directed: bool = False,
) -> nx.Graph:
    """Convert a (n_out, n_in) mask/weight matrix to a NetworkX graph.

    Creates a bipartite graph with input nodes 0..n_in-1 and output nodes
    n_in..n_in+n_out-1.
    """
    n_out, n_in = mask.shape
    GraphClass = nx.DiGraph if directed else nx.Graph
    G = GraphClass()

    G.add_nodes_from(range(n_in), bipartite=0, layer="input")
    G.add_nodes_from(range(n_in, n_in + n_out), bipartite=1, layer="output")

    rows, cols = np.where(mask > 0)
    for r, c in zip(rows, cols):
        w = float(weights[r, c]) if weights is not None else 1.0
        G.add_edge(c, n_in + r, weight=w)

    return G


def masks_to_full_graph(
    masks: list,
    layer_sizes: list,
    weights_list: Optional[list] = None,
) -> nx.Graph:
    """Convert a list of layer masks to a single graph spanning all layers."""
    G = nx.Graph()
    offset = 0
    node_layers = {}

    for layer_idx, size in enumerate(layer_sizes):
        for i in range(size):
            node_id = offset + i
            G.add_node(node_id, layer=layer_idx)
            node_layers[node_id] = layer_idx
        offset += size

    layer_offsets = [0]
    for s in layer_sizes[:-1]:
        layer_offsets.append(layer_offsets[-1] + s)

    for layer_idx, mask in enumerate(masks):
        n_out, n_in = mask.shape
        w = weights_list[layer_idx] if weights_list else None
        rows, cols = np.where(mask > 0)
        for r, c in zip(rows, cols):
            src = layer_offsets[layer_idx] + c
            dst = layer_offsets[layer_idx + 1] + r
            weight = float(w[r, c]) if w is not None else 1.0
            G.add_edge(src, dst, weight=weight)

    return G


def clustering_coefficient(G: nx.Graph) -> dict:
    """Compute local and global clustering coefficients."""
    if len(G) < 3:
        return {"local_mean": 0.0, "global": 0.0}

    local_cc = nx.clustering(G, weight="weight")
    return {
        "local_mean": np.mean(list(local_cc.values())),
        "global": nx.transitivity(G),
    }


def average_path_length(G: nx.Graph) -> float:
    """Compute average shortest path length (uses largest component if disconnected)."""
    if len(G) < 2:
        return 0.0

    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)

    # Use largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    if len(subgraph) < 2:
        return 0.0
    return nx.average_shortest_path_length(subgraph)


def small_world_coefficient(
    G: nx.Graph, n_random: int = 10, seed: int = 42
) -> float:
    """Compute small-world coefficient sigma = (C/C_rand) / (L/L_rand).

    Sigma > 1 indicates small-world properties.
    Uses matched Erdos-Renyi random graphs for normalization.
    """
    n = len(G)
    m = G.number_of_edges()

    if n < 4 or m < 2:
        return 0.0

    C = nx.average_clustering(G)
    L = average_path_length(G)

    if L == 0 or C == 0:
        return 0.0

    C_rands = []
    L_rands = []
    p = 2.0 * m / (n * (n - 1))

    for i in range(n_random):
        G_rand = nx.erdos_renyi_graph(n, p, seed=seed + i)
        if len(G_rand) > 0 and G_rand.number_of_edges() > 0:
            c_r = nx.average_clustering(G_rand)
            if nx.is_connected(G_rand):
                l_r = nx.average_shortest_path_length(G_rand)
            else:
                lcc = max(nx.connected_components(G_rand), key=len)
                sg = G_rand.subgraph(lcc)
                l_r = nx.average_shortest_path_length(sg) if len(sg) > 1 else 0
            if c_r > 0 and l_r > 0:
                C_rands.append(c_r)
                L_rands.append(l_r)

    if not C_rands or not L_rands:
        return 0.0

    C_rand = np.mean(C_rands)
    L_rand = np.mean(L_rands)

    if C_rand == 0 or L_rand == 0:
        return 0.0

    sigma = (C / C_rand) / (L / L_rand)
    return sigma


def modularity(G: nx.Graph) -> dict:
    """Compute modularity using Louvain community detection."""
    try:
        # Try the standard python-louvain import first
        try:
            import community.community_louvain as community_louvain
        except (ImportError, ModuleNotFoundError):
            import community as community_louvain
        # Louvain requires non-negative weights; use absolute values
        G_abs = G.copy()
        for u, v, d in G_abs.edges(data=True):
            if "weight" in d:
                d["weight"] = abs(d["weight"])
        partition = community_louvain.best_partition(G_abs, random_state=42)
        Q = community_louvain.modularity(partition, G_abs)
        n_communities = len(set(partition.values()))
        return {"modularity": Q, "n_communities": n_communities, "partition": partition}
    except (ImportError, ValueError, AttributeError):
        return {"modularity": 0.0, "n_communities": 0, "partition": {}}


def degree_distribution_stats(G: nx.Graph) -> dict:
    """Compute degree distribution statistics."""
    degrees = [d for _, d in G.degree()]
    if not degrees:
        return {"mean": 0, "std": 0, "skew": 0, "kurtosis": 0, "max": 0}
    return {
        "mean": np.mean(degrees),
        "std": np.std(degrees),
        "skew": float(stats.skew(degrees)) if len(degrees) > 2 else 0,
        "kurtosis": float(stats.kurtosis(degrees)) if len(degrees) > 3 else 0,
        "max": max(degrees),
    }


def weight_distribution_stats(weights: np.ndarray, mask: np.ndarray) -> dict:
    """Compute weight distribution statistics for active connections."""
    active = np.abs(weights[mask > 0])
    if len(active) == 0:
        return {"mean": 0, "std": 0, "skew": 0, "kurtosis": 0, "lognormal_fit": None}

    result = {
        "mean": float(active.mean()),
        "std": float(active.std()),
        "skew": float(stats.skew(active)) if len(active) > 2 else 0,
        "kurtosis": float(stats.kurtosis(active)) if len(active) > 3 else 0,
    }

    # Test lognormal fit (biological weight distributions are lognormal)
    if len(active) > 10 and active.min() > 0:
        try:
            shape, loc, scale = stats.lognorm.fit(active, floc=0)
            _, p_value = stats.kstest(active, "lognorm", args=(shape, loc, scale))
            result["lognormal_fit"] = {"shape": shape, "scale": scale, "ks_pvalue": p_value}
        except Exception:
            result["lognormal_fit"] = None
    else:
        result["lognormal_fit"] = None

    return result


def network_efficiency(G: nx.Graph) -> float:
    """Global efficiency: mean of inverse shortest path lengths."""
    if len(G) < 2:
        return 0.0
    return nx.global_efficiency(G)


def compute_topology_metrics(
    mask: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_random: int = 5,
    seed: int = 42,
    compute_expensive: bool = True,
) -> dict:
    """Compute comprehensive topology metrics for a single-layer mask.

    Args:
        mask: Binary (n_out, n_in) mask.
        weights: Optional weight matrix.
        n_random: Number of random graphs for sigma computation.
        seed: Random seed.
        compute_expensive: If True, compute path length and small-world (slow for large graphs).

    Returns:
        Dictionary of all topology metrics.
    """
    G = mask_to_graph(mask, weights)
    n_active = np.count_nonzero(mask)
    total = mask.size
    sparsity = 1.0 - n_active / total

    metrics = {
        "sparsity": sparsity,
        "n_edges": n_active,
        "n_nodes": len(G),
    }

    # Clustering
    cc = clustering_coefficient(G)
    metrics["clustering_local"] = cc["local_mean"]
    metrics["clustering_global"] = cc["global"]

    # Degree distribution
    dd = degree_distribution_stats(G)
    metrics["degree_mean"] = dd["mean"]
    metrics["degree_std"] = dd["std"]
    metrics["degree_skew"] = dd["skew"]

    # Weight distribution
    if weights is not None:
        wd = weight_distribution_stats(weights, mask)
        metrics["weight_mean"] = wd["mean"]
        metrics["weight_std"] = wd["std"]
        metrics["weight_skew"] = wd["skew"]
        metrics["weight_lognormal_p"] = (
            wd["lognormal_fit"]["ks_pvalue"] if wd["lognormal_fit"] else None
        )

    # Modularity
    mod = modularity(G)
    metrics["modularity"] = mod["modularity"]
    metrics["n_communities"] = mod["n_communities"]

    # Expensive metrics
    if compute_expensive and len(G) <= 2000:
        metrics["avg_path_length"] = average_path_length(G)
        metrics["small_world_sigma"] = small_world_coefficient(G, n_random, seed)
        metrics["efficiency"] = network_efficiency(G)
    else:
        metrics["avg_path_length"] = None
        metrics["small_world_sigma"] = None
        metrics["efficiency"] = None

    return metrics


def functional_connectivity_graph(
    mask: np.ndarray,
    weights: Optional[np.ndarray] = None,
    project_to: str = "output",
    density: float = 0.1,
) -> nx.Graph:
    """Create a functional connectivity graph from a layer mask.

    Computes cosine similarity between neurons' weight vectors and thresholds
    to keep the top connections, following the standard approach in neuroimaging
    studies (correlation-based functional connectivity with density threshold).

    Two neurons in the same set (input or output) are connected if their
    weight patterns (connections to the other layer) are sufficiently similar.
    This parallels how functional connectivity is measured in neuroscience:
    correlated activity patterns indicate functional coupling.

    Args:
        mask: Binary (n_out, n_in) mask matrix.
        weights: Optional weight matrix, same shape as mask.
        project_to: 'input' or 'output' — which set of neurons to project onto.
        density: Target density for the projected graph (fraction of possible
                 edges to keep). Default 0.1 matches neuroscience convention.

    Returns:
        NetworkX Graph with edges weighted by cosine similarity.
    """
    n_out, n_in = mask.shape
    binary = (mask > 0).astype(np.float32)

    # Use weights if available, otherwise binary mask
    W = np.abs(weights) * binary if weights is not None else binary

    if project_to == "input":
        vecs = W.T  # (n_in, n_out) — each input node's connections to outputs
    else:
        vecs = W    # (n_out, n_in) — each output node's connections to inputs

    n = vecs.shape[0]

    # Cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normed = vecs / norms
    sim = normed @ normed.T
    np.fill_diagonal(sim, 0)

    # Threshold to keep target density
    upper = sim[np.triu_indices(n, k=1)]
    if len(upper) == 0:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        return G

    n_keep = max(1, int(density * len(upper)))
    threshold = np.partition(upper, -n_keep)[-n_keep]
    threshold = max(threshold, 0.01)  # minimum similarity

    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = np.where((sim >= threshold) & (np.triu(np.ones_like(sim), k=1) > 0))
    for r, c in zip(rows, cols):
        G.add_edge(r, c, weight=float(sim[r, c]))

    return G


def compute_projected_metrics(
    mask: np.ndarray,
    weights: Optional[np.ndarray] = None,
    project_to: str = "output",
    n_random: int = 5,
    seed: int = 42,
    max_nodes_for_expensive: int = 600,
    projection_density: float = 0.1,
) -> dict:
    """Compute topology metrics on the functional connectivity graph.

    Converts the bipartite layer into a unipartite graph via cosine-similarity-
    based functional connectivity, then computes clustering, modularity,
    small-world coefficient, and other standard graph metrics.
    """
    G = functional_connectivity_graph(mask, weights, project_to, projection_density)

    metrics = {
        "n_nodes": len(G),
        "n_edges": G.number_of_edges(),
    }

    # Clustering
    cc = clustering_coefficient(G)
    metrics["clustering_local"] = cc["local_mean"]
    metrics["clustering_global"] = cc["global"]

    # Modularity
    mod = modularity(G)
    metrics["modularity"] = mod["modularity"]
    metrics["n_communities"] = mod["n_communities"]

    # Degree stats
    dd = degree_distribution_stats(G)
    metrics["degree_mean"] = dd["mean"]
    metrics["degree_std"] = dd["std"]

    # Expensive metrics
    if len(G) <= max_nodes_for_expensive:
        metrics["avg_path_length"] = average_path_length(G)
        metrics["small_world_sigma"] = small_world_coefficient(G, n_random, seed)
        metrics["efficiency"] = network_efficiency(G)
    else:
        metrics["avg_path_length"] = None
        metrics["small_world_sigma"] = None
        metrics["efficiency"] = None

    return metrics


def compute_multilayer_metrics(
    masks: list,
    layer_sizes: list,
    weights_list: Optional[list] = None,
    seed: int = 42,
) -> dict:
    """Compute topology metrics for a full multi-layer network.

    Uses projected graphs for meaningful clustering/small-world metrics,
    since feedforward layered graphs are triangle-free by construction.
    The projection computes functional connectivity: two neurons in the
    same layer are connected if they share neighbors in adjacent layers.
    """
    # Per-layer projected metrics (project each layer onto its output nodes)
    per_layer = []
    for i, mask in enumerate(masks):
        w = weights_list[i] if weights_list else None
        # Project onto output side (hidden/output neurons)
        proj = compute_projected_metrics(
            mask, w, project_to="output",
            max_nodes_for_expensive=600, seed=seed + i
        )
        proj["layer"] = i
        per_layer.append(proj)

    # Combined projected metrics: merge all hidden-layer projections
    # Use the first hidden layer projection as the primary metric
    # (it has the most connections and nodes)
    primary = per_layer[0] if per_layer else {}

    # Full-network graph for modularity (modularity works on layered graphs)
    G_full = masks_to_full_graph(masks, layer_sizes, weights_list)
    full_mod = modularity(G_full)

    full_metrics = {
        "n_nodes": sum(layer_sizes),
        "n_edges": G_full.number_of_edges(),
        # Use projected clustering (meaningful, non-zero)
        "clustering_local": primary.get("clustering_local", 0),
        "clustering_global": primary.get("clustering_global", 0),
        # Use full-graph modularity (works on layered graphs)
        "modularity": full_mod["modularity"],
        "n_communities": full_mod["n_communities"],
        # Use projected path length and sigma
        "avg_path_length": primary.get("avg_path_length", None),
        "small_world_sigma": primary.get("small_world_sigma", None),
        "efficiency": primary.get("efficiency", None),
    }

    return {
        "per_layer": per_layer,
        "full_network": full_metrics,
    }
