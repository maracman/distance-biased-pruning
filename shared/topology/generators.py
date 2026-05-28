"""
Sparse network topology generators.

Generates binary masks and weight matrices for initializing sparse neural network
layers. Each generator produces a (out_features, in_features) adjacency/mask matrix
representing which connections exist.

Methods:
    - Erdos-Renyi: random baseline
    - Watts-Strogatz: small-world baseline
    - Barabasi-Albert: scale-free baseline
    - Bio-inspired: inverse-square distance + spatial embedding (our method)
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist


def _graph_to_mask(G: nx.Graph, n_out: int, n_in: int) -> np.ndarray:
    """Convert a networkx graph to a (n_out, n_in) binary mask.

    Nodes 0..n_in-1 are input neurons, n_in..n_in+n_out-1 are output neurons.
    Mask[i, j] = 1 if there is an edge from input j to output i.
    """
    mask = np.zeros((n_out, n_in), dtype=np.float32)
    for u, v in G.edges():
        if u < n_in and v >= n_in:
            mask[v - n_in, u] = 1.0
        elif v < n_in and u >= n_in:
            mask[u - n_in, v] = 1.0
    return mask


def compute_sparsity(mask: np.ndarray) -> float:
    """Fraction of zero entries in mask."""
    return 1.0 - np.count_nonzero(mask) / mask.size


def generate_erdos_renyi_mask(
    n_out: int, n_in: int, target_sparsity: float = 0.9, seed: int = 42
) -> np.ndarray:
    """Generate a random sparse mask via Erdos-Renyi model."""
    rng = np.random.RandomState(seed)
    p = 1.0 - target_sparsity
    mask = (rng.random((n_out, n_in)) < p).astype(np.float32)
    return mask


def generate_watts_strogatz_mask(
    n_out: int, n_in: int, target_sparsity: float = 0.9,
    rewire_prob: float = 0.1, seed: int = 42
) -> np.ndarray:
    """Generate a small-world sparse mask via Watts-Strogatz on a bipartite layout."""
    n_total = n_in + n_out
    desired_edges = int((1.0 - target_sparsity) * n_out * n_in)
    k = max(2, int(2 * desired_edges / n_total))
    if k % 2 != 0:
        k += 1
    k = min(k, n_total - 1)

    G = nx.watts_strogatz_graph(n_total, k, rewire_prob, seed=seed)
    mask = _graph_to_mask(G, n_out, n_in)

    # Adjust to target sparsity
    actual_sparsity = compute_sparsity(mask)
    rng = np.random.RandomState(seed + 1)
    if actual_sparsity > target_sparsity:
        # Need more connections
        zeros = np.argwhere(mask == 0)
        n_add = int((actual_sparsity - target_sparsity) * mask.size)
        if n_add > 0 and len(zeros) > 0:
            idx = rng.choice(len(zeros), min(n_add, len(zeros)), replace=False)
            for i in idx:
                mask[zeros[i][0], zeros[i][1]] = 1.0
    elif actual_sparsity < target_sparsity:
        # Need fewer connections
        ones = np.argwhere(mask == 1)
        n_remove = int((target_sparsity - actual_sparsity) * mask.size)
        if n_remove > 0 and len(ones) > 0:
            idx = rng.choice(len(ones), min(n_remove, len(ones)), replace=False)
            for i in idx:
                mask[ones[i][0], ones[i][1]] = 0.0

    return mask


def generate_barabasi_albert_mask(
    n_out: int, n_in: int, target_sparsity: float = 0.9, seed: int = 42
) -> np.ndarray:
    """Generate a scale-free sparse mask via Barabasi-Albert."""
    n_total = n_in + n_out
    desired_edges = int((1.0 - target_sparsity) * n_out * n_in)
    m = max(1, desired_edges // n_total)

    G = nx.barabasi_albert_graph(n_total, m, seed=seed)
    mask = _graph_to_mask(G, n_out, n_in)

    # Adjust sparsity
    actual_sparsity = compute_sparsity(mask)
    rng = np.random.RandomState(seed + 1)
    if actual_sparsity > target_sparsity:
        zeros = np.argwhere(mask == 0)
        n_add = int((actual_sparsity - target_sparsity) * mask.size)
        if n_add > 0 and len(zeros) > 0:
            idx = rng.choice(len(zeros), min(n_add, len(zeros)), replace=False)
            for i in idx:
                mask[zeros[i][0], zeros[i][1]] = 1.0
    elif actual_sparsity < target_sparsity:
        ones = np.argwhere(mask == 1)
        n_remove = int((target_sparsity - actual_sparsity) * mask.size)
        if n_remove > 0 and len(ones) > 0:
            idx = rng.choice(len(ones), min(n_remove, len(ones)), replace=False)
            for i in idx:
                mask[ones[i][0], ones[i][1]] = 0.0

    return mask


def generate_bio_inspired_mask(
    n_out: int,
    n_in: int,
    target_sparsity: float = 0.9,
    distance_exponent: float = 2.0,
    dimensions: int = 2,
    seed: int = 42,
    return_weights: bool = False,
    return_positions: bool = False,
):
    """Generate a biologically-inspired sparse mask using inverse-square distance connectivity.

    Neurons are embedded in a spatial field. Connection probability follows
    P(i->j) ~ 1 / d(i,j)^distance_exponent. Weights are proportional to
    1/d, producing a biologically realistic heavy-tailed distribution.

    Args:
        n_out: Number of output neurons.
        n_in: Number of input neurons.
        target_sparsity: Target fraction of zero connections.
        distance_exponent: Exponent for distance decay (2.0 = inverse-square).
        dimensions: Spatial embedding dimensions (2 or 3).
        seed: Random seed.
        return_weights: If True, return (mask, weights) where weights encode distance.
        return_positions: If True, also return neuron positions.

    Returns:
        mask: Binary (n_out, n_in) array.
        weights (optional): Float (n_out, n_in) array with distance-based weights.
        positions (optional): Dict with 'input' and 'output' position arrays.
    """
    rng = np.random.RandomState(seed)

    # Embed neurons in spatial field
    # Input neurons on the "left" side, output on the "right"
    input_pos = rng.uniform(0, 0.5, (n_in, dimensions))
    input_pos[:, 0] = rng.uniform(0, 0.3, n_in)  # x-position clustered left

    output_pos = rng.uniform(0.5, 1.0, (n_out, dimensions))
    output_pos[:, 0] = rng.uniform(0.7, 1.0, n_out)  # x-position clustered right

    # Compute pairwise distances: (n_out, n_in)
    distances = cdist(output_pos, input_pos)

    # Connection probability: inverse distance law
    epsilon = 1e-6
    connection_prob = 1.0 / (distances ** distance_exponent + epsilon)

    # Normalize probabilities to achieve target density
    target_density = 1.0 - target_sparsity
    # Scale so mean probability matches target density
    scale = target_density / connection_prob.mean()
    connection_prob = np.clip(connection_prob * scale, 0, 1)

    # Sample connections
    mask = (rng.random((n_out, n_in)) < connection_prob).astype(np.float32)

    results = [mask]

    if return_weights:
        # Weights proportional to inverse distance (stronger for closer neurons)
        weights = mask / (distances + epsilon)
        # Normalize to reasonable range
        weights = weights / (weights.max() + epsilon)
        results.append(weights)

    if return_positions:
        results.append({"input": input_pos, "output": output_pos})

    return results[0] if len(results) == 1 else tuple(results)


def make_grid_positions(n_pixels: int, image_width: int = 28) -> np.ndarray:
    """Create 2D positions from a pixel grid (e.g., MNIST 28×28).

    Pixel i maps to (col, row) normalized to [0, 1]. This gives input neurons
    spatially meaningful positions so that "tight clusters" = local receptive fields.

    Args:
        n_pixels: Number of pixels (784 for MNIST).
        image_width: Width of square image (28 for MNIST).

    Returns:
        positions: (n_pixels, 2) array of (x, y) positions in [0, 1].
    """
    positions = np.zeros((n_pixels, 2))
    for i in range(n_pixels):
        positions[i, 0] = (i % image_width) / (image_width - 1)   # x = column
        positions[i, 1] = (i // image_width) / (image_width - 1)  # y = row
    return positions


def make_adjacency_positions(
    prev_positions: np.ndarray,
    prev_mask: np.ndarray,
    noise_std: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Derive neuron positions from adjacency to the previous layer.

    Each neuron's position is the weighted centroid of its connected inputs'
    positions, plus Gaussian noise for resolution. This makes graph adjacency
    approximate spatial distance: neurons connected to similar inputs end up
    nearby, like biological topographic maps.

    Args:
        prev_positions: (n_in, d) positions of the previous layer's neurons.
        prev_mask: (n_out, n_in) binary mask from the previous layer.
        noise_std: Std of Gaussian noise added for spatial resolution.
        seed: Random seed for noise.

    Returns:
        positions: (n_out, d) array of derived positions.
    """
    rng = np.random.RandomState(seed)
    n_out = prev_mask.shape[0]
    d = prev_positions.shape[1]
    positions = np.zeros((n_out, d))

    for i in range(n_out):
        connected = prev_mask[i] > 0
        if connected.any():
            # Centroid of connected input positions
            positions[i] = prev_positions[connected].mean(axis=0)
        else:
            # Disconnected neuron: random position
            positions[i] = rng.uniform(0, 1, d)

    # Add noise for resolution
    positions += rng.normal(0, noise_std, positions.shape)
    return positions


def generate_locality_mask(
    n_out: int,
    n_in: int,
    target_sparsity: float = 0.9,
    locality: float = 1.0,
    input_positions: np.ndarray = None,
    output_positions: np.ndarray = None,
    seed: int = 42,
    return_weights: bool = False,
) -> np.ndarray:
    """Generate sparse mask with controlled locality of receptive fields.

    Each output neuron gets exactly k connections (balanced fan-in).
    Of those, k_local are the nearest inputs by Euclidean distance, and
    k_random are drawn uniformly from the remaining inputs.

    This cleanly interpolates between:
        - locality=1.0: pure local receptive fields (like convolution)
        - locality=0.0: pure random connections (no spatial structure)

    Unlike weight-variability pruning, this gives direct, interpretable control
    over the spatial extent of each neuron's receptive field.

    Args:
        n_out: Number of output neurons.
        n_in: Number of input neurons.
        target_sparsity: Fraction of zero connections.
        locality: Fraction of each neuron's connections that are local (nearest).
            1.0 = all local, 0.0 = all random.
        input_positions: (n_in, d) positions. If None, uses random positions.
        output_positions: (n_out, d) positions. If None, uses random positions.
        seed: Random seed.
        return_weights: If True, return Kaiming-scaled initial weights.

    Returns:
        mask: Binary (n_out, n_in) array.
        weights (optional): (n_out, n_in) initial weight array.
    """
    rng = np.random.RandomState(seed)

    # Positions
    if input_positions is None:
        input_positions = rng.uniform(0, 1, (n_in, 2))
    if output_positions is None:
        output_positions = rng.uniform(0, 1, (n_out, 2))

    distances = cdist(output_positions, input_positions)

    # Connections per output neuron
    n_total = n_out * n_in
    n_keep = int((1.0 - target_sparsity) * n_total)
    k_per_row = n_keep // n_out
    remainder = n_keep - k_per_row * n_out

    mask = np.zeros((n_out, n_in), dtype=np.float32)

    for h in range(n_out):
        k = k_per_row + (1 if h < remainder else 0)
        k_local = int(locality * k)
        k_random = k - k_local

        chosen = set()

        # Pick k_local nearest inputs by distance
        if k_local > 0:
            nearest = np.argsort(distances[h])[:k_local]
            chosen.update(nearest)

        # Pick k_random uniformly from remaining inputs
        if k_random > 0:
            available = np.array(list(set(range(n_in)) - chosen))
            if len(available) > 0:
                picks = rng.choice(available, min(k_random, len(available)), replace=False)
                chosen.update(picks)

        for idx in chosen:
            mask[h, idx] = 1.0

    # Optional: return Kaiming-scaled initial weights
    init_weights = None
    if return_weights:
        mean_fan_in = max(mask.sum(axis=1).mean(), 1.0)
        target_std = np.sqrt(2.0 / mean_fan_in)
        init_weights = rng.normal(0, target_std, (n_out, n_in)).astype(np.float32)
        init_weights *= mask  # zero out non-connections

    if return_weights:
        return mask, init_weights
    return mask


def generate_distance_pruned_mask(
    n_out: int,
    n_in: int,
    target_sparsity: float = 0.9,
    weight_variability: float = 0.0,
    distance_exponent: float = 2.0,
    dimensions: int = 2,
    seed: int = 42,
    return_distances: bool = False,
    return_positions: bool = False,
    balanced: bool = False,
    return_weights: bool = False,
    input_positions: np.ndarray = None,
    output_positions: np.ndarray = None,
    position_noise: float = 0.0,
) -> np.ndarray:
    """Generate sparse mask by pruning a fully-connected network using weight × distance.

    Start with ALL connections, assign weights with controlled variability,
    compute pruning score = weight_magnitude × (1/distance^exponent), and keep
    only the top-scoring connections to reach target sparsity.

    The weight_variability parameter controls cluster tightness:
        - 0.0: all weights identical → score = 1/d^α → purely spatial pruning → tight clusters
        - 1.0: weights ~ Uniform(0,1) → weight dominates for some connections → loose clusters
        - intermediate: smooth blend between spatial and weight-driven pruning

    Position handling:
        - For input layers: pass input_positions from data structure (e.g., pixel grid).
          "Tight clusters" then means local receptive fields over the data.
        - For hidden layers in multi-layer networks: pass positions derived from
          adjacency to the previous layer (+ noise for resolution).
        - If no positions given: falls back to random spatial embedding.

    Args:
        n_out: Number of output neurons.
        n_in: Number of input neurons.
        target_sparsity: Target fraction of zero connections (e.g., 0.9 = keep 10%).
        weight_variability: Controls weight spread. 0.0 = uniform weights (tight clusters),
            1.0 = full uniform random spread (loose clusters).
        distance_exponent: Exponent for distance decay (2.0 = inverse-square).
        dimensions: Spatial embedding dimensions.
        seed: Random seed.
        return_distances: If True, also return the distance matrix.
        return_positions: If True, also return neuron positions.
        input_positions: (n_in, d) array of input neuron positions. If None, random.
        output_positions: (n_out, d) array of output neuron positions. If None, random.
        position_noise: Std of Gaussian noise added to provided positions (for resolution).

    Returns:
        mask: Binary (n_out, n_in) array.
        distances (optional): (n_out, n_in) distance matrix.
        positions (optional): Dict with 'input' and 'output' position arrays.
    """
    rng = np.random.RandomState(seed)

    # --- Neuron positions ---
    if input_positions is not None:
        input_pos = input_positions.copy()
        if position_noise > 0:
            input_pos += rng.normal(0, position_noise, input_pos.shape)
    else:
        # Fallback: random positions (legacy behavior)
        input_pos = rng.uniform(0, 0.5, (n_in, dimensions))
        input_pos[:, 0] = rng.uniform(0, 0.3, n_in)

    if output_positions is not None:
        output_pos = output_positions.copy()
        if position_noise > 0:
            output_pos += rng.normal(0, position_noise, output_pos.shape)
    else:
        # Fallback: random positions (legacy behavior)
        output_pos = rng.uniform(0.5, 1.0, (n_out, dimensions))
        output_pos[:, 0] = rng.uniform(0.7, 1.0, n_out)

    # Compute pairwise distances: (n_out, n_in)
    distances = cdist(output_pos, input_pos)
    epsilon = 1e-6

    # Spatial score: inverse distance (closer = higher score)
    spatial_score = 1.0 / (distances ** distance_exponent + epsilon)
    # Normalize to [0, 1]
    spatial_score = spatial_score / (spatial_score.max() + epsilon)

    # Random weight component with controlled variability
    if weight_variability <= 0.0:
        # All weights identical → pure spatial pruning
        weight_score = np.ones((n_out, n_in))
    else:
        # Weights centered at 1.0 with spread controlled by variability
        # At variability=1.0: Uniform(0, 2) → full spread
        # At variability=0.5: Uniform(0.5, 1.5) → moderate spread
        half_range = weight_variability
        weight_score = rng.uniform(1.0 - half_range, 1.0 + half_range, (n_out, n_in))
        weight_score = np.clip(weight_score, 0.0, None)

    # Combined pruning score: weight × spatial
    pruning_score = weight_score * spatial_score

    # Keep top connections to achieve target sparsity
    n_total = n_out * n_in
    n_keep = int((1.0 - target_sparsity) * n_total)

    if balanced:
        # Per-row top-k: each output neuron keeps its best connections.
        # This ensures uniform fan-in across all neurons, eliminating the
        # confound where tight-cluster topologies leave many neurons disconnected.
        k_per_row = n_keep // n_out
        remainder = n_keep - k_per_row * n_out
        mask = np.zeros((n_out, n_in), dtype=np.float32)
        for row in range(n_out):
            row_scores = pruning_score[row]
            # Each row gets k_per_row connections (+ 1 extra for first 'remainder' rows)
            k = k_per_row + (1 if row < remainder else 0)
            k = min(k, n_in)  # can't keep more than available
            if k > 0:
                top_idx = np.argpartition(row_scores, -k)[-k:]
                mask[row, top_idx] = 1.0
    else:
        # Global top-k: keep highest-scoring connections regardless of which neuron
        flat_scores = pruning_score.flatten()
        # Find the threshold: keep top n_keep scores
        if n_keep >= n_total:
            threshold = 0.0
        else:
            threshold = np.partition(flat_scores, -n_keep)[-n_keep]

        mask = (pruning_score >= threshold).astype(np.float32)

        # If we have too many (ties at threshold), randomly remove excess
        n_active = int(mask.sum())
        if n_active > n_keep:
            tied_idx = np.argwhere(pruning_score.flatten() == threshold).flatten()
            n_remove = n_active - n_keep
            remove_idx = rng.choice(tied_idx, n_remove, replace=False)
            flat_mask = mask.flatten()
            flat_mask[remove_idx] = 0.0
            mask = flat_mask.reshape(n_out, n_in)

    # Optionally return the topology-aligned initial weights, renormalized.
    # The weight_score values that survived pruning encode the topology—keeping
    # them preserves the weight-distance alignment. But different wv levels produce
    # different weight distributions, so we renormalize all surviving weights to
    # have the same scale (Kaiming-appropriate for sparse fan-in).
    init_weights = None
    if return_weights:
        # Start with the raw weight scores that created this topology
        raw_weights = weight_score * mask
        # Assign random signs (the original weights were magnitudes only)
        signs = rng.choice([-1.0, 1.0], size=raw_weights.shape).astype(np.float32)
        signed_weights = raw_weights * signs
        # Renormalize surviving weights to Kaiming scale for this fan-in
        active_per_row = mask.sum(axis=1)
        mean_fan_in = max(active_per_row.mean(), 1.0)
        target_std = np.sqrt(2.0 / mean_fan_in)
        # Get current std of surviving weights
        surviving = signed_weights[mask > 0]
        if len(surviving) > 0 and surviving.std() > 0:
            current_std = surviving.std()
            signed_weights = signed_weights * (target_std / current_std)
            # Zero-center the surviving weights
            surviving_mean = signed_weights[mask > 0].mean()
            signed_weights[mask > 0] -= surviving_mean
        init_weights = signed_weights.astype(np.float32)

    results = [mask]
    if return_weights:
        results.append(init_weights)
    if return_distances:
        results.append(distances)
    if return_positions:
        results.append({"input": input_pos, "output": output_pos})

    return results[0] if len(results) == 1 else tuple(results)


def generate_bio_inspired_mask_multilayer(
    layer_sizes: list,
    target_sparsity: float = 0.9,
    distance_exponent: float = 2.0,
    dimensions: int = 2,
    seed: int = 42,
):
    """Generate bio-inspired masks for a full multi-layer network.

    Neurons across all layers are embedded in a shared spatial field with
    x-positions arranged by layer depth. This produces inter-layer connectivity
    that respects spatial locality across the full network.

    Args:
        layer_sizes: List of layer widths [input, hidden1, ..., output].
        target_sparsity: Target sparsity for each layer.
        distance_exponent: Distance decay exponent.
        dimensions: Spatial embedding dimensions.
        seed: Random seed.

    Returns:
        masks: List of (layer_sizes[i+1], layer_sizes[i]) binary masks.
        all_positions: List of position arrays for each layer.
    """
    rng = np.random.RandomState(seed)
    n_layers = len(layer_sizes)

    # Embed all neurons in shared spatial field
    all_positions = []
    for layer_idx, size in enumerate(layer_sizes):
        x_center = layer_idx / (n_layers - 1)  # evenly spaced along x-axis
        x_spread = 0.05
        pos = rng.uniform(0, 1, (size, dimensions))
        pos[:, 0] = rng.normal(x_center, x_spread, size)
        pos[:, 0] = np.clip(pos[:, 0], 0, 1)
        all_positions.append(pos)

    # Generate masks for adjacent layers
    masks = []
    for i in range(n_layers - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i + 1]

        distances = cdist(all_positions[i + 1], all_positions[i])
        epsilon = 1e-6
        connection_prob = 1.0 / (distances ** distance_exponent + epsilon)

        target_density = 1.0 - target_sparsity
        scale = target_density / connection_prob.mean()
        connection_prob = np.clip(connection_prob * scale, 0, 1)

        mask = (rng.random((n_out, n_in)) < connection_prob).astype(np.float32)
        masks.append(mask)

    return masks, all_positions
