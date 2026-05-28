"""
Developmental pruning for biologically-inspired sparse networks.

Implements the over-build-then-prune principle from neurodevelopment:
1. Start with a denser-than-target network
2. Apply iterative cycles of: weight decay -> Pareto reinforcement -> probabilistic pruning
3. Result: sparse network with biologically realistic properties

This mirrors the critical-period pruning experiments and the
developmental trajectory modeled in Notebook 6.
"""

import numpy as np
from typing import Optional


def weight_decay(weights: np.ndarray, mask: np.ndarray, decay_rate: float) -> np.ndarray:
    """Apply uniform weight decay (metabolic maintenance cost)."""
    weights = weights * (1.0 - decay_rate)
    weights = weights * mask  # maintain sparsity pattern
    return weights


def pareto_reinforce(
    weights: np.ndarray,
    mask: np.ndarray,
    reinforcement_total: float = 1.0,
    pareto_fraction: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    """Apply Pareto-principle reinforcement: top fraction gets majority of reinforcement.

    Models activity-dependent plasticity: strong connections get stronger.
    Top 20% of edges by weight receive 80% of reinforcement budget.
    Bottom 80% share remaining 20%.
    """
    active = np.abs(weights) * mask
    active_flat = active.flatten()
    nonzero_idx = np.where(active_flat > 0)[0]

    if len(nonzero_idx) == 0:
        return weights

    nonzero_weights = active_flat[nonzero_idx]
    threshold = np.percentile(nonzero_weights, (1.0 - pareto_fraction) * 100)

    top_mask = nonzero_weights >= threshold
    bottom_mask = ~top_mask

    n_top = top_mask.sum()
    n_bottom = bottom_mask.sum()

    reinforcement = np.zeros_like(active_flat)
    top_share = 0.8 * reinforcement_total
    bottom_share = 0.2 * reinforcement_total

    if n_top > 0:
        reinforcement[nonzero_idx[top_mask]] = top_share / n_top
    if n_bottom > 0:
        reinforcement[nonzero_idx[bottom_mask]] = bottom_share / n_bottom

    weights_flat = weights.flatten()
    signs = np.sign(weights_flat)
    weights_flat = np.abs(weights_flat) + reinforcement
    weights_flat = weights_flat * signs
    weights = weights_flat.reshape(weights.shape)
    weights = weights * mask

    return weights


def prune_probabilistic(
    weights: np.ndarray,
    mask: np.ndarray,
    target_removal_fraction: float = 0.15,
    steepness: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    """Probabilistic pruning using sigmoid survival probability.

    Survival probability is a sigmoid function of normalized weight magnitude.
    Stronger connections are more likely to survive.
    Steepness controls how sharply survival depends on weight
    (high = nearly deterministic, low = more stochastic).
    """
    rng = np.random.RandomState(seed)

    active = np.abs(weights) * mask
    active_flat = active.flatten()
    nonzero_idx = np.where(active_flat > 0)[0]

    if len(nonzero_idx) == 0:
        return mask.copy()

    nonzero_weights = active_flat[nonzero_idx]

    # Normalize weights to [0, 1]
    w_min, w_max = nonzero_weights.min(), nonzero_weights.max()
    if w_max > w_min:
        w_norm = (nonzero_weights - w_min) / (w_max - w_min)
    else:
        w_norm = np.ones_like(nonzero_weights) * 0.5

    # Sigmoid survival probability centered at median
    survival_prob = 1.0 / (1.0 + np.exp(-steepness * (w_norm - 0.5)))

    # Scale to achieve target removal fraction
    # Adjust the offset to get approximately target_removal_fraction deaths
    current_removal = 1.0 - survival_prob.mean()
    if current_removal > 0:
        scale = target_removal_fraction / current_removal
        survival_prob = 1.0 - (1.0 - survival_prob) * min(scale, 2.0)
        survival_prob = np.clip(survival_prob, 0.01, 0.99)

    # Sample survival
    survives = rng.random(len(nonzero_idx)) < survival_prob

    new_mask = mask.copy().flatten()
    for i, idx in enumerate(nonzero_idx):
        if not survives[i]:
            new_mask[idx] = 0.0

    return new_mask.reshape(mask.shape)


def prune_by_percentile(
    weights: np.ndarray, mask: np.ndarray, percentile: float = 20.0
) -> np.ndarray:
    """Deterministic pruning: remove all connections below weight percentile."""
    active = np.abs(weights) * mask
    active_flat = active.flatten()
    nonzero_vals = active_flat[active_flat > 0]

    if len(nonzero_vals) == 0:
        return mask.copy()

    threshold = np.percentile(nonzero_vals, percentile)

    new_mask = mask.copy()
    new_mask[active < threshold] = 0.0
    return new_mask


def spatial_reinforce(
    weights: np.ndarray,
    mask: np.ndarray,
    distances: np.ndarray,
    locality_bias: float = 0.5,
    reinforcement_total: float = 1.0,
    pareto_fraction: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    """Spatially-aware Pareto reinforcement.

    Blends weight-magnitude and spatial-proximity criteria to control
    cluster tightness while keeping connection count constant.

    Args:
        weights: Current weight matrix (n_out, n_in).
        mask: Binary mask.
        distances: Pairwise distance matrix (n_out, n_in) from spatial embedding.
        locality_bias: 0.0 = reinforce by weight magnitude only (loose clusters),
                       1.0 = reinforce by spatial proximity only (tight dense clusters).
                       Values in between blend both criteria.
        reinforcement_total: Total reinforcement budget.
        pareto_fraction: Top fraction that receives majority of reinforcement.
        seed: Random seed.

    Returns:
        Updated weights with reinforcement applied.
    """
    active = np.abs(weights) * mask
    active_flat = active.flatten()
    nonzero_idx = np.where(active_flat > 0)[0]

    if len(nonzero_idx) == 0:
        return weights

    # Score by weight magnitude (normalized to [0, 1])
    nonzero_weights = active_flat[nonzero_idx]
    w_min, w_max = nonzero_weights.min(), nonzero_weights.max()
    if w_max > w_min:
        weight_score = (nonzero_weights - w_min) / (w_max - w_min)
    else:
        weight_score = np.ones_like(nonzero_weights) * 0.5

    # Score by spatial proximity (inverse distance, normalized to [0, 1])
    dist_flat = distances.flatten()
    nonzero_dists = dist_flat[nonzero_idx]
    proximity = 1.0 / (nonzero_dists + 1e-6)
    p_min, p_max = proximity.min(), proximity.max()
    if p_max > p_min:
        proximity_score = (proximity - p_min) / (p_max - p_min)
    else:
        proximity_score = np.ones_like(proximity) * 0.5

    # Blend: combined score determines who gets reinforced
    combined_score = (1.0 - locality_bias) * weight_score + locality_bias * proximity_score

    # Top fraction by combined score gets the lion's share
    threshold = np.percentile(combined_score, (1.0 - pareto_fraction) * 100)
    top_mask = combined_score >= threshold
    bottom_mask = ~top_mask

    n_top = top_mask.sum()
    n_bottom = bottom_mask.sum()

    reinforcement = np.zeros_like(active_flat)
    top_share = 0.8 * reinforcement_total
    bottom_share = 0.2 * reinforcement_total

    if n_top > 0:
        reinforcement[nonzero_idx[top_mask]] = top_share / n_top
    if n_bottom > 0:
        reinforcement[nonzero_idx[bottom_mask]] = bottom_share / n_bottom

    weights_flat = weights.flatten()
    signs = np.sign(weights_flat)
    weights_flat = np.abs(weights_flat) + reinforcement
    weights_flat = weights_flat * signs
    weights = weights_flat.reshape(weights.shape)
    weights = weights * mask

    return weights


def developmental_pruning(
    weights: np.ndarray,
    mask: np.ndarray,
    n_cycles: int = 5,
    decay_rate: float = 0.05,
    reinforcement_total: float = 1.0,
    pareto_fraction: float = 0.2,
    prune_fraction: float = 0.15,
    steepness: float = 10.0,
    seed: int = 42,
    track_history: bool = False,
    distances: np.ndarray = None,
    locality_bias: float = None,
) -> dict:
    """Execute full developmental pruning trajectory.

    Iterates cycles of: decay -> Pareto reinforce -> probabilistic prune.
    Models biological neurodevelopment where networks are over-built then
    refined through activity-dependent processes.

    Args:
        distances: Optional (n_out, n_in) distance matrix for spatial reinforcement.
        locality_bias: If provided (with distances), use spatially-aware reinforcement.
            0.0 = weight-only (loose clusters), 1.0 = proximity-only (tight clusters).

    Returns:
        dict with 'weights', 'mask', and optionally 'history' (list of
        per-cycle metrics).
    """
    use_spatial = distances is not None and locality_bias is not None
    history = [] if track_history else None

    for cycle in range(n_cycles):
        cycle_seed = seed + cycle * 100

        # Step 1: Weight decay
        weights = weight_decay(weights, mask, decay_rate)

        # Step 2: Reinforcement (spatial-aware or weight-only)
        if use_spatial:
            weights = spatial_reinforce(
                weights, mask, distances, locality_bias,
                reinforcement_total, pareto_fraction, cycle_seed,
            )
        else:
            weights = pareto_reinforce(
                weights, mask, reinforcement_total, pareto_fraction, cycle_seed
            )

        # Step 3: Probabilistic pruning
        mask = prune_probabilistic(
            weights, mask, prune_fraction, steepness, cycle_seed + 1
        )

        # Zero out pruned weights
        weights = weights * mask

        if track_history:
            n_active = np.count_nonzero(mask)
            total = mask.size
            active_weights = np.abs(weights[mask > 0])
            history.append({
                "cycle": cycle + 1,
                "n_edges": n_active,
                "sparsity": 1.0 - n_active / total,
                "mean_weight": active_weights.mean() if len(active_weights) > 0 else 0,
                "weight_std": active_weights.std() if len(active_weights) > 0 else 0,
            })

    result = {"weights": weights, "mask": mask}
    if track_history:
        result["history"] = history
    return result


def developmental_pruning_with_input_drive(
    weights: np.ndarray,
    mask: np.ndarray,
    input_activity: Optional[np.ndarray] = None,
    n_cycles: int = 5,
    decay_rate: float = 0.05,
    reinforcement_total: float = 1.0,
    pareto_fraction: float = 0.2,
    prune_fraction: float = 0.15,
    steepness: float = 10.0,
    input_boost: float = 1.5,
    seed: int = 42,
) -> dict:
    """Developmental pruning with input-driven plasticity.

    Connections from highly active input neurons receive additional
    reinforcement, modeling experience-dependent development.
    """
    for cycle in range(n_cycles):
        cycle_seed = seed + cycle * 100

        weights = weight_decay(weights, mask, decay_rate)
        weights = pareto_reinforce(
            weights, mask, reinforcement_total, pareto_fraction, cycle_seed
        )

        # Input-driven boost: scale columns by input activity
        if input_activity is not None:
            activity_scale = input_activity / (input_activity.mean() + 1e-8)
            activity_scale = np.clip(activity_scale, 0.5, input_boost)
            weights = weights * activity_scale[np.newaxis, :]

        mask = prune_probabilistic(
            weights, mask, prune_fraction, steepness, cycle_seed + 1
        )
        weights = weights * mask

    return {"weights": weights, "mask": mask}


def repair_connectivity(
    mask: np.ndarray,
    distances: np.ndarray,
    min_fan_in: int = 1,
    min_fan_out: int = 1,
) -> tuple:
    """Repair a sparse mask to ensure no nodes are orphaned.

    After probabilistic pruning, some nodes may lose all connections.
    This repairs the mask by adding minimum-distance edges, mimicking
    how biological pruning preserves baseline connectivity.

    Args:
        mask: Binary (n_out, n_in) array.
        distances: (n_out, n_in) pairwise distance matrix.
        min_fan_in: Minimum connections per output neuron (row).
        min_fan_out: Minimum connections per input neuron (column).

    Returns:
        (repaired_mask, n_repaired) tuple.
    """
    mask = mask.copy()
    n_repaired = 0
    n_out, n_in = mask.shape

    # 1. Repair orphaned output neurons (rows with insufficient fan-in)
    for i in range(n_out):
        while mask[i].sum() < min_fan_in:
            zeros_j = np.where(mask[i] == 0)[0]
            if len(zeros_j) == 0:
                break
            best_j = zeros_j[np.argmin(distances[i, zeros_j])]
            mask[i, best_j] = 1.0
            n_repaired += 1

    # 2. Repair orphaned input neurons (columns with insufficient fan-out)
    for j in range(n_in):
        while mask[:, j].sum() < min_fan_out:
            zeros_i = np.where(mask[:, j] == 0)[0]
            if len(zeros_i) == 0:
                break
            best_i = zeros_i[np.argmin(distances[zeros_i, j])]
            mask[best_i, j] = 1.0
            n_repaired += 1

    return mask, n_repaired


def bridge_disconnected_components(
    mask: np.ndarray,
    distances: np.ndarray,
) -> tuple:
    """Detect and bridge disconnected components in a bipartite graph.

    Uses scipy's connected_components on the full bipartite adjacency.
    If multiple components exist, adds minimum-distance edges to connect
    each isolated component to the largest one.

    Args:
        mask: Binary (n_out, n_in) array.
        distances: (n_out, n_in) pairwise distance matrix.

    Returns:
        (repaired_mask, n_components_found, n_bridges_added) tuple.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n_out, n_in = mask.shape
    n_total = n_out + n_in

    # Build bipartite adjacency: input nodes 0..n_in-1, output nodes n_in..n_total-1
    edge_locs = np.argwhere(mask > 0)
    if len(edge_locs) == 0:
        return mask.copy(), 0, 0

    rows = np.concatenate([edge_locs[:, 1], n_in + edge_locs[:, 0]])
    cols = np.concatenate([n_in + edge_locs[:, 0], edge_locs[:, 1]])
    adj = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_total, n_total))

    n_components, labels = connected_components(adj, directed=False)

    if n_components <= 1:
        return mask.copy(), 1, 0

    mask = mask.copy()
    n_bridges = 0

    # Bridge each component to the largest
    component_sizes = np.bincount(labels)
    main_component = int(np.argmax(component_sizes))

    for comp in range(n_components):
        if comp == main_component:
            continue

        # Boolean masks for nodes in this component vs main
        comp_out = np.array([labels[n_in + i] == comp for i in range(n_out)])
        comp_in = np.array([labels[j] == comp for j in range(n_in)])
        main_out = np.array([labels[n_in + i] == main_component for i in range(n_out)])
        main_in = np.array([labels[j] == main_component for j in range(n_in)])

        # Candidate cross-component edges (where mask == 0)
        empty = (1.0 - mask)
        candidates = (np.outer(comp_out, main_in) + np.outer(main_out, comp_in))
        candidates = np.clip(candidates, 0, 1) * empty

        if candidates.sum() > 0:
            # Mask out non-candidates with inf distance
            candidate_dists = np.where(candidates > 0, distances, np.inf)
            best_idx = np.unravel_index(np.argmin(candidate_dists), distances.shape)
            mask[best_idx] = 1.0
            n_bridges += 1

    return mask, n_components, n_bridges


def developmental_pruning_connected(
    weights: np.ndarray,
    mask: np.ndarray,
    distances: np.ndarray,
    n_cycles: int = 10,
    decay_rate: float = 0.05,
    reinforcement_total: float = 1.0,
    pareto_fraction: float = 0.2,
    prune_fraction: float = 0.15,
    steepness: float = 10.0,
    seed: int = 42,
    track_history: bool = False,
    locality_bias: float = None,
    min_fan_in: int = 1,
    min_fan_out: int = 1,
    check_components: bool = True,
) -> dict:
    """Developmental pruning with connectivity preservation after each cycle.

    Identical to developmental_pruning, but after each probabilistic prune step:
    1. Repairs orphaned nodes (adds minimum-distance edges)
    2. Optionally bridges disconnected components

    This models biological critical-period pruning where connectivity is always
    maintained \u2014 neurons never become completely isolated.

    Repaired edges receive weight 0, which means prune_probabilistic ignores
    them in subsequent cycles (it only considers nonzero-weight edges). This
    is a natural \u201clifeline\u201d mechanism: structurally essential connections are
    preserved but must earn their keep through reinforcement.

    Args:
        distances: (n_out, n_in) pairwise distance matrix (REQUIRED).
        min_fan_in: Minimum connections per output neuron (row).
        min_fan_out: Minimum connections per input neuron (column).
        check_components: If True, bridge disconnected components after repair.
        (Other args same as developmental_pruning.)

    Returns:
        dict with 'weights', 'mask', 'total_repaired', 'total_bridges',
        and optionally 'history'.
    """
    use_spatial = locality_bias is not None
    history = [] if track_history else None
    total_repaired = 0
    total_bridges = 0

    for cycle in range(n_cycles):
        cycle_seed = seed + cycle * 100

        # Step 1: Weight decay
        weights = weight_decay(weights, mask, decay_rate)

        # Step 2: Reinforcement
        if use_spatial:
            weights = spatial_reinforce(
                weights, mask, distances, locality_bias,
                reinforcement_total, pareto_fraction, cycle_seed)
        else:
            weights = pareto_reinforce(
                weights, mask, reinforcement_total, pareto_fraction, cycle_seed)

        # Step 3: Probabilistic pruning
        mask = prune_probabilistic(
            weights, mask, prune_fraction, steepness, cycle_seed + 1)

        # Step 4: Connectivity repair
        mask, n_repaired = repair_connectivity(
            mask, distances, min_fan_in, min_fan_out)
        total_repaired += n_repaired

        # Step 5: Component bridging (optional)
        n_comp = 1
        n_bridges_cycle = 0
        if check_components:
            mask, n_comp, n_bridges_cycle = bridge_disconnected_components(
                mask, distances)
            total_bridges += n_bridges_cycle

        # Zero out pruned weights (repaired edges get weight 0 \u2014 lifeline mechanism)
        weights = weights * mask

        if track_history:
            n_active = np.count_nonzero(mask)
            total = mask.size
            active_weights = np.abs(weights[mask > 0])
            history.append({
                "cycle": cycle + 1,
                "n_edges": n_active,
                "sparsity": 1.0 - n_active / total,
                "mean_weight": float(active_weights.mean()) if len(active_weights) > 0 else 0,
                "weight_std": float(active_weights.std()) if len(active_weights) > 0 else 0,
                "n_repaired": n_repaired,
                "n_components": n_comp,
                "n_bridges": n_bridges_cycle,
            })

    result = {
        "weights": weights,
        "mask": mask,
        "total_repaired": total_repaired,
        "total_bridges": total_bridges,
    }
    if track_history:
        result["history"] = history
    return result
