"""
Representational Similarity Analysis (RSA) for comparing learned representations.

RSA compares the representational geometry of different networks by computing
pairwise dissimilarity matrices (RDMs) of hidden layer activations and
correlating these across models. This is a standard tool in computational
neuroscience (Kriegeskorte et al. 2008) applied here to compare how different
network topologies shape learned representations.

Key questions RSA answers for this project:
1. Do bio-inspired networks learn similar representations to dense networks?
2. Does signal degradation change representational geometry?
3. Do networks with different topologies converge to similar representations?
"""

import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kendalltau


@torch.no_grad()
def extract_activations(
    model, data_loader, device: str = "cpu", n_samples: int = 1000, layer_idx: int = -1
) -> np.ndarray:
    """Extract hidden layer activations for a set of inputs.

    Args:
        model: SparseMLP or similar model with get_layer_activations method.
        data_loader: DataLoader with input data.
        device: Device.
        n_samples: Number of samples to use.
        layer_idx: Which hidden layer's activations to extract (-1 = last hidden).

    Returns:
        (n_samples, hidden_dim) array of activations.
    """
    model.eval()
    model = model.to(device)
    all_activations = []
    n_collected = 0

    for batch_x, _ in data_loader:
        batch_x = batch_x.to(device)
        if batch_x.dim() > 2:
            batch_x = batch_x.view(batch_x.size(0), -1)

        if hasattr(model, "get_layer_activations"):
            acts = model.get_layer_activations(batch_x)
            act = acts[layer_idx].cpu().numpy()
        else:
            # Fallback: use model output
            act = model(batch_x).cpu().numpy()

        all_activations.append(act)
        n_collected += act.shape[0]
        if n_collected >= n_samples:
            break

    activations = np.concatenate(all_activations, axis=0)[:n_samples]
    return activations


def compute_rdm(activations: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Compute Representational Dissimilarity Matrix.

    Args:
        activations: (n_samples, n_features) array.
        metric: Distance metric ('correlation', 'euclidean', 'cosine').

    Returns:
        (n_samples, n_samples) symmetric dissimilarity matrix.
    """
    if activations.shape[0] < 2:
        return np.zeros((activations.shape[0], activations.shape[0]))

    # Handle constant features
    std = activations.std(axis=0)
    active_features = std > 1e-8
    if active_features.sum() < 2:
        return np.zeros((activations.shape[0], activations.shape[0]))

    act_filtered = activations[:, active_features]
    distances = pdist(act_filtered, metric=metric)
    return squareform(distances)


def compute_rsa(
    rdm1: np.ndarray, rdm2: np.ndarray, method: str = "spearman"
) -> dict:
    """Compute RSA correlation between two RDMs.

    Args:
        rdm1, rdm2: Representational Dissimilarity Matrices (same size).
        method: 'spearman' or 'kendall'.

    Returns:
        Dict with correlation coefficient and p-value.
    """
    # Extract upper triangle (excluding diagonal)
    idx = np.triu_indices_from(rdm1, k=1)
    v1 = rdm1[idx]
    v2 = rdm2[idx]

    if len(v1) < 3:
        return {"correlation": 0.0, "p_value": 1.0}

    if method == "spearman":
        r, p = spearmanr(v1, v2)
    elif method == "kendall":
        r, p = kendalltau(v1, v2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {"correlation": float(r), "p_value": float(p)}


def compare_representations(
    models: dict,
    data_loader,
    device: str = "cpu",
    n_samples: int = 500,
    layer_idx: int = -1,
) -> dict:
    """Compare representations across multiple models using RSA.

    Args:
        models: Dict mapping model_name -> model.
        data_loader: Shared data loader.
        device: Device.
        n_samples: Number of samples.
        layer_idx: Hidden layer to compare.

    Returns:
        Dict with 'rdms' (per model) and 'rsa_matrix' (pairwise RSA scores).
    """
    # Extract RDMs
    rdms = {}
    for name, model in models.items():
        acts = extract_activations(model, data_loader, device, n_samples, layer_idx)
        rdms[name] = compute_rdm(acts)

    # Pairwise RSA
    names = list(models.keys())
    n = len(names)
    rsa_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if rdms[names[i]].shape == rdms[names[j]].shape:
                result = compute_rsa(rdms[names[i]], rdms[names[j]])
                rsa_matrix[i, j] = result["correlation"]
                rsa_matrix[j, i] = result["correlation"]
            else:
                rsa_matrix[i, j] = rsa_matrix[j, i] = 0.0

    return {
        "rdms": rdms,
        "rsa_matrix": rsa_matrix,
        "model_names": names,
    }
