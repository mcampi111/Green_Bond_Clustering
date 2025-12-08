# %% [markdown]
# # Spectral Clustering for Green Bond Coefficient Curves
#
# Based on: Cristianini, Shawe-Taylor, Kandola (2001) "Spectral Kernel Methods for Clustering"
#
# Two methods:
# 1. Alignment optimization - first eigenvector of kernel matrix K
# 2. Cut-cost optimization - Fiedler vector of Laplacian L = D - K

# %%
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadr
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
print("Imports done!")

# %%
RDS_PATH = "results.rds"
FEATURES_PATH = "Features.xlsx"

# %%
def load_rds_file(filepath):
    result = pyreadr.read_r(filepath)
    if len(result) == 1:
        return list(result.values())[0]
    return result

print(f"Loading: {RDS_PATH}")
results = load_rds_file(RDS_PATH)
print(f"Results type: {type(results)}")
if isinstance(results, dict):
    print(f"Keys: {list(results.keys())}")

# %%
# Extract coefficients - try different structures
if isinstance(results, dict) and 'coefficients_curve' in results:
    coef_curves = results['coefficients_curve']
elif isinstance(results, dict) and 'coefficients' in results:
    coef_curves = results['coefficients']
else:
    coef_curves = results

print(f"coef_curves type: {type(coef_curves)}")
print(f"Number of bonds: {len(coef_curves)}")

# Look at first few elements to understand structure
if isinstance(coef_curves, dict):
    keys = list(coef_curves.keys())[:3]
elif isinstance(coef_curves, list):
    keys = [0, 1, 2]
else:
    keys = []

for k in keys:
    elem = coef_curves[k]
    print(f"\nBond {k}: type={type(elem).__name__}", end="")
    if hasattr(elem, 'shape'):
        print(f", shape={elem.shape}")
    elif hasattr(elem, '__len__'):
        print(f", len={len(elem)}")
    else:
        print()

# %%
def prepare_coefficient_matrix(coef_curves):
    """
    Convert coefficient curves to a matrix with zero-padding.
    
    Bonds have different lengths (different issuance dates).
    Pad shorter bonds with zeros at the end to match the longest.
    """
    coef_list = []
    bond_ids = []
    
    # Handle dict or list
    if isinstance(coef_curves, dict):
        items = coef_curves.items()
    else:
        items = enumerate(coef_curves)
    
    # First pass: extract all curves and record lengths
    for key, coef_data in items:
        try:
            # Extract curve based on data type
            if isinstance(coef_data, np.ndarray):
                curve = coef_data.flatten()
            elif isinstance(coef_data, pd.DataFrame):
                # If DataFrame, flatten all values
                curve = coef_data.values.flatten()
            elif isinstance(coef_data, pd.Series):
                curve = coef_data.values
            elif isinstance(coef_data, dict):
                # Concatenate all values from the dict
                curve = np.concatenate([np.array(v).flatten() for v in coef_data.values()])
            elif isinstance(coef_data, (list, tuple)):
                curve = np.array(coef_data).flatten()
            else:
                print(f"Warning: Unexpected type {type(coef_data)} for key {key}, skipping")
                continue
            
            # Skip if empty or all NaN
            if len(curve) == 0:
                print(f"Warning: Empty curve for key {key}, skipping")
                continue
            if np.all(np.isnan(curve)):
                print(f"Warning: All NaN curve for key {key}, skipping")
                continue
                
            coef_list.append(curve)
            bond_ids.append(key)
                
        except Exception as e:
            print(f"Warning: Could not process key {key}: {e}")
            continue
    
    if len(coef_list) == 0:
        raise ValueError("No valid coefficient curves found!")
    
    # Check lengths
    lengths = [len(c) for c in coef_list]
    min_len = min(lengths)
    max_len = max(lengths)
    
    print(f"\nCurve lengths: min={min_len}, max={max_len}, mean={np.mean(lengths):.1f}")
    print(f"Number of unique lengths: {len(set(lengths))}")
    
    # Zero-pad all curves to max length
    if min_len != max_len:
        print(f"Padding {sum(1 for l in lengths if l < max_len)} curves with zeros to length {max_len}")
        padded_list = []
        for curve in coef_list:
            if len(curve) < max_len:
                # Pad with zeros at the end
                padded = np.pad(curve, (0, max_len - len(curve)), mode='constant', constant_values=0)
            else:
                padded = curve
            padded_list.append(padded)
        coef_list = padded_list
    
    coef_matrix = np.array(coef_list)
    
    # Replace any remaining NaN with 0
    if np.any(np.isnan(coef_matrix)):
        nan_count = np.sum(np.isnan(coef_matrix))
        print(f"Replacing {nan_count} NaN values with 0")
        coef_matrix = np.nan_to_num(coef_matrix, nan=0.0)
    
    print(f"\nFinal coefficient matrix shape: {coef_matrix.shape}")
    print(f"  {coef_matrix.shape[0]} bonds x {coef_matrix.shape[1]} time points")
    
    return coef_matrix, bond_ids

# %%
coef_matrix, bond_ids = prepare_coefficient_matrix(coef_curves)
print(f"\nFirst 5 bond IDs: {bond_ids[:5]}")

# %%
# Visualize some coefficient curves
fig, ax = plt.subplots(figsize=(14, 6))

n_show = 30
np.random.seed(42)
sample_idx = np.random.choice(len(coef_matrix), min(n_show, len(coef_matrix)), replace=False)

for i in sample_idx:
    ax.plot(coef_matrix[i], alpha=0.4)

ax.set_xlabel("Time/Coefficient Index")
ax.set_ylabel("Coefficient Value")
ax.set_title(f"Sample of {n_show} Coefficient Curves")
plt.tight_layout()
plt.show()

# %%
def compute_kernel_matrix(X, kernel_type='linear', gamma=None, normalize=True, center=True):
    """
    Compute the kernel (similarity) matrix.
    """
    X = X.copy()
    
    # Normalize to unit norm
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X = X / norms
    
    # Compute kernel
    if kernel_type == 'linear':
        K = X @ X.T
    elif kernel_type == 'rbf':
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = np.exp(-gamma * sq_dists)
    
    # Center kernel matrix (Cristianini et al.)
    if center:
        m = K.shape[0]
        j = np.ones((m, 1))
        g = K.sum(axis=1, keepdims=True)
        J = np.ones((m, m))
        K = K - (1/m) * j @ g.T - (1/m) * g @ j.T + (1/m**2) * (j.T @ K @ j) * J
    
    return K

# %%
K_linear = compute_kernel_matrix(coef_matrix, kernel_type='linear')
K_rbf = compute_kernel_matrix(coef_matrix, kernel_type='rbf')

print(f"Linear kernel: shape={K_linear.shape}, range=[{K_linear.min():.4f}, {K_linear.max():.4f}]")
print(f"RBF kernel: shape={K_rbf.shape}, range=[{K_rbf.min():.4f}, {K_rbf.max():.4f}]")

# %%
# Visualize kernel matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im1 = axes[0].imshow(K_linear, cmap='RdBu_r', aspect='auto')
axes[0].set_title('Linear Kernel (Centered)')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(K_rbf, cmap='RdBu_r', aspect='auto')
axes[1].set_title('RBF Kernel (Centered)')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# %%
def alignment_clustering(K, n_clusters=2):
    """
    Cluster using alignment optimization (Cristianini Section 3.1).
    Uses first eigenvector of kernel matrix K.
    """
    eigenvalues, eigenvectors = linalg.eigh(K)
    
    # Sort by decreasing eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    K_norm = np.linalg.norm(K, 'fro')
    
    if n_clusters == 2:
        v_max = eigenvectors[:, 0]
        
        sorted_vals = np.sort(np.unique(v_max))
        thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
        
        best_alignment = -np.inf
        best_threshold = 0
        all_alignments = []
        
        for thresh in thresholds:
            y = np.where(v_max >= thresh, 1, -1)
            yy_T = np.outer(y, y)
            alignment = np.sum(K * yy_T) / (len(y) * K_norm)
            all_alignments.append((thresh, alignment))
            
            if alignment > best_alignment:
                best_alignment = alignment
                best_threshold = thresh
                best_labels = (y + 1) // 2
        
        upper_bound = eigenvalues[0] / K_norm
        
        info = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'first_eigenvector': v_max,
            'optimal_threshold': best_threshold,
            'alignment': best_alignment,
            'upper_bound': upper_bound,
            'all_alignments': all_alignments
        }
        
        return best_labels, info
    
    else:
        V = eigenvectors[:, :n_clusters]
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1
        V_normalized = V / norms
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(V_normalized)
        
        info = {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}
        return labels, info

# %%
labels_align, info_align = alignment_clustering(K_linear, n_clusters=2)

print("Alignment-based Clustering Results (Linear Kernel)")
print("=" * 50)
print(f"Optimal threshold: {info_align['optimal_threshold']:.4f}")
print(f"Alignment achieved: {info_align['alignment']:.4f}")
print(f"Upper bound (λ_max/||K||): {info_align['upper_bound']:.4f}")
print(f"Gap to upper bound: {info_align['upper_bound'] - info_align['alignment']:.4f}")
print(f"\nCluster distribution: {np.bincount(labels_align)}")

# %%
# Plot eigenvalue spectrum and alignment vs threshold
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n_show = 30
axes[0].bar(range(n_show), info_align['eigenvalues'][:n_show])
axes[0].set_xlabel('Eigenvalue Index')
axes[0].set_ylabel('Eigenvalue')
axes[0].set_title('Top Eigenvalues of Kernel Matrix')

thresholds, alignments = zip(*info_align['all_alignments'])
axes[1].plot(range(len(alignments)), alignments, 'b-', label='Alignment')
axes[1].axhline(info_align['upper_bound'], color='r', linestyle='--', label='Upper bound')
axes[1].axhline(info_align['alignment'], color='g', linestyle=':', label='Best alignment')
axes[1].set_xlabel('Threshold Index')
axes[1].set_ylabel('Alignment')
axes[1].set_title('Alignment vs Threshold')
axes[1].legend()

plt.tight_layout()
plt.show()

# %%
def cutcost_clustering(K, n_clusters=2, normalized=True):
    """
    Cluster using cut-cost optimization (Cristianini Section 3.2).
    Uses Fiedler vector of Laplacian L = D - K.
    """
    K_pos = K - K.min() if K.min() < 0 else K
    
    d = K_pos.sum(axis=1)
    D = np.diag(d)
    L = D - K_pos
    
    if normalized:
        d_inv_sqrt = np.zeros_like(d)
        d_inv_sqrt[d > 0] = 1.0 / np.sqrt(d[d > 0])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = D_inv_sqrt @ L @ D_inv_sqrt
    
    eigenvalues, eigenvectors = linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    K_norm = np.linalg.norm(K, 'fro')
    
    if n_clusters == 2:
        fiedler = eigenvectors[:, 1]
        
        sorted_vals = np.sort(np.unique(fiedler))
        thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
        
        best_cutcost = np.inf
        all_cutcosts = []
        
        for thresh in thresholds:
            y = np.where(fiedler >= thresh, 1, -1)
            cutcost = 0.5 * y @ L @ y
            all_cutcosts.append((thresh, cutcost))
            
            if cutcost < best_cutcost:
                best_cutcost = cutcost
                best_threshold = thresh
                best_labels = (y + 1) // 2
        
        lower_bound = eigenvalues[1] / (2 * K_norm)
        
        info = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'fiedler_vector': fiedler,
            'optimal_threshold': best_threshold,
            'cutcost': best_cutcost,
            'lower_bound': lower_bound,
            'all_cutcosts': all_cutcosts,
            'laplacian': L
        }
        
        return best_labels, info
    
    else:
        V = eigenvectors[:, 1:n_clusters+1]
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1
        V_normalized = V / norms
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(V_normalized)
        
        info = {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors, 'laplacian': L}
        return labels, info

# %%
labels_cut, info_cut = cutcost_clustering(K_linear, n_clusters=2)

print("Cut-cost Clustering Results (Linear Kernel)")
print("=" * 50)
print(f"Optimal threshold: {info_cut['optimal_threshold']:.4f}")
print(f"Cut-cost achieved: {info_cut['cutcost']:.4f}")
print(f"Lower bound (λ_2/2||K||): {info_cut['lower_bound']:.4f}")
print(f"\nCluster distribution: {np.bincount(labels_cut)}")

# %%
# Compare methods
print("\nComparison of Methods")
print("=" * 50)
print(f"Agreement: {(labels_align == labels_cut).mean()*100:.1f}%")

sil_align = silhouette_score(coef_matrix, labels_align)
sil_cut = silhouette_score(coef_matrix, labels_cut)
print(f"Silhouette (alignment): {sil_align:.4f}")
print(f"Silhouette (cut-cost): {sil_cut:.4f}")

# %%
# Plot sorted kernel matrix
def plot_sorted_kernel(K, labels, title="Kernel Matrix"):
    idx = np.argsort(labels)
    K_sorted = K[idx][:, idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(K, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Original Order')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(K_sorted, cmap='RdBu_r', aspect='auto')
    axes[1].set_title('Sorted by Cluster')
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

plot_sorted_kernel(K_linear, labels_align, "Kernel Matrix (Alignment Clustering)")
plt.show()

# %%
# Plot cluster means
def plot_cluster_means(coef_matrix, labels):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    n_clusters = len(np.unique(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for c in range(n_clusters):
        mask = labels == c
        mean_curve = coef_matrix[mask].mean(axis=0)
        std_curve = coef_matrix[mask].std(axis=0)
        
        x = np.arange(len(mean_curve))
        ax.plot(x, mean_curve, color=colors[c], 
                label=f'Cluster {c} (n={mask.sum()})', linewidth=2)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, 
                       color=colors[c], alpha=0.2)
    
    ax.set_xlabel('Time/Coefficient Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Mean Coefficient Curves by Cluster (±1 std)')
    ax.legend()
    return fig

plot_cluster_means(coef_matrix, labels_align)
plt.tight_layout()
plt.show()

# %%
# Try different numbers of clusters
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n_show = 20
axes[0].bar(range(n_show), info_align['eigenvalues'][:n_show])
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Eigenvalue')
axes[0].set_title('Kernel Eigenvalues')

eig_L = info_cut['eigenvalues'][:n_show]
axes[1].bar(range(len(eig_L)), eig_L)
axes[1].set_xlabel('Index')
axes[1].set_ylabel('Eigenvalue')
axes[1].set_title('Laplacian Eigenvalues (gaps suggest clusters)')

plt.tight_layout()
plt.show()

print("\nLaplacian eigengaps:")
for i in range(min(10, len(eig_L)-1)):
    gap = eig_L[i+1] - eig_L[i]
    print(f"  Gap {i} -> {i+1}: {gap:.4f}")

# %%
cluster_range = range(2, 8)
silhouettes = []

for k in cluster_range:
    labels_k, _ = alignment_clustering(K_linear, n_clusters=k)
    sil = silhouette_score(coef_matrix, labels_k)
    silhouettes.append(sil)
    print(f"k={k}: Silhouette={sil:.4f}, Distribution={np.bincount(labels_k)}")

plt.figure(figsize=(8, 5))
plt.plot(list(cluster_range), silhouettes, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Load bond features
try:
    features_df = pd.read_excel(FEATURES_PATH)
    print(f"Features loaded: {features_df.shape}")
    print(f"Columns: {list(features_df.columns)}")
    print(features_df.head())
except FileNotFoundError:
    print(f"Features file not found: {FEATURES_PATH}")
    features_df = None

# %%
# Create results dataframe
results_df = pd.DataFrame({
    'bond_id': bond_ids,
    'cluster_alignment': labels_align,
    'cluster_cutcost': labels_cut
})

if features_df is not None:
    id_cols = [c for c in features_df.columns if 'CUSIP' in c.upper() or 'ID' in c.upper()]
    if id_cols:
        id_col = id_cols[0]
        results_df = results_df.merge(features_df, left_on='bond_id', right_on=id_col, how='left')
        print(f"Merged on: {id_col}")

print(f"\nResults: {results_df.shape}")
print(results_df.head())

# %%
# Save results
results_df.to_csv('green_bond_clusters.csv', index=False)
print("Saved to green_bond_clusters.csv")
