# %% [markdown]
# # Spectral Clustering for Green Bond Coefficient Curves
#
# Based on: Cristianini, Shawe-Taylor, Kandola (2001) "Spectral Kernel Methods for Clustering"
#
# Data structure from R:
#   results$models$coefficients_curve[[bond_id]][[regressor]]$coefficient
#
# Each bond has 12 regressors, each with a time-varying coefficient curve (the diagonal
# of the bivariate coefficient function γ(s,t) from function-on-function regression).
#
# Clustering is performed per regressor to maintain interpretability.

# %%
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
print("Imports successful!")

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - UPDATE THESE to match your local setup
# RDS_PATH = "results.rds"
FEATURES_PATH = "Features.xlsx"
CSV_PATH = "coefficient_curves.csv"


# List of regressors (from the R output)
REGRESSORS = [
    'XAUUSD',                    # Gold price
    'WTI',                       # Oil price
    'DXY',                       # Dollar index
    'SPX_Index',                 # S&P 500
    'US_fed_funds_effe_rate',    # Fed funds rate
    'CPI_CHNG_Index',            # CPI change
    'M_0',                       # Monetary base
    'BOJ_rate',                  # Bank of Japan rate
    'Carbon_Price_Index',        # Carbon price
    'CO',                        # Carbon monoxide
    'AQI',                       # Air quality index
    'Temperature'                # Temperature
]

# Default regressor for initial analysis
DEFAULT_REGRESSOR = 'Carbon_Price_Index'

# %%
# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_coefficient_curves_from_csv(csv_path):
    """
    Load coefficient curves from a CSV file exported from R.
    
    Expected CSV structure:
        bond_id, regressor, date, coefficient, se
    
    Returns:
        dict: {bond_id: {regressor: {'date': array, 'coefficient': array, 'se': array}}}
    """
    df = pd.read_csv(csv_path)
    
    # Convert to nested dict structure
    data = {}
    for bond_id in df['bond_id'].unique():
        bond_data = df[df['bond_id'] == bond_id]
        data[bond_id] = {}
        
        for regressor in bond_data['regressor'].unique():
            reg_data = bond_data[bond_data['regressor'] == regressor].sort_values('date')
            data[bond_id][regressor] = {
                'date': reg_data['date'].values,
                'coefficient': reg_data['coefficient'].values,
                'se': reg_data['se'].values if 'se' in reg_data.columns else None
            }
    
    return data


def try_load_rds(rds_path):
    """
    Attempt to load RDS file using pyreadr.
    Returns None if pyreadr not available or file can't be loaded.
    """
    try:
        import pyreadr
        result = pyreadr.read_r(rds_path)
        print(f"pyreadr loaded file with {len(result)} top-level objects")
        
        # Debug: print structure
        for key, val in result.items():
            print(f"  Key: {key}, Type: {type(val)}, ", end="")
            if hasattr(val, 'shape'):
                print(f"Shape: {val.shape}")
            elif hasattr(val, '__len__'):
                print(f"Len: {len(val)}")
            else:
                print()
        
        return result
    except ImportError:
        print("pyreadr not installed. Please export data from R to CSV.")
        return None
    except Exception as e:
        print(f"Could not load RDS file: {e}")
        print("Please export coefficient curves from R to CSV format.")
        return None


# %%
# =============================================================================
# R EXPORT HELPER
# =============================================================================

R_EXPORT_CODE = '''
# Run this in R to export coefficient curves to CSV
library(tidyverse)

# Load your results
results <- readRDS("path/to/results.rds")

# Extract coefficient curves
coef_curves <- results$models$coefficients_curve

# Convert to long-format dataframe
all_curves <- map2_dfr(coef_curves, names(coef_curves), function(bond_data, bond_id) {
  map2_dfr(bond_data, names(bond_data), function(regressor_data, regressor_name) {
    regressor_data %>%
      mutate(bond_id = bond_id, regressor = regressor_name)
  })
})

# Save to CSV
write_csv(all_curves, "coefficient_curves.csv")

# Check dimensions
print(paste("Exported", n_distinct(all_curves$bond_id), "bonds"))
print(paste("Regressors:", paste(unique(all_curves$regressor), collapse=", ")))
'''

print("If pyreadr fails, run this R code to export data:")
print("-" * 60)
print(R_EXPORT_CODE)
print("-" * 60)

# %%
# =============================================================================
# COEFFICIENT MATRIX PREPARATION
# =============================================================================

def extract_regressor_matrix(coef_data, regressor, pad_value=0.0):
    """
    Extract coefficient curves for a single regressor across all bonds.
    
    Parameters:
        coef_data: dict of {bond_id: {regressor: {'coefficient': array, ...}}}
        regressor: str, name of regressor to extract
        pad_value: float, value to use for padding shorter curves
    
    Returns:
        coef_matrix: np.array of shape (n_bonds, max_length)
        bond_ids: list of bond identifiers
        curve_lengths: list of original curve lengths
    """
    curves = []
    bond_ids = []
    curve_lengths = []
    
    for bond_id, bond_regressors in coef_data.items():
        if regressor not in bond_regressors:
            print(f"Warning: {regressor} not found for bond {bond_id}, skipping")
            continue
        
        reg_data = bond_regressors[regressor]
        
        # Extract coefficient values
        if isinstance(reg_data, dict) and 'coefficient' in reg_data:
            coef = np.array(reg_data['coefficient'])
        elif isinstance(reg_data, pd.DataFrame) and 'coefficient' in reg_data.columns:
            coef = reg_data['coefficient'].values
        elif isinstance(reg_data, np.ndarray):
            coef = reg_data.flatten()
        else:
            print(f"Warning: Unexpected data type for {bond_id}/{regressor}: {type(reg_data)}")
            continue
        
        # Skip invalid curves
        if len(coef) == 0:
            print(f"Warning: Empty curve for {bond_id}/{regressor}, skipping")
            continue
        if np.all(np.isnan(coef)):
            print(f"Warning: All NaN for {bond_id}/{regressor}, skipping")
            continue
        
        curves.append(coef)
        bond_ids.append(bond_id)
        curve_lengths.append(len(coef))
    
    if len(curves) == 0:
        raise ValueError(f"No valid curves found for regressor: {regressor}")
    
    # Report length statistics
    min_len, max_len = min(curve_lengths), max(curve_lengths)
    print(f"\nRegressor: {regressor}")
    print(f"  Bonds: {len(curves)}")
    print(f"  Curve lengths: min={min_len}, max={max_len}, mean={np.mean(curve_lengths):.1f}")
    print(f"  Unique lengths: {len(set(curve_lengths))}")
    
    # Zero-pad to max length
    padded_curves = []
    for curve in curves:
        if len(curve) < max_len:
            padded = np.pad(curve, (0, max_len - len(curve)), 
                          mode='constant', constant_values=pad_value)
        else:
            padded = curve
        padded_curves.append(padded)
    
    coef_matrix = np.array(padded_curves)
    
    # Handle NaN values
    if np.any(np.isnan(coef_matrix)):
        nan_count = np.sum(np.isnan(coef_matrix))
        print(f"  Replacing {nan_count} NaN values with {pad_value}")
        coef_matrix = np.nan_to_num(coef_matrix, nan=pad_value)
    
    print(f"  Final matrix shape: {coef_matrix.shape}")
    
    return coef_matrix, bond_ids, curve_lengths


# %%
# =============================================================================
# KERNEL COMPUTATION (Cristianini et al. 2001)
# =============================================================================

def compute_kernel_matrix(X, kernel_type='linear', gamma=None, normalize=True, center=True):
    """
    Compute the kernel (similarity) matrix following Cristianini et al. (2001).
    
    Parameters:
        X: np.array of shape (n_samples, n_features)
        kernel_type: 'linear' or 'rbf'
        gamma: float, RBF kernel bandwidth (default: 1/n_features)
        normalize: bool, normalize rows to unit norm before computing kernel
        center: bool, center kernel matrix in feature space
    
    Returns:
        K: np.array of shape (n_samples, n_samples), the kernel matrix
    """
    X = X.copy()
    
    # Normalize to unit norm (important for alignment interpretation)
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        X = X / norms
    
    # Compute kernel
    if kernel_type == 'linear':
        K = X @ X.T
    elif kernel_type == 'rbf':
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = np.exp(-gamma * sq_dists)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Center kernel matrix (Section 2 of Cristianini et al.)
    # This corresponds to centering data in feature space
    if center:
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        K = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    return K


# %%
# =============================================================================
# SPECTRAL CLUSTERING: ALIGNMENT OPTIMIZATION (Section 3.1)
# =============================================================================

def alignment_clustering(K, n_clusters=2):
    """
    Cluster using alignment optimization from Cristianini et al. (2001) Section 3.1.
    
    For binary clustering, uses the first eigenvector of kernel matrix K.
    The alignment A(K, yy') measures how well the kernel agrees with a clustering.
    
    Parameters:
        K: np.array, centered kernel matrix
        n_clusters: int, number of clusters
    
    Returns:
        labels: np.array of cluster assignments
        info: dict with eigenvalues, eigenvectors, alignment scores, etc.
    """
    # Compute eigendecomposition
    eigenvalues, eigenvectors = linalg.eigh(K)
    
    # Sort by decreasing eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Frobenius norm for normalization
    K_norm = np.linalg.norm(K, 'fro')
    
    if n_clusters == 2:
        # Binary clustering: threshold first eigenvector
        v1 = eigenvectors[:, 0]
        
        # Try all possible thresholds
        sorted_vals = np.sort(np.unique(v1))
        thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
        
        best_alignment = -np.inf
        all_alignments = []
        
        for thresh in thresholds:
            # Create label vector y in {-1, +1}
            y = np.where(v1 >= thresh, 1, -1)
            
            # Alignment = <K, yy'> / ||K||_F (equation 4)
            yy_outer = np.outer(y, y)
            alignment = np.sum(K * yy_outer) / K_norm
            all_alignments.append((thresh, alignment))
            
            if alignment > best_alignment:
                best_alignment = alignment
                best_threshold = thresh
                best_labels = (y + 1) // 2  # Convert to {0, 1}
        
        # Upper bound on alignment is lambda_max / ||K||_F
        upper_bound = eigenvalues[0] / K_norm
        
        info = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'first_eigenvector': v1,
            'optimal_threshold': best_threshold,
            'alignment': best_alignment,
            'upper_bound': upper_bound,
            'all_alignments': all_alignments
        }
        
        return best_labels, info
    
    else:
        # Multi-class: use top k eigenvectors + k-means
        V = eigenvectors[:, :n_clusters]
        
        # Normalize rows
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1
        V_normalized = V / norms
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(V_normalized)
        
        info = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'embedding': V_normalized
        }
        
        return labels, info


# %%
# =============================================================================
# SPECTRAL CLUSTERING: CUT-COST OPTIMIZATION (Section 3.2)
# =============================================================================

def cutcost_clustering(K, n_clusters=2, normalized=True):
    """
    Cluster using cut-cost optimization from Cristianini et al. (2001) Section 3.2.
    
    Uses the Fiedler vector (second smallest eigenvector) of the graph Laplacian L = D - K.
    
    Parameters:
        K: np.array, kernel matrix
        n_clusters: int, number of clusters
        normalized: bool, use normalized Laplacian L_sym = D^{-1/2} L D^{-1/2}
    
    Returns:
        labels: np.array of cluster assignments
        info: dict with eigenvalues, Fiedler vector, cut costs, etc.
    """
    # Ensure non-negative kernel for graph interpretation
    K_pos = K - K.min() if K.min() < 0 else K.copy()
    
    # Degree matrix
    d = K_pos.sum(axis=1)
    D = np.diag(d)
    
    # Laplacian L = D - K
    L = D - K_pos
    
    if normalized:
        # Normalized Laplacian: L_sym = D^{-1/2} L D^{-1/2}
        d_inv_sqrt = np.zeros_like(d)
        d_inv_sqrt[d > 0] = 1.0 / np.sqrt(d[d > 0])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = D_inv_sqrt @ L @ D_inv_sqrt
    
    # Eigendecomposition
    eigenvalues, eigenvectors = linalg.eigh(L)
    
    # Sort by increasing eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    K_norm = np.linalg.norm(K, 'fro')
    
    if n_clusters == 2:
        # Binary clustering: threshold Fiedler vector (2nd eigenvector)
        fiedler = eigenvectors[:, 1]
        
        sorted_vals = np.sort(np.unique(fiedler))
        thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
        
        best_cutcost = np.inf
        all_cutcosts = []
        
        for thresh in thresholds:
            y = np.where(fiedler >= thresh, 1, -1)
            
            # Cut cost = (1/2) y' L y
            cutcost = 0.5 * y @ L @ y
            all_cutcosts.append((thresh, cutcost))
            
            if cutcost < best_cutcost:
                best_cutcost = cutcost
                best_threshold = thresh
                best_labels = (y + 1) // 2
        
        # Lower bound on cut cost
        lower_bound = eigenvalues[1] / (2 * K_norm) if K_norm > 0 else 0
        
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
        # Multi-class: use first k eigenvectors (excluding constant)
        V = eigenvectors[:, 1:n_clusters]
        
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1
        V_normalized = V / norms
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(V_normalized)
        
        info = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'embedding': V_normalized,
            'laplacian': L
        }
        
        return labels, info


# %%
# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_coefficient_curves(coef_matrix, bond_ids=None, n_show=30, title=None, ax=None):
    """Plot a sample of coefficient curves."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    np.random.seed(42)
    n_show = min(n_show, len(coef_matrix))
    sample_idx = np.random.choice(len(coef_matrix), n_show, replace=False)
    
    for i in sample_idx:
        ax.plot(coef_matrix[i], alpha=0.4, linewidth=0.8)
    
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Coefficient Value (Diagonal)")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Sample of {n_show} Coefficient Curves")
    
    return ax


def plot_kernel_matrix(K, title="Kernel Matrix", ax=None):
    """Plot the kernel matrix as a heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(K, cmap='RdBu_r', aspect='auto')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    
    return ax


def plot_sorted_kernel(K, labels, title="Kernel Matrix"):
    """Plot kernel matrix sorted by cluster assignment."""
    idx = np.argsort(labels)
    K_sorted = K[idx][:, idx]
    labels_sorted = labels[idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(K, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Original Order')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(K_sorted, cmap='RdBu_r', aspect='auto')
    axes[1].set_title('Sorted by Cluster')
    plt.colorbar(im2, ax=axes[1])
    
    # Add cluster boundaries
    cluster_sizes = np.bincount(labels)
    boundaries = np.cumsum(cluster_sizes)[:-1]
    for b in boundaries:
        axes[1].axhline(b - 0.5, color='black', linewidth=2)
        axes[1].axvline(b - 0.5, color='black', linewidth=2)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_cluster_means(coef_matrix, labels, title=None, ax=None):
    """Plot mean coefficient curves for each cluster with confidence bands."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    n_clusters = len(np.unique(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for c in range(n_clusters):
        mask = labels == c
        cluster_curves = coef_matrix[mask]
        
        mean_curve = cluster_curves.mean(axis=0)
        std_curve = cluster_curves.std(axis=0)
        
        x = np.arange(len(mean_curve))
        ax.plot(x, mean_curve, color=colors[c], 
                label=f'Cluster {c} (n={mask.sum()})', linewidth=2)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, 
                       color=colors[c], alpha=0.2)
    
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Coefficient Value')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Mean Coefficient Curves by Cluster (±1 std)')
    ax.legend()
    
    return ax


def plot_eigenspectrum(info_align, info_cut, n_show=20):
    """Plot eigenvalue spectra for both methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Kernel eigenvalues (for alignment method)
    eig_K = info_align['eigenvalues'][:n_show]
    axes[0].bar(range(len(eig_K)), eig_K, color='steelblue')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title('Kernel Matrix Eigenvalues\n(larger = more important for alignment)')
    
    # Laplacian eigenvalues (for cut-cost method)
    eig_L = info_cut['eigenvalues'][:n_show]
    axes[1].bar(range(len(eig_L)), eig_L, color='darkorange')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].set_title('Laplacian Eigenvalues\n(gaps suggest number of clusters)')
    
    plt.tight_layout()
    return fig


# %%
# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def run_spectral_clustering_for_regressor(coef_data, regressor, n_clusters=2, 
                                          kernel_type='linear', show_plots=True):
    """
    Run full spectral clustering pipeline for a single regressor.
    
    Parameters:
        coef_data: dict, coefficient curves data
        regressor: str, name of regressor to analyze
        n_clusters: int, number of clusters
        kernel_type: str, 'linear' or 'rbf'
        show_plots: bool, whether to display plots
    
    Returns:
        results: dict with labels, metrics, and diagnostic info
    """
    print(f"\n{'='*60}")
    print(f"SPECTRAL CLUSTERING: {regressor}")
    print(f"{'='*60}")
    
    # Extract coefficient matrix for this regressor
    coef_matrix, bond_ids, curve_lengths = extract_regressor_matrix(coef_data, regressor)
    
    # Compute kernel matrix
    K = compute_kernel_matrix(coef_matrix, kernel_type=kernel_type, normalize=True, center=True)
    print(f"\nKernel matrix: shape={K.shape}, range=[{K.min():.4f}, {K.max():.4f}]")
    
    # Method 1: Alignment clustering
    labels_align, info_align = alignment_clustering(K, n_clusters=n_clusters)
    
    print(f"\n--- Alignment Method (Section 3.1) ---")
    print(f"Optimal threshold: {info_align['optimal_threshold']:.4f}")
    print(f"Alignment achieved: {info_align['alignment']:.4f}")
    print(f"Upper bound: {info_align['upper_bound']:.4f}")
    print(f"Cluster sizes: {np.bincount(labels_align)}")
    
    # Method 2: Cut-cost clustering
    labels_cut, info_cut = cutcost_clustering(K, n_clusters=n_clusters)
    
    print(f"\n--- Cut-cost Method (Section 3.2) ---")
    print(f"Optimal threshold: {info_cut['optimal_threshold']:.4f}")
    print(f"Cut-cost achieved: {info_cut['cutcost']:.4f}")
    print(f"Cluster sizes: {np.bincount(labels_cut)}")
    
    # Compare methods
    agreement = (labels_align == labels_cut).mean()
    sil_align = silhouette_score(coef_matrix, labels_align)
    sil_cut = silhouette_score(coef_matrix, labels_cut)
    
    print(f"\n--- Comparison ---")
    print(f"Method agreement: {agreement*100:.1f}%")
    print(f"Silhouette (alignment): {sil_align:.4f}")
    print(f"Silhouette (cut-cost): {sil_cut:.4f}")
    
    # Visualizations
    if show_plots:
        # 1. Sample coefficient curves
        fig1, ax1 = plt.subplots(figsize=(14, 5))
        plot_coefficient_curves(coef_matrix, title=f"Coefficient Curves: {regressor}", ax=ax1)
        plt.tight_layout()
        plt.show()
        
        # 2. Kernel matrix sorted by clusters
        fig2 = plot_sorted_kernel(K, labels_align, 
                                  title=f"Kernel Matrix - {regressor} (Alignment Clustering)")
        plt.show()
        
        # 3. Cluster means
        fig3, ax3 = plt.subplots(figsize=(14, 5))
        plot_cluster_means(coef_matrix, labels_align, 
                          title=f"Cluster Means: {regressor}", ax=ax3)
        plt.tight_layout()
        plt.show()
        
        # 4. Eigenspectrum
        fig4 = plot_eigenspectrum(info_align, info_cut)
        plt.show()
    
    # Compile results
    results = {
        'regressor': regressor,
        'coef_matrix': coef_matrix,
        'bond_ids': bond_ids,
        'curve_lengths': curve_lengths,
        'kernel_matrix': K,
        'labels_alignment': labels_align,
        'labels_cutcost': labels_cut,
        'info_alignment': info_align,
        'info_cutcost': info_cut,
        'silhouette_alignment': sil_align,
        'silhouette_cutcost': sil_cut,
        'method_agreement': agreement
    }
    
    return results


def analyze_optimal_clusters(coef_matrix, K, max_clusters=8):
    """
    Analyze optimal number of clusters using silhouette scores and eigengaps.
    """
    print("\n--- Optimal Number of Clusters ---")
    
    silhouettes = []
    for k in range(2, max_clusters + 1):
        labels_k, _ = alignment_clustering(K, n_clusters=k)
        sil = silhouette_score(coef_matrix, labels_k)
        silhouettes.append((k, sil))
        print(f"k={k}: Silhouette={sil:.4f}, Sizes={list(np.bincount(labels_k))}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ks, sils = zip(*silhouettes)
    ax.plot(ks, sils, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs Number of Clusters')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    best_k = ks[np.argmax(sils)]
    print(f"\nOptimal k by silhouette: {best_k}")
    
    return silhouettes, best_k


# %%
# =============================================================================
# LOAD DATA AND RUN ANALYSIS
# =============================================================================

# Load from CSV
print(f"Loading from: {CSV_PATH}")
coef_data = load_coefficient_curves_from_csv(CSV_PATH)
print(f"Loaded {len(coef_data)} bonds")

# Get available regressors
first_bond = list(coef_data.keys())[0]
available_regressors = list(coef_data[first_bond].keys())
print(f"Available regressors: {available_regressors}")

## %%
## Run clustering for Carbon_Price_Index
#results = run_spectral_clustering_for_regressor(
#    coef_data, 
#    DEFAULT_REGRESSOR, 
#    n_clusters=2,
#    show_plots=True
#)
#
## %%
## Analyze optimal number of clusters
#silhouettes, best_k = analyze_optimal_clusters(
#    results['coef_matrix'], 
#    results['kernel_matrix']
#)
#
## %%
## Save results
#results_df = pd.DataFrame({
#    'bond_id': results['bond_ids'],
#    'cluster_alignment': results['labels_alignment'],
#    'cluster_cutcost': results['labels_cutcost'],
#    'curve_length': results['curve_lengths']
#})
#
#output_file = f"clusters_{DEFAULT_REGRESSOR}.csv"
#results_df.to_csv(output_file, index=False)
#print(f"Results saved to: {output_file}")
#
# %%
# =============================================================================
# BATCH ANALYSIS: ALL REGRESSORS
# =============================================================================

def run_all_regressors(coef_data, regressors=None, n_clusters=2, show_plots=False):
    """
    Run spectral clustering for all regressors and compile comparison.
    """
    if regressors is None:
        first_bond = list(coef_data.keys())[0]
        regressors = list(coef_data[first_bond].keys())
    
    all_results = {}
    summary_rows = []
    
    for reg in regressors:
        try:
            results = run_spectral_clustering_for_regressor(
                coef_data, reg, n_clusters=n_clusters, show_plots=show_plots
            )
            all_results[reg] = results
            
            summary_rows.append({
                'regressor': reg,
                'n_bonds': len(results['bond_ids']),
                'silhouette_alignment': results['silhouette_alignment'],
                'silhouette_cutcost': results['silhouette_cutcost'],
                'method_agreement': results['method_agreement'],
                'cluster_0_size': np.sum(results['labels_alignment'] == 0),
                'cluster_1_size': np.sum(results['labels_alignment'] == 1)
            })
        except Exception as e:
            print(f"Error processing {reg}: {e}")
            continue
    
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL REGRESSORS")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    return all_results, summary_df

# %%
# =============================================================================
# FIND OPTIMAL K FOR ALL REGRESSORS
# =============================================================================

def find_optimal_k_all_regressors(coef_data, regressors=None, max_k=6):
    """
    Find optimal number of clusters for each regressor.
    """
    if regressors is None:
        first_bond = list(coef_data.keys())[0]
        regressors = list(coef_data[first_bond].keys())
    
    optimal_k_results = []
    
    for reg in regressors:
        print(f"\n{'='*60}")
        print(f"OPTIMAL K ANALYSIS: {reg}")
        print(f"{'='*60}")
        
        coef_matrix, bond_ids, curve_lengths = extract_regressor_matrix(coef_data, reg)
        K = compute_kernel_matrix(coef_matrix, kernel_type='linear', normalize=True, center=True)
        
        best_k = 2
        best_sil = -1
        k_scores = []
        
        for k in range(2, max_k + 1):
            labels_k, _ = alignment_clustering(K, n_clusters=k)
            sil = silhouette_score(coef_matrix, labels_k)
            k_scores.append({'k': k, 'silhouette': sil})
            print(f"  k={k}: silhouette={sil:.4f}, sizes={list(np.bincount(labels_k))}")
            
            if sil > best_sil:
                best_sil = sil
                best_k = k
        
        print(f"  --> Optimal k = {best_k} (silhouette = {best_sil:.4f})")
        
        optimal_k_results.append({
            'regressor': reg,
            'optimal_k': best_k,
            'best_silhouette': best_sil,
            'k_scores': k_scores
        })
    
    # Summary table
    summary = pd.DataFrame([{
        'regressor': r['regressor'],
        'optimal_k': r['optimal_k'],
        'best_silhouette': r['best_silhouette']
    } for r in optimal_k_results])
    
    print("\n" + "="*60)
    print("OPTIMAL K SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    
    return optimal_k_results, summary

# %%
# Run batch analysis with k=2
if coef_data is not None:
    all_results, summary_df = run_all_regressors(coef_data, show_plots=False)
    summary_df.to_csv("clustering_summary_all_regressors.csv", index=False)

# %%
# Find optimal k for all regressors
optimal_k_results, optimal_k_summary = find_optimal_k_all_regressors(coef_data, max_k=6)
optimal_k_summary.to_csv("optimal_k_all_regressors.csv", index=False)

# %%
# =============================================================================
# SAVE PLOTS TO FILES (using optimal k for each regressor)
# =============================================================================

import os

os.makedirs("clustering_plots", exist_ok=True)

# Create lookup for optimal k
optimal_k_lookup = {r['regressor']: r['optimal_k'] for r in optimal_k_results}

# Plot ALL regressors with their optimal k
all_regressors = list(optimal_k_lookup.keys())

for reg in all_regressors:
    k = optimal_k_lookup.get(reg, 2)
    print(f"Processing: {reg} (k={k})")
    
    coef_matrix, bond_ids, curve_lengths = extract_regressor_matrix(coef_data, reg)
    K = compute_kernel_matrix(coef_matrix, kernel_type='linear', normalize=True, center=True)
    labels_align, info_align = alignment_clustering(K, n_clusters=k)
    labels_cut, info_cut = cutcost_clustering(K, n_clusters=k)
    
    # Plot 1: Coefficient curves
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    plot_coefficient_curves(coef_matrix, title=f"Coefficient Curves: {reg}", ax=ax1)
    fig1.savefig(f"clustering_plots/{reg}_1_curves.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot 2: Kernel matrix
    fig2 = plot_sorted_kernel(K, labels_align, title=f"Kernel Matrix - {reg} (k={k})")
    fig2.savefig(f"clustering_plots/{reg}_2_kernel.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # Plot 3: Cluster means
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    plot_cluster_means(coef_matrix, labels_align, title=f"Cluster Means: {reg} (k={k})", ax=ax3)
    fig3.savefig(f"clustering_plots/{reg}_3_means.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # Plot 4: Eigenspectrum
    fig4 = plot_eigenspectrum(info_align, info_cut)
    fig4.savefig(f"clustering_plots/{reg}_4_eigen.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    # Plot 5: Silhouette vs k
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    k_data = next(r['k_scores'] for r in optimal_k_results if r['regressor'] == reg)
    ks = [d['k'] for d in k_data]
    sils = [d['silhouette'] for d in k_data]
    ax5.plot(ks, sils, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax5.axvline(k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={k}')
    ax5.set_xlabel('Number of Clusters (k)')
    ax5.set_ylabel('Silhouette Score')
    ax5.set_title(f'Optimal k Selection: {reg}')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    fig5.savefig(f"clustering_plots/{reg}_5_optimal_k.png", dpi=150, bbox_inches='tight')
    plt.close(fig5)
    
    print(f"  Saved 5 plots")

print(f"\nDone! Check: {os.path.abspath('clustering_plots')}")

# %%
# =============================================================================
# DETAILED SINGLE REGRESSOR ANALYSIS
# =============================================================================

# RUN THIS FOR INDIVIDUAL REGRESSORS

def detailed_single_regressor_analysis(coef_data, regressor, max_k=8, save_plots=True):
    """
    Comprehensive analysis for a single regressor with optimal k selection.
    """
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS: {regressor}")
    print(f"{'='*60}")
    
    # Extract data
    coef_matrix, bond_ids, curve_lengths = extract_regressor_matrix(coef_data, regressor)
    K = compute_kernel_matrix(coef_matrix, kernel_type='linear', normalize=True, center=True)
    
    # Find optimal k
    print(f"\n--- Optimal K Selection ---")
    k_results = []
    for k in range(2, max_k + 1):
        labels_k, _ = alignment_clustering(K, n_clusters=k)
        sil = silhouette_score(coef_matrix, labels_k)
        sizes = list(np.bincount(labels_k))
        k_results.append({'k': k, 'silhouette': sil, 'sizes': sizes})
        print(f"  k={k}: silhouette={sil:.4f}, sizes={sizes}")
    
    best_result = max(k_results, key=lambda x: x['silhouette'])
    optimal_k = best_result['k']
    print(f"\n  --> OPTIMAL k = {optimal_k} (silhouette = {best_result['silhouette']:.4f})")
    
    # Run clustering with optimal k
    labels_align, info_align = alignment_clustering(K, n_clusters=optimal_k)
    labels_cut, info_cut = cutcost_clustering(K, n_clusters=optimal_k)
    
    sil_align = silhouette_score(coef_matrix, labels_align)
    sil_cut = silhouette_score(coef_matrix, labels_cut)
    
    print(f"\n--- Clustering Results (k={optimal_k}) ---")
    print(f"Alignment method: silhouette={sil_align:.4f}, sizes={list(np.bincount(labels_align))}")
    print(f"Cut-cost method:  silhouette={sil_cut:.4f}, sizes={list(np.bincount(labels_cut))}")
    
    # Save plots
    if save_plots:
        os.makedirs("clustering_plots", exist_ok=True)
        prefix = f"clustering_plots/{regressor}_detailed"
        
        # Plot 1: Coefficient curves
        fig1, ax1 = plt.subplots(figsize=(14, 5))
        plot_coefficient_curves(coef_matrix, title=f"Coefficient Curves: {regressor}", ax=ax1)
        fig1.savefig(f"{prefix}_1_curves.png", dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        # Plot 2: Kernel matrix
        fig2 = plot_sorted_kernel(K, labels_align, title=f"Kernel Matrix - {regressor} (k={optimal_k})")
        fig2.savefig(f"{prefix}_2_kernel.png", dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        # Plot 3: Cluster means
        fig3, ax3 = plt.subplots(figsize=(14, 5))
        plot_cluster_means(coef_matrix, labels_align, title=f"Cluster Means: {regressor} (k={optimal_k})", ax=ax3)
        fig3.savefig(f"{prefix}_3_means.png", dpi=150, bbox_inches='tight')
        plt.close(fig3)
        
        # Plot 4: Eigenspectrum
        fig4 = plot_eigenspectrum(info_align, info_cut)
        fig4.savefig(f"{prefix}_4_eigen.png", dpi=150, bbox_inches='tight')
        plt.close(fig4)
        
        # Plot 5: Silhouette vs k
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ks = [r['k'] for r in k_results]
        sils = [r['silhouette'] for r in k_results]
        ax5.plot(ks, sils, 'o-', linewidth=2, markersize=8, color='steelblue')
        ax5.axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
        ax5.set_xlabel('Number of Clusters (k)')
        ax5.set_ylabel('Silhouette Score')
        ax5.set_title(f'Silhouette Score vs k: {regressor}')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        fig5.savefig(f"{prefix}_5_optimal_k.png", dpi=150, bbox_inches='tight')
        plt.close(fig5)
        
        print(f"\nPlots saved with prefix: {prefix}")
    
    # Return results
    results = {
        'regressor': regressor,
        'optimal_k': optimal_k,
        'k_results': k_results,
        'coef_matrix': coef_matrix,
        'bond_ids': bond_ids,
        'labels': labels_align,
        'silhouette': sil_align,
        'kernel_matrix': K
    }
    
    return results

# %%
# Run detailed analysis for Carbon_Price_Index
single_results = detailed_single_regressor_analysis(coef_data, 'Carbon_Price_Index', max_k=8)

# %%