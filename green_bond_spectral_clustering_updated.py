# %% [markdown]
# # Spectral Clustering for Green Bond Coefficient Curves (Updated)
#
# Based on: Cristianini, Shawe-Taylor, Kandola (2001) "Spectral Kernel Methods for Clustering"
#
# Data structure from R:
#   results$models$coefficients_curve[[bond_id]][[regressor]]$coefficient
#
# Each bond has 12 regressors, each with a time-varying coefficient curve (the diagonal
# of the bivariate coefficient function γ(s,t) from function-on-function regression).
#
# ## Key Updates (based on methodological review):
# 1. Added min-max normalization (0-1 transformation) before kernel computation
# 2. Added option to cluster on multiple regressors jointly
# 3. Added extraction of off-diagonals for lag effect analysis
# 4. Improved documentation of the coefficient matrix interpretation
#
# ## Interpretation of γ(s,t) matrix:
# - Column-wise (fix t, vary s): Increasing smoothing at fixed response time t
# - Diagonal (s=t): Contemporaneous effects (no smoothing)
# - Off-diagonals below main: Lag effects (k-th subdiagonal = k-lag effect)
# - Bottom row: Maximally smoothed estimates with decreasing smoothing as t increases
#
# Note: Taking only the diagonal effectively reduces the functional regression
# to a time-varying simple regression, discarding lag structure.

# %%
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
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


# %%
# =============================================================================
# COEFFICIENT MATRIX PREPARATION
# =============================================================================

def extract_regressor_matrix(coef_data, regressor, pad_value=0.0):
    """
    Extract coefficient curves for a single regressor across all bonds.
    
    These are the DIAGONAL extractions γ(t,t) from the bivariate coefficient
    function, capturing contemporaneous effects only.
    
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
            continue
        
        # Skip invalid curves
        if len(coef) == 0 or np.all(np.isnan(coef)):
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


def extract_multiple_regressors_matrix(coef_data, regressors, pad_value=0.0):
    """
    Extract and concatenate coefficient curves for MULTIPLE regressors.
    
    This allows clustering bonds based on their joint sensitivity profile
    across multiple factors, rather than clustering per-regressor.
    
    Parameters:
        coef_data: dict of {bond_id: {regressor: {'coefficient': array, ...}}}
        regressors: list of regressor names to include
        pad_value: float, value to use for padding
    
    Returns:
        coef_matrix: np.array of shape (n_bonds, sum of curve lengths)
        bond_ids: list of bond identifiers
        regressor_boundaries: dict mapping regressor to (start_idx, end_idx) in concatenated vector
    """
    # First, find bonds that have ALL specified regressors
    valid_bonds = []
    for bond_id, bond_regressors in coef_data.items():
        if all(reg in bond_regressors for reg in regressors):
            valid_bonds.append(bond_id)
    
    print(f"\nJoint extraction for {len(regressors)} regressors")
    print(f"  Bonds with all regressors: {len(valid_bonds)}")
    
    if len(valid_bonds) == 0:
        raise ValueError("No bonds have all specified regressors")
    
    # Extract each regressor's curves
    all_curves = []
    regressor_boundaries = {}
    current_idx = 0
    
    for reg in regressors:
        reg_curves = []
        for bond_id in valid_bonds:
            reg_data = coef_data[bond_id][reg]
            if isinstance(reg_data, dict) and 'coefficient' in reg_data:
                coef = np.array(reg_data['coefficient'])
            else:
                coef = np.array(reg_data).flatten()
            reg_curves.append(coef)
        
        # Pad within regressor
        max_len = max(len(c) for c in reg_curves)
        padded = [np.pad(c, (0, max_len - len(c)), constant_values=pad_value) 
                  if len(c) < max_len else c for c in reg_curves]
        
        regressor_boundaries[reg] = (current_idx, current_idx + max_len)
        current_idx += max_len
        all_curves.append(np.array(padded))
    
    # Concatenate horizontally
    coef_matrix = np.hstack(all_curves)
    
    # Handle NaN
    coef_matrix = np.nan_to_num(coef_matrix, nan=pad_value)
    
    print(f"  Final matrix shape: {coef_matrix.shape}")
    print(f"  Regressor boundaries: {regressor_boundaries}")
    
    return coef_matrix, valid_bonds, regressor_boundaries


# %%
# =============================================================================
# PRE-PROCESSING: MIN-MAX NORMALIZATION (NEW - per Gareth's recommendation)
# =============================================================================

def minmax_normalize_curves(coef_matrix, feature_range=(0, 1)):
    """
    Apply min-max normalization to coefficient curves.
    
    This is CRITICAL for spectral clustering as it:
    1. Removes scale differences between bonds
    2. Makes all coefficients positive (important for kernel interpretation)
    3. Focuses clustering on SHAPE rather than magnitude
    
    Parameters:
        coef_matrix: np.array of shape (n_bonds, n_timepoints)
        feature_range: tuple (min, max) for scaling
    
    Returns:
        normalized_matrix: np.array with same shape, values in [0, 1]
        scaler: fitted MinMaxScaler (for inverse transform if needed)
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    
    # Normalize each bond's curve independently (row-wise)
    normalized = np.zeros_like(coef_matrix)
    for i in range(coef_matrix.shape[0]):
        curve = coef_matrix[i].reshape(-1, 1)
        normalized[i] = scaler.fit_transform(curve).flatten()
    
    print(f"  Min-max normalized to range {feature_range}")
    print(f"  Original range: [{coef_matrix.min():.4f}, {coef_matrix.max():.4f}]")
    print(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    return normalized, scaler


def minmax_normalize_global(coef_matrix, feature_range=(0, 1)):
    """
    Apply GLOBAL min-max normalization across all bonds.
    
    Unlike per-bond normalization, this preserves relative magnitude
    differences between bonds while still scaling to [0, 1].
    
    Use this when magnitude differences ARE meaningful for clustering.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    
    # Fit on all data, transform
    flat = coef_matrix.flatten().reshape(-1, 1)
    scaler.fit(flat)
    
    normalized = scaler.transform(coef_matrix.reshape(-1, 1)).reshape(coef_matrix.shape)
    
    print(f"  Global min-max normalized to range {feature_range}")
    
    return normalized, scaler


# %%
# =============================================================================
# KERNEL COMPUTATION
# =============================================================================

def compute_kernel_matrix(X, kernel_type='linear', gamma=None, 
                          normalize_rows=True, center=True, 
                          minmax_normalize=True, minmax_global=False):
    """
    Compute the kernel (similarity) matrix following Cristianini et al. (2001).
    
    UPDATED: Now includes optional min-max normalization as pre-processing step.
    
    Parameters:
        X: np.array of shape (n_samples, n_features)
        kernel_type: 'linear' or 'rbf'
        gamma: float, RBF kernel bandwidth (default: 1/n_features)
        normalize_rows: bool, normalize rows to unit norm before kernel
        center: bool, center kernel matrix in feature space
        minmax_normalize: bool, apply min-max normalization first (RECOMMENDED)
        minmax_global: bool, if True use global normalization, else per-row
    
    Returns:
        K: np.array of shape (n_samples, n_samples), the kernel matrix
    """
    X = X.copy()
    
    # Step 1: Min-max normalization (NEW)
    if minmax_normalize:
        if minmax_global:
            X, _ = minmax_normalize_global(X)
        else:
            X, _ = minmax_normalize_curves(X)
    
    # Step 2: Row normalization to unit norm
    if normalize_rows:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X = X / norms
    
    # Step 3: Compute kernel
    if kernel_type == 'linear':
        K = X @ X.T
    elif kernel_type == 'rbf':
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = np.exp(-gamma * sq_dists)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Step 4: Center kernel matrix
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
    eigenvalues, eigenvectors = linalg.eigh(K)
    
    # Sort by decreasing eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    K_norm = np.linalg.norm(K, 'fro')
    
    if n_clusters == 2:
        v1 = eigenvectors[:, 0]
        
        sorted_vals = np.sort(np.unique(v1))
        thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
        
        best_alignment = -np.inf
        all_alignments = []
        
        for thresh in thresholds:
            y = np.where(v1 >= thresh, 1, -1)
            yy_outer = np.outer(y, y)
            alignment = np.sum(K * yy_outer) / K_norm
            all_alignments.append((thresh, alignment))
            
            if alignment > best_alignment:
                best_alignment = alignment
                best_threshold = thresh
                best_labels = (y + 1) // 2
        
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
        V = eigenvectors[:, :n_clusters]
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
    
    Uses the Fiedler vector (second smallest eigenvector) of the graph Laplacian.
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
# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_coefficient_curves(coef_matrix, title="Coefficient Curves", 
                           n_sample=50, ax=None):
    """Plot a sample of coefficient curves."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    
    n_bonds = coef_matrix.shape[0]
    n_show = min(n_sample, n_bonds)
    
    np.random.seed(42)
    sample_idx = np.random.choice(n_bonds, n_show, replace=False)
    
    for i in sample_idx:
        ax.plot(coef_matrix[i], alpha=0.4, linewidth=0.8)
    
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(f'{title} (showing {n_show}/{n_bonds} bonds)')
    
    return ax


def plot_cluster_means(coef_matrix, labels, title="Cluster Means", ax=None):
    """Plot mean coefficient curves by cluster with confidence bands."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    
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
    
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(title)
    ax.legend()
    
    return ax


def plot_sorted_kernel(K, labels, title="Kernel Matrix"):
    """Plot kernel matrix sorted by cluster assignment."""
    idx = np.argsort(labels)
    K_sorted = K[idx][:, idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(K, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Original Order')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(K_sorted, cmap='RdBu_r', aspect='auto')
    axes[1].set_title('Sorted by Cluster')
    plt.colorbar(im2, ax=axes[1])
    
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_eigenspectrum(info_align, info_cut, n_show=20):
    """Plot eigenvalue spectra for both methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    eig_K = info_align['eigenvalues'][:n_show]
    axes[0].bar(range(len(eig_K)), eig_K, color='steelblue')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title('Kernel Matrix Eigenvalues')
    
    eig_L = info_cut['eigenvalues'][:n_show]
    axes[1].bar(range(len(eig_L)), eig_L, color='coral')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].set_title('Laplacian Eigenvalues (gaps → clusters)')
    
    plt.tight_layout()
    return fig


# %%
# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def run_spectral_clustering(coef_matrix, n_clusters=2, 
                            minmax_normalize=True, minmax_global=False,
                            show_plots=True):
    """
    Run complete spectral clustering analysis on coefficient matrix.
    
    Parameters:
        coef_matrix: np.array of shape (n_bonds, n_timepoints)
        n_clusters: int, number of clusters
        minmax_normalize: bool, apply min-max normalization (RECOMMENDED)
        minmax_global: bool, use global vs per-row normalization
        show_plots: bool, display plots
    
    Returns:
        results: dict with labels, silhouette scores, kernel matrix, etc.
    """
    print(f"\n{'='*60}")
    print(f"Running spectral clustering (k={n_clusters})")
    print(f"  Min-max normalization: {minmax_normalize} (global={minmax_global})")
    print(f"{'='*60}")
    
    # Compute kernel
    K = compute_kernel_matrix(
        coef_matrix, 
        kernel_type='linear',
        normalize_rows=True,
        center=True,
        minmax_normalize=minmax_normalize,
        minmax_global=minmax_global
    )
    
    # Clustering
    labels_align, info_align = alignment_clustering(K, n_clusters=n_clusters)
    labels_cut, info_cut = cutcost_clustering(K, n_clusters=n_clusters)
    
    # Evaluation
    sil_align = silhouette_score(coef_matrix, labels_align)
    sil_cut = silhouette_score(coef_matrix, labels_cut)
    agreement = (labels_align == labels_cut).mean()
    
    print(f"\nResults:")
    print(f"  Alignment: silhouette={sil_align:.4f}, sizes={list(np.bincount(labels_align))}")
    print(f"  Cut-cost:  silhouette={sil_cut:.4f}, sizes={list(np.bincount(labels_cut))}")
    print(f"  Agreement: {agreement*100:.1f}%")
    
    if show_plots:
        # Plot 1: Curves
        fig1, ax1 = plt.subplots(figsize=(14, 5))
        plot_coefficient_curves(coef_matrix, ax=ax1)
        plt.show()
        
        # Plot 2: Kernel matrix
        fig2 = plot_sorted_kernel(K, labels_align)
        plt.show()
        
        # Plot 3: Cluster means
        fig3, ax3 = plt.subplots(figsize=(14, 5))
        plot_cluster_means(coef_matrix, labels_align, ax=ax3)
        plt.show()
        
        # Plot 4: Eigenspectrum
        fig4 = plot_eigenspectrum(info_align, info_cut)
        plt.show()
    
    results = {
        'coef_matrix': coef_matrix,
        'kernel_matrix': K,
        'labels_alignment': labels_align,
        'labels_cutcost': labels_cut,
        'silhouette_alignment': sil_align,
        'silhouette_cutcost': sil_cut,
        'agreement': agreement,
        'info_alignment': info_align,
        'info_cutcost': info_cut
    }
    
    return results


def run_single_regressor_analysis(coef_data, regressor, n_clusters=2,
                                   minmax_normalize=True, show_plots=True):
    """
    Complete analysis for a single regressor.
    """
    # Extract data
    coef_matrix, bond_ids, curve_lengths = extract_regressor_matrix(coef_data, regressor)
    
    # Run clustering
    results = run_spectral_clustering(
        coef_matrix, 
        n_clusters=n_clusters,
        minmax_normalize=minmax_normalize,
        show_plots=show_plots
    )
    
    results['regressor'] = regressor
    results['bond_ids'] = bond_ids
    results['curve_lengths'] = curve_lengths
    
    return results


def run_joint_regressor_analysis(coef_data, regressors, n_clusters=2,
                                  minmax_normalize=True, show_plots=True):
    """
    Cluster bonds based on their JOINT sensitivity profile across multiple regressors.
    
    This addresses the issue of clustering per-regressor vs. on collections of variables.
    """
    print(f"\n{'='*60}")
    print(f"JOINT CLUSTERING on {len(regressors)} regressors")
    print(f"  Regressors: {regressors}")
    print(f"{'='*60}")
    
    # Extract concatenated matrix
    coef_matrix, bond_ids, boundaries = extract_multiple_regressors_matrix(
        coef_data, regressors
    )
    
    # Run clustering
    results = run_spectral_clustering(
        coef_matrix,
        n_clusters=n_clusters,
        minmax_normalize=minmax_normalize,
        show_plots=show_plots
    )
    
    results['regressors'] = regressors
    results['bond_ids'] = bond_ids
    results['regressor_boundaries'] = boundaries
    
    return results


def find_optimal_k(coef_matrix, max_k=6, minmax_normalize=True):
    """
    Find optimal number of clusters using silhouette score.
    """
    print(f"\nFinding optimal k (max={max_k})...")
    
    K = compute_kernel_matrix(
        coef_matrix,
        minmax_normalize=minmax_normalize
    )
    
    k_results = []
    for k in range(2, max_k + 1):
        labels, _ = alignment_clustering(K, n_clusters=k)
        sil = silhouette_score(coef_matrix, labels)
        sizes = list(np.bincount(labels))
        k_results.append({'k': k, 'silhouette': sil, 'sizes': sizes})
        print(f"  k={k}: silhouette={sil:.4f}, sizes={sizes}")
    
    best = max(k_results, key=lambda x: x['silhouette'])
    print(f"\n  --> Optimal k = {best['k']} (silhouette = {best['silhouette']:.4f})")
    
    return best['k'], k_results


# %%
# =============================================================================
# COMPARISON: WITH vs WITHOUT MIN-MAX NORMALIZATION
# =============================================================================

def compare_normalization_effect(coef_data, regressor, n_clusters=2):
    """
    Compare clustering results with and without min-max normalization.
    
    This helps understand the impact of Gareth's recommended pre-processing.
    """
    print(f"\n{'='*60}")
    print(f"COMPARING NORMALIZATION EFFECT: {regressor}")
    print(f"{'='*60}")
    
    coef_matrix, bond_ids, _ = extract_regressor_matrix(coef_data, regressor)
    
    # Without min-max normalization
    print("\n--- WITHOUT min-max normalization ---")
    K_no_norm = compute_kernel_matrix(coef_matrix, minmax_normalize=False)
    labels_no_norm, _ = alignment_clustering(K_no_norm, n_clusters=n_clusters)
    sil_no_norm = silhouette_score(coef_matrix, labels_no_norm)
    
    # With min-max normalization (per-row)
    print("\n--- WITH min-max normalization (per-row) ---")
    K_with_norm = compute_kernel_matrix(coef_matrix, minmax_normalize=True, minmax_global=False)
    labels_with_norm, _ = alignment_clustering(K_with_norm, n_clusters=n_clusters)
    sil_with_norm = silhouette_score(coef_matrix, labels_with_norm)
    
    # With global min-max normalization
    print("\n--- WITH min-max normalization (global) ---")
    K_global_norm = compute_kernel_matrix(coef_matrix, minmax_normalize=True, minmax_global=True)
    labels_global_norm, _ = alignment_clustering(K_global_norm, n_clusters=n_clusters)
    sil_global_norm = silhouette_score(coef_matrix, labels_global_norm)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Without normalization:     silhouette={sil_no_norm:.4f}, sizes={list(np.bincount(labels_no_norm))}")
    print(f"With per-row normalization: silhouette={sil_with_norm:.4f}, sizes={list(np.bincount(labels_with_norm))}")
    print(f"With global normalization:  silhouette={sil_global_norm:.4f}, sizes={list(np.bincount(labels_global_norm))}")
    
    # Agreement between methods
    agree_no_vs_row = (labels_no_norm == labels_with_norm).mean()
    agree_no_vs_global = (labels_no_norm == labels_global_norm).mean()
    agree_row_vs_global = (labels_with_norm == labels_global_norm).mean()
    
    print(f"\nAgreement rates:")
    print(f"  No-norm vs Row-norm: {agree_no_vs_row*100:.1f}%")
    print(f"  No-norm vs Global-norm: {agree_no_vs_global*100:.1f}%")
    print(f"  Row-norm vs Global-norm: {agree_row_vs_global*100:.1f}%")
    
    return {
        'no_norm': {'labels': labels_no_norm, 'silhouette': sil_no_norm},
        'row_norm': {'labels': labels_with_norm, 'silhouette': sil_with_norm},
        'global_norm': {'labels': labels_global_norm, 'silhouette': sil_global_norm}
    }


# %%
# =============================================================================
# EXECUTION: RUN ANALYSIS IF DATA EXISTS
# =============================================================================

import os

if __name__ == "__main__":
    print("="*60)
    print("GREEN BOND SPECTRAL CLUSTERING - UPDATED VERSION")
    print("="*60)
    print("\nKey updates:")
    print("1. Min-max normalization added (Gareth's recommendation)")
    print("2. Joint clustering across multiple regressors supported")
    print("3. Improved documentation of γ(s,t) interpretation")
    
    # Check if data file exists
    if os.path.exists(CSV_PATH):
        print(f"\n>>> Found data file: {CSV_PATH}")
        print(">>> Running analysis...\n")
        
        # Load data
        coef_data = load_coefficient_curves_from_csv(CSV_PATH)
        print(f"Loaded {len(coef_data)} bonds")
        
        # Check available regressors
        first_bond = list(coef_data.keys())[0]
        available_regressors = list(coef_data[first_bond].keys())
        print(f"Available regressors: {available_regressors}")
        
        # Create output directory
        OUTPUT_DIR = "clustering_results"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"\nOutput directory: {OUTPUT_DIR}/")
        
        # =====================================================================
        # ANALYSIS 1: Single regressor with normalization comparison
        # =====================================================================
        print("\n" + "="*60)
        print("ANALYSIS 1: Normalization comparison for Carbon_Price_Index")
        print("="*60)
        
        if 'Carbon_Price_Index' in available_regressors:
            norm_comparison = compare_normalization_effect(coef_data, 'Carbon_Price_Index', n_clusters=2)
        else:
            # Use first available regressor
            reg = available_regressors[0]
            print(f"Carbon_Price_Index not found, using {reg}")
            norm_comparison = compare_normalization_effect(coef_data, reg, n_clusters=2)
        
        # =====================================================================
        # ANALYSIS 2: Run clustering for all regressors
        # =====================================================================
        print("\n" + "="*60)
        print("ANALYSIS 2: Clustering all regressors (with min-max normalization)")
        print("="*60)
        
        all_results = {}
        summary_rows = []
        
        for reg in available_regressors:
            print(f"\n--- {reg} ---")
            try:
                results = run_single_regressor_analysis(
                    coef_data, reg, 
                    n_clusters=2, 
                    minmax_normalize=True,
                    show_plots=False
                )
                all_results[reg] = results
                
                summary_rows.append({
                    'regressor': reg,
                    'n_bonds': len(results['bond_ids']),
                    'silhouette_alignment': results['silhouette_alignment'],
                    'silhouette_cutcost': results['silhouette_cutcost'],
                    'cluster_0_size': np.sum(results['labels_alignment'] == 0),
                    'cluster_1_size': np.sum(results['labels_alignment'] == 1),
                    'agreement': results['agreement']
                })
            except Exception as e:
                print(f"  ERROR: {e}")
        
        # Save summary
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values('silhouette_alignment', ascending=False)
        summary_path = os.path.join(OUTPUT_DIR, "clustering_summary_all_regressors.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n>>> Saved summary to: {summary_path}")
        print("\nSUMMARY TABLE:")
        print(summary_df.to_string(index=False))
        
        # =====================================================================
        # ANALYSIS 3: Joint clustering on monetary policy variables
        # =====================================================================
        print("\n" + "="*60)
        print("ANALYSIS 3: Joint clustering on monetary policy variables")
        print("="*60)
        
        monetary_regressors = ['BOJ_rate', 'M_0', 'US_fed_funds_effe_rate']
        available_monetary = [r for r in monetary_regressors if r in available_regressors]
        
        if len(available_monetary) >= 2:
            print(f"Using regressors: {available_monetary}")
            joint_results = run_joint_regressor_analysis(
                coef_data, 
                available_monetary,
                n_clusters=2,
                minmax_normalize=True,
                show_plots=False
            )
            
            # Save joint clustering results
            joint_df = pd.DataFrame({
                'bond_id': joint_results['bond_ids'],
                'cluster': joint_results['labels_alignment']
            })
            joint_path = os.path.join(OUTPUT_DIR, "joint_clustering_monetary.csv")
            joint_df.to_csv(joint_path, index=False)
            print(f">>> Saved joint clustering to: {joint_path}")
        else:
            print(f"Not enough monetary regressors found. Available: {available_monetary}")
        
        # =====================================================================
        # ANALYSIS 4: Find optimal k for each regressor
        # =====================================================================
        print("\n" + "="*60)
        print("ANALYSIS 4: Finding optimal k for each regressor")
        print("="*60)
        
        optimal_k_rows = []
        for reg in available_regressors[:3]:  # Just first 3 to save time
            print(f"\n--- {reg} ---")
            try:
                coef_matrix, _, _ = extract_regressor_matrix(coef_data, reg)
                best_k, k_results = find_optimal_k(coef_matrix, max_k=6, minmax_normalize=True)
                optimal_k_rows.append({
                    'regressor': reg,
                    'optimal_k': best_k,
                    'best_silhouette': max(r['silhouette'] for r in k_results)
                })
            except Exception as e:
                print(f"  ERROR: {e}")
        
        if optimal_k_rows:
            optimal_k_df = pd.DataFrame(optimal_k_rows)
            optimal_k_path = os.path.join(OUTPUT_DIR, "optimal_k_summary.csv")
            optimal_k_df.to_csv(optimal_k_path, index=False)
            print(f"\n>>> Saved optimal k summary to: {optimal_k_path}")
        
        # =====================================================================
        # ANALYSIS 5: Generate and save plots for all regressors
        # =====================================================================
        print("\n" + "="*60)
        print("ANALYSIS 5: Generating and saving plots")
        print("="*60)
        
        PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        for reg in available_regressors:
            print(f"\n--- Plotting {reg} ---")
            try:
                # Get results (already computed)
                if reg not in all_results:
                    continue
                    
                res = all_results[reg]
                coef_matrix = res['coef_matrix']
                K = res['kernel_matrix']
                labels = res['labels_alignment']
                info_align = res['info_alignment']
                info_cut = res['info_cutcost']
                
                # Plot 1: Coefficient curves
                fig1, ax1 = plt.subplots(figsize=(14, 5))
                plot_coefficient_curves(coef_matrix, title=f"Coefficient Curves: {reg}", ax=ax1)
                fig1.savefig(os.path.join(PLOTS_DIR, f"{reg}_1_curves.png"), dpi=150, bbox_inches='tight')
                plt.close(fig1)
                
                # Plot 2: Kernel matrix (original and sorted)
                fig2 = plot_sorted_kernel(K, labels, title=f"Kernel Matrix: {reg}")
                fig2.savefig(os.path.join(PLOTS_DIR, f"{reg}_2_kernel.png"), dpi=150, bbox_inches='tight')
                plt.close(fig2)
                
                # Plot 3: Cluster means
                fig3, ax3 = plt.subplots(figsize=(14, 5))
                plot_cluster_means(coef_matrix, labels, title=f"Cluster Means: {reg}", ax=ax3)
                fig3.savefig(os.path.join(PLOTS_DIR, f"{reg}_3_cluster_means.png"), dpi=150, bbox_inches='tight')
                plt.close(fig3)
                
                # Plot 4: Eigenspectrum
                fig4 = plot_eigenspectrum(info_align, info_cut)
                fig4.suptitle(f"Eigenspectrum: {reg}")
                fig4.savefig(os.path.join(PLOTS_DIR, f"{reg}_4_eigenspectrum.png"), dpi=150, bbox_inches='tight')
                plt.close(fig4)
                
                # Plot 5: Curves colored by cluster
                fig5, ax5 = plt.subplots(figsize=(14, 5))
                colors = ['steelblue', 'coral']
                for c in [0, 1]:
                    mask = labels == c
                    for i in np.where(mask)[0][:20]:  # Show up to 20 per cluster
                        ax5.plot(coef_matrix[i], color=colors[c], alpha=0.3, linewidth=0.8)
                ax5.set_xlabel('Time Index')
                ax5.set_ylabel('Coefficient Value')
                ax5.set_title(f'Curves by Cluster: {reg} (Cluster 0=blue, Cluster 1=red)')
                fig5.savefig(os.path.join(PLOTS_DIR, f"{reg}_5_curves_by_cluster.png"), dpi=150, bbox_inches='tight')
                plt.close(fig5)
                
                print(f"  Saved 5 plots for {reg}")
                
            except Exception as e:
                print(f"  ERROR plotting {reg}: {e}")
        
        # =====================================================================
        # ANALYSIS 6: Summary comparison plot
        # =====================================================================
        print("\n--- Creating summary comparison plot ---")
        
        try:
            # Bar chart of silhouette scores
            fig_summary, ax_summary = plt.subplots(figsize=(12, 6))
            regressors_sorted = summary_df['regressor'].tolist()
            silhouettes_sorted = summary_df['silhouette_alignment'].tolist()
            
            bars = ax_summary.barh(range(len(regressors_sorted)), silhouettes_sorted, color='steelblue')
            ax_summary.set_yticks(range(len(regressors_sorted)))
            ax_summary.set_yticklabels(regressors_sorted)
            ax_summary.set_xlabel('Silhouette Score (Alignment Method)')
            ax_summary.set_title('Clustering Quality by Regressor (sorted)')
            ax_summary.axvline(x=0.25, color='red', linestyle='--', label='Good threshold (0.25)')
            ax_summary.legend()
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, silhouettes_sorted)):
                ax_summary.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            fig_summary.savefig(os.path.join(PLOTS_DIR, "00_summary_silhouette_comparison.png"), dpi=150, bbox_inches='tight')
            plt.close(fig_summary)
            print("  Saved summary comparison plot")
            
        except Exception as e:
            print(f"  ERROR creating summary plot: {e}")
        
        # =====================================================================
        # ANALYSIS 7: Cross-regressor cluster correlation
        # =====================================================================
        print("\n--- Computing cross-regressor cluster correlations ---")
        
        try:
            # Build correlation matrix of cluster assignments
            cluster_assignments = pd.DataFrame({
                reg: all_results[reg]['labels_alignment'] 
                for reg in available_regressors if reg in all_results
            })
            
            corr_matrix = cluster_assignments.corr()
            
            # Save correlation matrix
            corr_path = os.path.join(OUTPUT_DIR, "cluster_correlation_matrix.csv")
            corr_matrix.to_csv(corr_path)
            print(f"  Saved correlation matrix to: {corr_path}")
            
            # Plot correlation heatmap
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, vmin=-1, vmax=1, ax=ax_corr)
            ax_corr.set_title('Cross-Regressor Cluster Assignment Correlations')
            plt.tight_layout()
            fig_corr.savefig(os.path.join(PLOTS_DIR, "00_cluster_correlation_heatmap.png"), dpi=150, bbox_inches='tight')
            plt.close(fig_corr)
            print("  Saved correlation heatmap")
            
        except Exception as e:
            print(f"  ERROR computing correlations: {e}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nOutput files saved to: {OUTPUT_DIR}/")
        print("  - clustering_summary_all_regressors.csv")
        print("  - joint_clustering_monetary.csv")
        print("  - optimal_k_summary.csv")
        print("  - cluster_correlation_matrix.csv")
        print(f"\nPlots saved to: {PLOTS_DIR}/")
        print("  - 00_summary_silhouette_comparison.png")
        print("  - 00_cluster_correlation_heatmap.png")
        print("  - [REGRESSOR]_1_curves.png")
        print("  - [REGRESSOR]_2_kernel.png")
        print("  - [REGRESSOR]_3_cluster_means.png")
        print("  - [REGRESSOR]_4_eigenspectrum.png")
        print("  - [REGRESSOR]_5_curves_by_cluster.png")
        
    else:
        print(f"\n>>> Data file not found: {CSV_PATH}")
        print("\n>>> TO USE THIS CODE:")
        print("    1. Export coefficient curves from R (see R code below)")
        print("    2. Save as 'coefficient_curves.csv' in the same folder")
        print("    3. Run this script again")
        print("\n>>> R EXPORT CODE:")
        print("-"*60)
        print("""
library(tidyverse)

# Load your results
results <- readRDS("results.rds")

# Extract coefficient curves (these are the DIAGONALS γ(t,t))
coef_curves <- results$models$coefficients_curve

# Convert to long-format dataframe
all_curves <- map2_dfr(coef_curves, names(coef_curves), function(bond_data, bond_id) {
  map2_dfr(bond_data, names(bond_data), function(regressor_data, regressor_name) {
    tibble(
      bond_id = bond_id,
      regressor = regressor_name,
      date = regressor_data$date,
      coefficient = regressor_data$coefficient,
      se = regressor_data$se
    )
  })
})

# Save to CSV
write_csv(all_curves, "coefficient_curves.csv")

# Check dimensions
print(paste("Exported", n_distinct(all_curves$bond_id), "bonds"))
print(paste("Regressors:", paste(unique(all_curves$regressor), collapse=", ")))
""")
        print("-"*60)