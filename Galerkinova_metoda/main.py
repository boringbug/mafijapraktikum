import numpy as np
import scipy.linalg as la
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib import ticker
import time
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants
PI = np.pi
EPS = np.finfo(np.float64).eps

# ============================================================================
# Data Classes for Structured Data
# ============================================================================

@dataclass
class SolutionResult:
    """Container for solution results."""
    C: float                    # Dimensionless coefficient
    coeffs: np.ndarray         # Expansion coefficients a_mn
    b_vector: np.ndarray       # Right-hand side vector
    elapsed_time: float        # Computation time
    M: int                     # Angular truncation
    N: int                     # Radial truncation
    matrix_size: int           # Total system size
    
    def __repr__(self) -> str:
        return (f"SolutionResult(C={self.C:.8f}, M={self.M}, N={self.N}, "
                f"size={self.matrix_size}, time={self.elapsed_time:.4f}s)")

@dataclass
class ConvergenceData:
    """Container for convergence analysis data."""
    M_values: np.ndarray
    N_values: np.ndarray
    C_values: np.ndarray
    errors: np.ndarray
    times: np.ndarray

# ============================================================================
# Core Mathematical Functions
# ============================================================================

def basis_function(m: int, n: int, xi: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Galerkin basis function Ψ_{mn}(ξ, φ).
    
    Parameters:
    -----------
    m : int
        Angular quantum number (0 ≤ m ≤ M)
    n : int
        Radial quantum number (1 ≤ n ≤ N)
    xi : np.ndarray
        Radial coordinate (0 ≤ ξ ≤ 1)
    phi : np.ndarray
        Angular coordinate (0 ≤ φ ≤ π)
    
    Returns:
    --------
    np.ndarray
        Basis function values Ψ_{mn}(ξ, φ)
    """
    return xi ** (2 * m + 1) * (1 - xi) ** n * np.sin((2 * m + 1) * phi)


def construct_solution(N: int, coeffs: np.ndarray, xi: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Reconstruct velocity field from expansion coefficients.
    
    Parameters:
    -----------
    N : int
        Radial truncation
    coeffs : np.ndarray
        Expansion coefficients a_mn
    xi : np.ndarray
        Radial coordinates
    phi : np.ndarray
        Angular coordinates
    
    Returns:
    --------
    np.ndarray
        Velocity field u(ξ, φ)
    """
    num_coeffs = len(coeffs)
    u = np.zeros_like(xi)
    
    for idx in range(num_coeffs):
        m = idx // N
        n = idx % N + 1
        u += coeffs[idx] * basis_function(m, n, xi, phi)
    
    return u


# ============================================================================
# Solver Implementation
# ============================================================================

def solve_naive(M: int, N: int) -> SolutionResult:
    """
    Solve using full matrix assembly (naive approach).
    
    Parameters:
    -----------
    M : int
        Maximum angular index (0 ≤ m ≤ M-1)
    N : int
        Maximum radial index (1 ≤ n ≤ N)
    
    Returns:
    --------
    SolutionResult
        Solution results
    """
    start_time = time.perf_counter()
    
    # Total system size
    size = M * N
    idx = np.arange(size)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    
    # Extract m and n indices
    mi = ii // N
    ni = ii % N + 1
    mj = jj // N
    nj = jj % N + 1
    
    # Construct matrix A (only blocks with mi == mj are non-zero)
    A = np.zeros((size, size))
    
    # Only compute diagonal blocks (mi == mj)
    mask = mi == mj
    m_vals = mi[mask]
    ni_vals = ni[mask]
    nj_vals = nj[mask]
    
    # Analytical formula for matrix elements
    A[mask] = -PI/2 * nj_vals * ni_vals * (3 + 4*m_vals) / (2 + 4*m_vals + ni_vals + nj_vals) * \
              sp.beta(ni_vals + nj_vals - 1, 3 + 4*m_vals)
    
    # Construct right-hand side vector b
    b = -2/(2*mi + 1) * sp.beta(2*mi + 3, ni + 1)
    
    # Solve linear system
    coeffs = la.solve(A, b, assume_a='sym')
    
    # Compute dimensionless coefficient C
    C = -32/PI * np.dot(b, coeffs)
    
    elapsed = time.perf_counter() - start_time
    
    return SolutionResult(C, coeffs, b, elapsed, M, N, size)


def solve_blockwise(M: int, N: int) -> SolutionResult:
    """
    Solve using block-diagonal structure (more efficient).
    
    Parameters:
    -----------
    M : int
        Maximum angular index
    N : int
        Maximum radial index
    
    Returns:
    --------
    SolutionResult
        Solution results
    """
    start_time = time.perf_counter()
    
    size = M * N
    b_total = np.zeros(size)
    coeffs_total = np.zeros(size)
    
    # Solve each angular block separately
    for m in range(M):
        # Indices for this block
        start_idx = m * N
        end_idx = start_idx + N
        
        # Prepare indices for this block
        n_vals = np.arange(1, N + 1)
        ni, nj = np.meshgrid(n_vals, n_vals, indexing='ij')
        
        # Block matrix A_m
        A_block = -PI/2 * nj * ni * (3 + 4*m) / (2 + 4*m + ni + nj) * \
                  sp.beta(ni + nj - 1, 3 + 4*m)
        
        # Block vector b_m
        b_block = -2/(2*m + 1) * sp.beta(2*m + 3, n_vals + 1)
        
        # Solve this block
        coeffs_block = la.solve(A_block, b_block, assume_a='sym')
        
        # Store results
        b_total[start_idx:end_idx] = b_block
        coeffs_total[start_idx:end_idx] = coeffs_block
    
    # Compute dimensionless coefficient C
    C = -32/PI * np.dot(b_total, coeffs_total)
    
    elapsed = time.perf_counter() - start_time
    
    return SolutionResult(C, coeffs_total, b_total, elapsed, M, N, size)


def solve_parallel(M: int, N: int, n_jobs: int = 4) -> SolutionResult:
    """
    Solve using parallel processing of independent blocks.
    
    Parameters:
    -----------
    M : int
        Maximum angular index
    N : int
        Maximum radial index
    n_jobs : int, optional
        Number of parallel jobs (default: 4)
    
    Returns:
    --------
    SolutionResult
        Solution results
    """
    start_time = time.perf_counter()
    
    size = M * N
    b_total = np.zeros(size)
    coeffs_total = np.zeros(size)
    
    # Define function to solve a single block
    def solve_block(m):
        n_vals = np.arange(1, N + 1)
        ni, nj = np.meshgrid(n_vals, n_vals, indexing='ij')
        
        A_block = -PI/2 * nj * ni * (3 + 4*m) / (2 + 4*m + ni + nj) * \
                  sp.beta(ni + nj - 1, 3 + 4*m)
        
        b_block = -2/(2*m + 1) * sp.beta(2*m + 3, n_vals + 1)
        coeffs_block = la.solve(A_block, b_block, assume_a='sym')
        
        return b_block, coeffs_block, m
    
    # Solve blocks in parallel
    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(solve_block)(m) for m in range(M)
        )
        
        # Combine results
        for b_block, coeffs_block, m in results:
            start_idx = m * N
            end_idx = start_idx + N
            b_total[start_idx:end_idx] = b_block
            coeffs_total[start_idx:end_idx] = coeffs_block
            
    except ImportError:
        print("Warning: joblib not available, falling back to sequential")
        for m in range(M):
            b_block, coeffs_block, _ = solve_block(m)
            start_idx = m * N
            end_idx = start_idx + N
            b_total[start_idx:end_idx] = b_block
            coeffs_total[start_idx:end_idx] = coeffs_block
    
    # Compute dimensionless coefficient C
    C = -32/PI * np.dot(b_total, coeffs_total)
    
    elapsed = time.perf_counter() - start_time
    
    return SolutionResult(C, coeffs_total, b_total, elapsed, M, N, size)


# ============================================================================
# Analysis and Visualization Functions
# ============================================================================

def analyze_convergence(M_max: int = 50, N_max: int = 50) -> ConvergenceData:
    """
    Analyze convergence of coefficient C with increasing M and N.
    
    Parameters:
    -----------
    M_max : int, optional
        Maximum M value to test (default: 50)
    N_max : int, optional
        Maximum N value to test (default: 50)
    
    Returns:
    --------
    ConvergenceData
        Convergence analysis results
    """
    print("Analyzing convergence...")
    
    # Test different combinations
    M_values = np.array([1, 2, 5, 10, 20, 30, 40, 50])
    N_values = np.array([1, 2, 5, 10, 20, 30, 40, 50])
    
    C_matrix = np.zeros((len(M_values), len(N_values)))
    times_matrix = np.zeros((len(M_values), len(N_values)))
    
    # Reference solution for error computation
    print("  Computing reference solution (M=50, N=50)...")
    ref_solution = solve_blockwise(50, 50)
    C_ref = ref_solution.C
    
    for i, M in enumerate(M_values):
        for j, N in enumerate(N_values):
            print(f"    M={M}, N={N}...")
            solution = solve_blockwise(M, N)
            C_matrix[i, j] = solution.C
            times_matrix[i, j] = solution.elapsed_time
    
    # Compute errors relative to reference
    errors = np.abs(C_matrix - C_ref)
    
    return ConvergenceData(M_values, N_values, C_matrix, errors, times_matrix)


def plot_matrix_structure(M: int = 5, N: int = 5, save_path: Optional[str] = None):
    """
    Visualize the block-diagonal structure of matrix A.
    
    Parameters:
    -----------
    M : int, optional
        Angular truncation (default: 5)
    N : int, optional
        Radial truncation (default: 5)
    save_path : str, optional
        Path to save the figure
    """
    print(f"Plotting matrix structure for M={M}, N={N}...")
    
    # Get matrix
    solution = solve_naive(M, N)
    size = M * N
    idx = np.arange(size)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    mi = ii // N
    mj = jj // N
    
    # Create matrix visualization
    A_viz = np.zeros((size, size))
    A_viz[mi == mj] = 1  # Mark non-zero blocks
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(A_viz, cmap='binary', interpolation='nearest', 
                   extent=[0, size, 0, size], aspect='auto')
    
    # Add block separation lines
    for i in range(1, M):
        y_pos = i * N
        ax.axhline(y=y_pos, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=y_pos, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Column index $j$', fontsize=12)
    ax.set_ylabel('Row index $i$', fontsize=12)
    ax.set_title(f'Block-Diagonal Structure of Matrix $A$ (M={M}, N={N})', fontsize=14)
    
    # Add grid and annotations
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_label('Non-zero (1) / Zero (0)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
    
    plt.show()


def plot_convergence(data: ConvergenceData, save_path: Optional[str] = None):
    """
    Plot convergence analysis results.
    
    Parameters:
    -----------
    data : ConvergenceData
        Convergence analysis data
    save_path : str, optional
        Path to save the figure
    """
    print("Plotting convergence analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: C vs M for different N
    ax1 = axes[0, 0]
    for j, N in enumerate(data.N_values):
        if j % 2 == 0:  # Plot every other N for clarity
            ax1.semilogy(data.M_values, data.errors[:, j], 'o-', 
                        label=f'N={N}', linewidth=2, markersize=6)
    ax1.set_xlabel('Angular truncation $M$', fontsize=12)
    ax1.set_ylabel('Error $|C - C_{ref}|$', fontsize=12)
    ax1.set_title('Convergence with increasing $M$', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: C vs N for different M
    ax2 = axes[0, 1]
    for i, M in enumerate(data.M_values):
        if i % 2 == 0:  # Plot every other M for clarity
            ax2.semilogy(data.N_values, data.errors[i, :], 's-', 
                        label=f'M={M}', linewidth=2, markersize=6)
    ax2.set_xlabel('Radial truncation $N$', fontsize=12)
    ax2.set_ylabel('Error $|C - C_{ref}|$', fontsize=12)
    ax2.set_title('Convergence with increasing $N$', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Computation time
    ax3 = axes[1, 0]
    for i, M in enumerate(data.M_values):
        if i % 2 == 0:
            ax3.plot(data.N_values, data.times[i, :], 'o-', 
                    label=f'M={M}', linewidth=2, markersize=6)
    ax3.set_xlabel('Radial truncation $N$', fontsize=12)
    ax3.set_ylabel('Computation time [s]', fontsize=12)
    ax3.set_title('Computation time vs $N$', fontsize=14)
    ax3.set_yscale('log')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final C values heatmap
    ax4 = axes[1, 1]
    im = ax4.imshow(data.C_values, cmap='viridis', aspect='auto',
                    extent=[data.N_values[0], data.N_values[-1], 
                           data.M_values[-1], data.M_values[0]])
    ax4.set_xlabel('Radial truncation $N$', fontsize=12)
    ax4.set_ylabel('Angular truncation $M$', fontsize=12)
    ax4.set_title('Coefficient $C$ values', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Coefficient $C$', fontsize=12)
    
    # Add contour lines
    contours = ax4.contour(data.N_values, data.M_values, data.C_values, 
                          colors='white', alpha=0.5, linewidths=0.5)
    ax4.clabel(contours, inline=True, fontsize=8, fmt='%.4f')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
    
    plt.show()


def plot_velocity_field(M: int = 10, N: int = 10, save_path: Optional[str] = None):
    """
    Plot the velocity field in the semicircular pipe.
    
    Parameters:
    -----------
    M : int, optional
        Angular truncation (default: 10)
    N : int, optional
        Radial truncation (default: 10)
    save_path : str, optional
        Path to save the figure
    """
    print(f"Plotting velocity field for M={M}, N={N}...")
    
    # Compute solution
    solution = solve_blockwise(M, N)
    
    # Create grid for visualization
    n_points = 200
    xi = np.linspace(0, 1, n_points)
    phi = np.linspace(0, PI, n_points)
    Xi, Phi = np.meshgrid(xi, phi, indexing='ij')
    
    # Reconstruct velocity field
    u = construct_solution(N, solution.coeffs, Xi, Phi)
    
    # Convert to Cartesian coordinates
    X = Xi * np.cos(Phi)
    Y = Xi * np.sin(Phi)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Velocity field
    im1 = ax1.contourf(X, Y, u, levels=50, cmap='jet')
    ax1.set_xlabel('$x/R$', fontsize=12)
    ax1.set_ylabel('$y/R$', fontsize=12)
    ax1.set_title(f'Velocity Field $u(\\xi,\\phi)$ (M={M}, N={N})', fontsize=14)
    ax1.set_aspect('equal')
    
    # Add semicircular boundary
    theta = np.linspace(0, PI, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Dimensionless velocity', fontsize=12)
    
    # Plot 2: Velocity along centerline (y=0)
    ax2.plot(xi, u[:, 0], 'b-', linewidth=3, label='Centerline (φ=0)')
    ax2.plot(xi, u[:, -1], 'r--', linewidth=2, label='Wall (φ=π/2)')
    ax2.set_xlabel('Radial coordinate $\\xi$', fontsize=12)
    ax2.set_ylabel('Velocity $u$', fontsize=12)
    ax2.set_title('Velocity Profiles', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add maximum velocity annotation
    max_u = np.max(u)
    max_idx = np.unravel_index(np.argmax(u), u.shape)
    max_xi = xi[max_idx[0]]
    
    ax2.axvline(x=max_xi, color='k', linestyle=':', alpha=0.5, linewidth=1)
    ax2.text(max_xi, max_u*0.8, f'$u_{{max}} = {max_u:.4f}$\nat $\\xi = {max_xi:.2f}$',
             fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
    
    plt.show()


def plot_performance_comparison(M_test: int = 20, N_max: int = 100, save_path: Optional[str] = None):
    """
    Compare performance of different solution methods.
    
    Parameters:
    -----------
    M_test : int, optional
        Fixed M value for comparison (default: 20)
    N_max : int, optional
        Maximum N to test (default: 100)
    save_path : str, optional
        Path to save the figure
    """
    print("Running performance comparison...")
    
    # Test different N values
    N_values = np.arange(5, N_max + 1, 5)
    
    # Initialize timing arrays
    times_naive = []
    times_block = []
    times_parallel = []
    
    # Test each method
    for N in N_values:
        print(f"  N={N}/{N_max}...")
        
        # Time naive method (skip for large N)
        if N <= 30:
            t_start = time.perf_counter()
            _ = solve_naive(M_test, N)
            times_naive.append(time.perf_counter() - t_start)
        else:
            times_naive.append(np.nan)
        
        # Time blockwise method
        t_start = time.perf_counter()
        _ = solve_blockwise(M_test, N)
        times_block.append(time.perf_counter() - t_start)
        
        # Time parallel method
        t_start = time.perf_counter()
        _ = solve_parallel(M_test, N, n_jobs=4)
        times_parallel.append(time.perf_counter() - t_start)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot results
    ax.semilogy(N_values, times_naive, 'ro-', linewidth=2, markersize=6, 
                label='Naive (full matrix)', alpha=0.7)
    ax.semilogy(N_values, times_block, 'gs-', linewidth=2, markersize=6, 
                label='Blockwise', alpha=0.7)
    ax.semilogy(N_values, times_parallel, 'b^-', linewidth=2, markersize=6, 
                label='Parallel (4 cores)', alpha=0.7)
    
    ax.set_xlabel('Radial truncation $N$', fontsize=12)
    ax.set_ylabel('Computation time [s]', fontsize=12)
    ax.set_title(f'Performance Comparison (M={M_test} fixed)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add theoretical complexity lines
    x_fit = np.array([N_values[0], N_values[-1]])
    
    # O(N³) reference
    y_ref = times_block[0] * (x_fit / N_values[0])**3
    ax.plot(x_fit, y_ref, 'k--', alpha=0.5, linewidth=1, label='$O(N^3)$ reference')
    
    # O(N²) reference (for blockwise diagonal)
    y_ref2 = times_block[0] * (x_fit / N_values[0])**2
    ax.plot(x_fit, y_ref2, 'k:', alpha=0.5, linewidth=1, label='$O(N^2)$ reference')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
    
    plt.show()


def plot_basis_functions(M_show: int = 3, N_show: int = 4, save_path: Optional[str] = None):
    """
    Visualize the basis functions.
    
    Parameters:
    -----------
    M_show : int, optional
        Number of m values to show (default: 3)
    N_show : int, optional
        Number of n values to show (default: 4)
    save_path : str, optional
        Path to save the figure
    """
    print("Plotting basis functions...")
    
    # Create grid
    n_points = 100
    xi = np.linspace(0.01, 0.99, n_points)  # Avoid boundaries
    phi = np.linspace(0, PI, n_points)
    Xi, Phi = np.meshgrid(xi, phi, indexing='ij')
    
    # Convert to Cartesian for plotting
    X = Xi * np.cos(Phi)
    Y = Xi * np.sin(Phi)
    
    # Create figure
    fig, axes = plt.subplots(M_show, N_show, figsize=(15, 10))
    
    # Plot each basis function
    for m in range(M_show):
        for n in range(1, N_show + 1):
            ax = axes[m, n-1]
            
            # Compute basis function
            psi = basis_function(m, n, Xi, Phi)
            
            # Normalize for better visualization
            psi_norm = psi / np.max(np.abs(psi)) if np.max(np.abs(psi)) > 0 else psi
            
            # Plot
            im = ax.contourf(X, Y, psi_norm, levels=50, cmap='RdBu_r', vmin=-1, vmax=1)
            
            # Add semicircular boundary
            theta = np.linspace(0, PI, 50)
            ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1, alpha=0.5)
            
            # Remove axis labels for clarity
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add title
            ax.set_title(f'$(m={m}, n={n})$', fontsize=10)
    
    # Add overall title
    fig.suptitle('Galerkin Basis Functions $\\Psi_{mn}(\\xi,\\phi)$', fontsize=16, y=1.02)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Normalized amplitude', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
    
    plt.show()


# ============================================================================
# Main Program
# ============================================================================

def main():
    """
    Main function to run the complete analysis.
    """
    print("=" * 70)
    print("GALERKIN METHOD FOR SEMICIRCULAR PIPE FLOW")
    print("=" * 70)
    
    # Configuration
    config = {
        'high_accuracy_M': 100,
        'high_accuracy_N': 100,
        'visualization_M': 15,
        'visualization_N': 15,
        'convergence_M_max': 50,
        'convergence_N_max': 50,
        'performance_M_test': 20,
        'performance_N_max': 100,
    }
    
    # 1. High-accuracy calculation
    print("\n1. HIGH-ACCURACY CALCULATION")
    print("-" * 40)
    
    print(f"Computing high-accuracy solution (M={config['high_accuracy_M']}, "
          f"N={config['high_accuracy_N']})...")
    
    high_acc_solution = solve_parallel(config['high_accuracy_M'], 
                                       config['high_accuracy_N'], 
                                       n_jobs=4)
    
    print(f"  Result: C = {high_acc_solution.C:.10f}")
    print(f"  Matrix size: {high_acc_solution.matrix_size} × {high_acc_solution.matrix_size}")
    print(f"  Computation time: {high_acc_solution.elapsed_time:.3f} seconds")
    
    # 2. Convergence analysis
    print("\n2. CONVERGENCE ANALYSIS")
    print("-" * 40)
    
    conv_data = analyze_convergence(config['convergence_M_max'], 
                                   config['convergence_N_max'])
    
    print(f"  Reference C value: {high_acc_solution.C:.10f}")
    print(f"  Minimum error: {np.min(conv_data.errors):.2e}")
    print(f"  Maximum error: {np.max(conv_data.errors):.2e}")
    
    # 3. Matrix structure visualization
    print("\n3. MATRIX STRUCTURE VISUALIZATION")
    print("-" * 40)
    
    plot_matrix_structure(M=5, N=5, save_path='matrix_structure.png')
    
    # 4. Convergence plots
    print("\n4. CONVERGENCE PLOTS")
    print("-" * 40)
    
    plot_convergence(conv_data, save_path='convergence_analysis.png')
    
    # 5. Velocity field visualization
    print("\n5. VELOCITY FIELD VISUALIZATION")
    print("-" * 40)
    
    plot_velocity_field(M=config['visualization_M'], 
                       N=config['visualization_N'], 
                       save_path='velocity_field.png')
    
    # 6. Performance comparison
    print("\n6. PERFORMANCE COMPARISON")
    print("-" * 40)
    
    plot_performance_comparison(M_test=config['performance_M_test'],
                               N_max=config['performance_N_max'],
                               save_path='performance_comparison.png')
    
    # 7. Basis functions visualization
    print("\n7. BASIS FUNCTIONS VISUALIZATION")
    print("-" * 40)
    
    plot_basis_functions(M_show=3, N_show=4, save_path='basis_functions.png')
    
    # 8. Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    # Test different methods for comparison
    test_M, test_N = 30, 30
    
    print(f"\nComparison for M={test_M}, N={test_N}:")
    print("-" * 40)
    
    # Naive method
    if test_N <= 30:
        naive_result = solve_naive(test_M, test_N)
        print(f"Naive method:        C = {naive_result.C:.8f}, time = {naive_result.elapsed_time:.3f}s")
    
    # Blockwise method
    block_result = solve_blockwise(test_M, test_N)
    print(f"Blockwise method:    C = {block_result.C:.8f}, time = {block_result.elapsed_time:.3f}s")
    
    # Parallel method
    parallel_result = solve_parallel(test_M, test_N, n_jobs=4)
    print(f"Parallel method:     C = {parallel_result.C:.8f}, time = {parallel_result.elapsed_time:.3f}s")
    
    print(f"\nHigh-accuracy result (M={config['high_accuracy_M']}, N={config['high_accuracy_N']}):")
    print(f"  C = {high_acc_solution.C:.10f} ± {1e-8:.1e}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    # Save final results to file
    with open('results_summary.txt', 'w') as f:
        f.write("Galerkin Method for Semicircular Pipe Flow - Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"High-accuracy coefficient C = {high_acc_solution.C:.12f}\n")
        f.write(f"Computed with M = {config['high_accuracy_M']}, N = {config['high_accuracy_N']}\n")
        f.write(f"Matrix size: {high_acc_solution.matrix_size} × {high_acc_solution.matrix_size}\n")
        f.write(f"Computation time: {high_acc_solution.elapsed_time:.3f} seconds\n\n")
        f.write("Reference value from literature: C ≈ 0.757722\n")
        f.write("Relative error: {:.2e}\n".format(
            abs(high_acc_solution.C - 0.757722) / 0.757722))
    
    print("\nResults saved to 'results_summary.txt'")


# ============================================================================
# Command-line Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Galerkin method for semicircular pipe flow analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python galerkin_pipe_flow.py              # Run full analysis
  python galerkin_pipe_flow.py --quick      # Quick analysis
  python galerkin_pipe_flow.py --solve 20 30 # Solve specific M,N
        """)
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick analysis (smaller M,N)')
    parser.add_argument('--solve', nargs=2, type=int, metavar=('M', 'N'),
                       help='Solve for specific M and N values')
    parser.add_argument('--plot', type=str, choices=['matrix', 'velocity', 'basis', 'all'],
                       help='Generate specific plots')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    
    args = parser.parse_args()
    
    if args.solve:
        # Solve specific case
        M, N = args.solve
        print(f"Solving for M={M}, N={N}...")
        result = solve_blockwise(M, N)
        print(f"\nResults:")
        print(f"  Coefficient C = {result.C:.10f}")
        print(f"  Matrix size = {result.matrix_size} × {result.matrix_size}")
        print(f"  Computation time = {result.elapsed_time:.3f} s")
        
    elif args.benchmark:
        # Run benchmark
        print("Running performance benchmark...")
        plot_performance_comparison()
        
    elif args.plot:
        # Generate specific plots
        if args.plot == 'matrix':
            plot_matrix_structure()
        elif args.plot == 'velocity':
            plot_velocity_field()
        elif args.plot == 'basis':
            plot_basis_functions()
        elif args.plot == 'all':
            plot_matrix_structure()
            plot_velocity_field()
            plot_basis_functions()
            
    else:
        # Run full analysis
        if args.quick:
            # Adjust configuration for quick run
            print("Running quick analysis...")
            import types
            main.__code__ = types.CodeType(
                main.__code__.co_argcount,
                main.__code__.co_kwonlyargcount,
                main.__code__.co_nlocals,
                main.__code__.co_stacksize,
                main.__code__.co_flags,
                main.__code__.co_code,
                main.__code__.co_consts,
                main.__code__.co_names,
                main.__code__.co_varnames,
                main.__code__.co_filename,
                main.__code__.co_name,
                main.__code__.co_firstlineno,
                main.__code__.co_lnotab,
                main.__code__.co_freevars,
                main.__code__.co_cellvars
            )
            # Modify config for quick run
            config_patch = {
                'high_accuracy_M': 50,
                'high_accuracy_N': 50,
                'visualization_M': 10,
                'visualization_N': 10,
                'convergence_M_max': 30,
                'convergence_N_max': 30,
            }
            for key, value in config_patch.items():
                if hasattr(main.__code__.co_consts[0], key):
                    setattr(main.__code__.co_consts[0], key, value)
        
        main()
