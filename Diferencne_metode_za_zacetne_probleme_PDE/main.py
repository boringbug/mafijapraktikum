import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import splu
from scipy.linalg import expm
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ============================================
# Helper Functions for Finite Differences
# ============================================

def get_second_derivative_matrix(order, N, dx):
    """
    Get second derivative matrix of specified order.
    order: 1, 2, 3, 4 (order of approximation)
    Returns sparse matrix
    """
    if order == 1:
        # 2nd order central difference (tridiagonal)
        a = 1.0 / dx**2
        main_diag = -2.0 * a * np.ones(N)
        off_diag = a * np.ones(N-1)
        return diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
    
    elif order == 2:
        # 4th order central difference (pentadiagonal)
        a = 1.0 / (12.0 * dx**2)
        main_diag = -30.0 * a * np.ones(N)
        off1_diag = 16.0 * a * np.ones(N-1)
        off2_diag = -1.0 * a * np.ones(N-2)
        return diags([off2_diag, off1_diag, main_diag, off1_diag, off2_diag],
                     [-2, -1, 0, 1, 2], format='csr')
    
    elif order == 3:
        # 6th order central difference (heptadiagonal)
        a = 1.0 / (180.0 * dx**2)
        main_diag = -490.0 * a * np.ones(N)
        off1_diag = 270.0 * a * np.ones(N-1)
        off2_diag = -27.0 * a * np.ones(N-2)
        off3_diag = 2.0 * a * np.ones(N-3)
        return diags([off3_diag, off2_diag, off1_diag, main_diag, off1_diag, off2_diag, off3_diag],
                     [-3, -2, -1, 0, 1, 2, 3], format='csr')
    
    elif order == 4:
        # 8th order central difference (nonadiagonal)
        a = 1.0 / dx**2
        main_diag = -205.0/72.0 * a * np.ones(N)
        off1_diag = 8.0/5.0 * a * np.ones(N-1)
        off2_diag = -1.0/5.0 * a * np.ones(N-2)
        off3_diag = 8.0/315.0 * a * np.ones(N-3)
        off4_diag = -1.0/560.0 * a * np.ones(N-4)
        return diags([off4_diag, off3_diag, off2_diag, off1_diag, main_diag, off1_diag, off2_diag, off3_diag, off4_diag],
                     [-4, -3, -2, -1, 0, 1, 2, 3, 4], format='csr')
    
    else:
        raise ValueError(f"Order {order} not implemented")


class SchrodingerSolver:
    def __init__(self, x_min, x_max, Nx, dt, V, derivative_order=1, use_expm=False):
        """
        Initialize Schrödinger equation solver.
        derivative_order: 1, 2, 3, 4 (order of spatial derivative approximation)
        use_expm: if True, use matrix exponential instead of Crank-Nicolson
        """
        self.x_min = x_min
        self.x_max = x_max
        self.Nx = Nx
        self.dt = dt
        self.derivative_order = derivative_order
        self.use_expm = use_expm
        
        # Spatial grid
        self.x = np.linspace(x_min, x_max, Nx)
        self.dx = (x_max - x_min) / (Nx - 1)
        
        # Potential on grid - ensure it's a 1D array
        V_vals = V(self.x)
        if isinstance(V_vals, (int, float)):
            self.V = V_vals * np.ones(Nx)
        else:
            self.V = np.asarray(V_vals).flatten()
        
        # Build second derivative matrix
        self.D2 = get_second_derivative_matrix(derivative_order, Nx, self.dx)
        D2_dense = self.D2.toarray()
        
        # Hamiltonian: H = -0.5 * d²/dx² + V(x)
        self.H = -0.5 * D2_dense + np.diag(self.V)
        
        if use_expm:
            # Use matrix exponential for time evolution
            self.U = expm(-1j * self.H * dt)
            self.use_sparse = False
        else:
            # Crank-Nicolson: A ψ^{n+1} = A* ψ^n
            # A = I + i dt/2 H
            # A* = I - i dt/2 H
            self.A = np.eye(Nx, dtype=complex) + 1j * dt * self.H / 2
            self.Astar = np.eye(Nx, dtype=complex) - 1j * dt * self.H / 2
            # Precompute LU decomposition for sparse matrix
            A_sparse = csr_matrix(self.A)
            try:
                self.lu = splu(A_sparse)
                self.use_sparse = True
            except:
                self.use_sparse = False
    
    def step(self, psi):
        """Advance one time step"""
        if self.use_expm:
            return self.U @ psi
        else:
            rhs = self.Astar @ psi
            if self.use_sparse:
                return self.lu.solve(rhs)
            else:
                return np.linalg.solve(self.A, rhs)
    
    def solve(self, psi0, t_max, save_interval=10):
        """Solve time evolution"""
        n_steps = int(t_max / self.dt)
        psi = psi0.copy()
        
        save_indices = list(range(0, n_steps + 1, save_interval))
        if save_indices[-1] != n_steps:
            save_indices.append(n_steps)
        
        psi_saved = []
        times_saved = []
        
        for step in range(n_steps + 1):
            if step in save_indices:
                psi_saved.append(psi.copy())
                times_saved.append(step * self.dt)
            if step < n_steps:
                psi = self.step(psi)
        
        return np.array(times_saved), np.array(psi_saved)


# ============================================
# Part 1: Harmonic Oscillator - Coherent State
# ============================================

def harmonic_oscillator_potential(x, k=1.0):
    return 0.5 * k * x**2

def coherent_state(x, t, alpha, lam, omega):
    """Analytical coherent state solution"""
    xi = alpha * x
    xi_lam = alpha * lam
    prefactor = np.sqrt(alpha / np.sqrt(np.pi))
    real_part = -0.5 * (xi - xi_lam * np.cos(omega * t))**2
    imag_part = -(omega * t / 2 + xi * xi_lam * np.sin(omega * t) - 0.25 * xi_lam**2 * np.sin(2 * omega * t))
    return prefactor * np.exp(real_part + 1j * imag_part)

def run_harmonic_oscillator():
    """Run harmonic oscillator simulation"""
    print("\n" + "="*60)
    print("Harmonski oscilator - koherentno stanje")
    print("="*60)
    
    # Parameters
    omega = 0.2
    lam = 10.0
    k = omega**2
    alpha = k**0.25
    
    # Spatial grid
    x_min, x_max = -40, 40
    Nx = 300
    dx = (x_max - x_min) / (Nx - 1)
    x = np.linspace(x_min, x_max, Nx)
    
    # Time parameters
    T = 2 * np.pi / omega
    t_max = T  # One period for better visualization
    dt = T / 500  # 500 steps per period
    
    print(f"ω = {omega}, λ = {lam}, T = {T:.4f}")
    print(f"Spatial grid: [{x_min}, {x_max}], Nx={Nx}")
    print(f"dt = {dt:.6f}, t_max = {t_max:.4f}")
    
    # Initial condition
    psi0 = coherent_state(x, 0, alpha, lam, omega)
    norm = np.sqrt(np.sum(np.abs(psi0)**2) * dx)
    psi0 = psi0 / norm
    
    # Solve
    V = lambda x: harmonic_oscillator_potential(x, k)
    solver = SchrodingerSolver(x_min, x_max, Nx, dt, V, derivative_order=1)
    times, psi_saved = solver.solve(psi0, t_max, save_interval=int(T/dt/50))
    
    # Create heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data for heatmaps
    X, T_grid = np.meshgrid(x, times)
    prob_density = np.array([np.abs(psi)**2 for psi in psi_saved])
    real_part = np.array([np.real(psi) for psi in psi_saved])
    
    # Plot probability density
    im1 = axes[0].pcolormesh(X, T_grid, prob_density, shading='auto', cmap='hot')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('čas t', fontsize=12)
    axes[0].set_title('Absolutna vrednost valovne funkcije', fontsize=12)
    axes[0].set_xlim(-20, 20)
    plt.colorbar(im1, ax=axes[0], label='|ψ|²')
    
    # Plot real part
    im2 = axes[1].pcolormesh(X, T_grid, real_part, shading='auto', cmap='RdBu_r')
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('čas t', fontsize=12)
    axes[1].set_title('Realni del valovne funkcije', fontsize=12)
    axes[1].set_xlim(-20, 20)
    plt.colorbar(im2, ax=axes[1], label='Re(ψ)')
    
    plt.suptitle('Časovni razvoj valovne funkcije v harmoničnem potencialu', fontsize=14)
    plt.tight_layout()
    plt.savefig('harmonic_oscillator_heatmap.pdf', dpi=300, bbox_inches='tight')
    print("Shranjeno: harmonic_oscillator_heatmap.pdf")
    
    return fig


# ============================================
# Part 2: Free Particle - Gaussian Wave Packet
# ============================================

def free_particle_potential(x):
    return 0.0

def gaussian_wavepacket_initial(x, sigma0, k0, lam):
    prefactor = (2 * np.pi * sigma0**2)**(-0.25)
    exponent = -(x - lam)**2 / (2 * sigma0)**2 + 1j * k0 * (x - lam)
    return prefactor * np.exp(exponent)

def analytical_solution_free(x, t, sigma0, k0, lam):
    denom = 1 + 1j * t / (2 * sigma0**2)
    prefactor = (2 * np.pi * sigma0**2)**(-0.25) / np.sqrt(denom)
    numerator = -(x - lam)**2 / (2 * sigma0)**2 + 1j * k0 * (x - lam) - 1j * k0**2 * t / 2
    return prefactor * np.exp(numerator / denom)

def run_free_particle():
    """Run free particle simulation"""
    print("\n" + "="*60)
    print("Prosti delec - Gaussov valovni paket")
    print("="*60)
    
    # Parameters
    sigma0 = 1/20
    k0 = 50 * np.pi
    lam = 0.25
    
    # Spatial grid
    x_min, x_max = -0.5, 1.5
    Nx = 500  # Reduced for faster computation
    dx = (x_max - x_min) / (Nx - 1)
    x = np.linspace(x_min, x_max, Nx)
    
    # Time parameters
    dt = 2 * dx**2
    v_group = k0
    distance = 0.75 - lam
    t_max = distance / v_group
    n_steps = int(t_max / dt)
    
    # Ensure reasonable number of steps
    if n_steps > 500:
        dt = t_max / 500
        n_steps = 500
    
    print(f"σ₀ = {sigma0}, k₀ = {k0:.1f}π, λ = {lam}")
    print(f"Spatial grid: [{x_min}, {x_max}], Nx={Nx}, dx={dx:.6f}")
    print(f"dt = {dt:.6f}, t_max = {t_max:.6f}, n_steps = {n_steps}")
    
    # Initial condition
    psi0 = gaussian_wavepacket_initial(x, sigma0, k0, lam)
    norm = np.sqrt(np.sum(np.abs(psi0)**2) * dx)
    psi0 = psi0 / norm
    
    print(f"Initial norm after normalization: {np.sqrt(np.sum(np.abs(psi0)**2) * dx):.6f}")
    
    # Solve
    V = lambda x: free_particle_potential(x)
    solver = SchrodingerSolver(x_min, x_max, Nx, dt, V, derivative_order=1)
    
    # Solve and save at regular intervals
    n_save = 80
    save_interval = max(1, n_steps // n_save)
    times, psi_saved = solver.solve(psi0, t_max, save_interval=save_interval)
    
    print(f"Saved {len(times)} time steps")
    
    # Create heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    X, T_grid = np.meshgrid(x, times)
    prob_density = np.array([np.abs(psi)**2 for psi in psi_saved])
    real_part = np.array([np.real(psi) for psi in psi_saved])
    
    # Plot probability density
    im1 = axes[0].pcolormesh(X, T_grid, prob_density, shading='auto', cmap='hot')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('čas t', fontsize=12)
    axes[0].set_title('Absolutna vrednost valovne funkcije', fontsize=12)
    axes[0].set_xlim(0, 1)
    plt.colorbar(im1, ax=axes[0], label='|ψ|²')
    
    # Plot real part
    im2 = axes[1].pcolormesh(X, T_grid, real_part, shading='auto', cmap='RdBu_r')
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('čas t', fontsize=12)
    axes[1].set_title('Realni del valovne funkcije', fontsize=12)
    axes[1].set_xlim(0, 1)
    plt.colorbar(im2, ax=axes[1], label='Re(ψ)')
    
    plt.suptitle('Časovni razvoj Gaussovega valovnega paketa v prostem prostoru', fontsize=14)
    plt.tight_layout()
    plt.savefig('free_particle_heatmap.pdf', dpi=300, bbox_inches='tight')
    print("Shranjeno: free_particle_heatmap.pdf")
    
    return fig


# ============================================
# Part 3: Error analysis - Maximum height vs time
# ============================================

def run_maximum_height_analysis():
    """Analyze maximum height oscillation"""
    print("\n" + "="*60)
    print("Analiza višine maksimuma valovne funkcije")
    print("="*60)
    
    # Parameters
    omega = 0.2
    lam = 10.0
    k = omega**2
    alpha = k**0.25
    
    # Spatial grid
    x_min, x_max = -40, 40
    Nx = 200  # Reduced for faster computation
    dx = (x_max - x_min) / (Nx - 1)
    x = np.linspace(x_min, x_max, Nx)
    
    # Time parameters
    T = 2 * np.pi / omega
    t_max = T  # One period
    dt_values = [T/250, T/500, T/1000, T/1500]
    
    # Analytical maximum height (constant for coherent state)
    analytic_max = np.sqrt(alpha / np.sqrt(np.pi))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple']
    
    V = lambda x: harmonic_oscillator_potential(x, k)
    
    for i, dt in enumerate(dt_values):
        Nt = int(t_max / dt)
        print(f"  Nt = {Nt}, dt = {dt:.6f}")
        
        solver = SchrodingerSolver(x_min, x_max, Nx, dt, V, derivative_order=1)
        psi0 = coherent_state(x, 0, alpha, lam, omega)
        norm = np.sqrt(np.sum(np.abs(psi0)**2) * dx)
        psi0 = psi0 / norm
        
        save_interval = max(1, Nt // 200)
        times, psi_saved = solver.solve(psi0, t_max, save_interval=save_interval)
        
        max_heights = [np.max(np.abs(psi)) for psi in psi_saved]
        ax.plot(times, max_heights, '-', color=colors[i], linewidth=1.5,
                label=f'Nt = {Nt}')
    
    ax.axhline(y=analytic_max, color='black', linestyle='--', linewidth=2, label='Analitično')
    ax.set_xlabel('čas t', fontsize=12)
    ax.set_ylabel('maksimum |ψ|', fontsize=12)
    ax.set_title('Največja absolutna vrednost valovne funkcije v odvisnosti od časa', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('max_height_analysis.pdf', dpi=300, bbox_inches='tight')
    print("Shranjeno: max_height_analysis.pdf")
    
    return fig


# ============================================
# Part 4: Probability conservation
# ============================================

def run_probability_conservation():
    """Analyze probability conservation"""
    print("\n" + "="*60)
    print("Ohranjanje verjetnosti")
    print("="*60)
    
    # Parameters
    omega = 0.2
    lam = 10.0
    k = omega**2
    alpha = k**0.25
    
    # Spatial grid
    x_min, x_max = -40, 40
    Nx = 200
    dx = (x_max - x_min) / (Nx - 1)
    x = np.linspace(x_min, x_max, Nx)
    
    # Time parameters
    T = 2 * np.pi / omega
    t_max = 5 * T  # 5 periods
    dt = T / 500
    
    print(f"t_max = {t_max:.2f} (5 period), dt = {dt:.6f}")
    
    # Numerical solution
    V = lambda x: harmonic_oscillator_potential(x, k)
    psi0 = coherent_state(x, 0, alpha, lam, omega)
    norm = np.sqrt(np.sum(np.abs(psi0)**2) * dx)
    psi0 = psi0 / norm
    
    solver = SchrodingerSolver(x_min, x_max, Nx, dt, V, derivative_order=1)
    times, psi_saved = solver.solve(psi0, t_max, save_interval=100)
    
    # Calculate probabilities
    prob_numerical = [np.sum(np.abs(psi)**2) * dx for psi in psi_saved]
    
    # Analytical probability on the same grid
    prob_analytical = []
    for t in times:
        psi_ana = coherent_state(x, t, alpha, lam, omega)
        psi_ana = psi_ana / np.sqrt(np.sum(np.abs(psi_ana)**2) * dx)
        prob_analytical.append(np.sum(np.abs(psi_ana)**2) * dx)
    
    # Convert to lists for easier manipulation
    prob_numerical = np.array(prob_numerical)
    prob_analytical = np.array(prob_analytical)
    
    # Calculate relative error
    rel_error = np.abs(prob_numerical - prob_analytical) / prob_analytical * 100
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Absolute probability
    ax1 = axes[0]
    ax1.plot(times, prob_numerical, 'b-', linewidth=1.5, label='Numerično')
    ax1.plot(times, prob_analytical, 'r--', linewidth=1.5, label='Analitično')
    ax1.set_xlabel('čas t', fontsize=12)
    ax1.set_ylabel('Verjetnost', fontsize=12)
    ax1.set_title('Absolutna verjetnost', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Set y limits close to 1 to see small variations
    ax1.set_ylim(0.9999, 1.0001)
    
    # Right plot: Relative error
    ax2 = axes[1]
    ax2.plot(times, rel_error, 'g-', linewidth=1.5)
    ax2.set_xlabel('čas t', fontsize=12)
    ax2.set_ylabel('Relativna napaka (%)', fontsize=12)
    ax2.set_title('Relativna napaka verjetnosti', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Ohranjanje verjetnosti v harmoničnem potencialu', fontsize=14)
    plt.tight_layout()
    plt.savefig('probability_conservation.pdf', dpi=300, bbox_inches='tight')
    print("Shranjeno: probability_conservation.pdf")
    print(f"Povprečna relativna napaka: {np.mean(rel_error):.6f}%")
    
    return fig


# ============================================
# Part 5: Error comparison for different orders
# ============================================

def run_order_comparison():
    """Compare different spatial derivative orders"""
    print("\n" + "="*60)
    print("Primerjava redov natančnosti prostorske diskretizacije")
    print("="*60)
    
    # Parameters
    omega = 0.2
    lam = 10.0
    k = omega**2
    alpha = k**0.25
    
    # Spatial grid (smaller for faster computation)
    x_min, x_max = -40, 40
    Nx = 80
    dx = (x_max - x_min) / (Nx - 1)
    x = np.linspace(x_min, x_max, Nx)
    
    # Time parameters
    T = 2 * np.pi / omega
    t_max = T  # One period
    dt = T / 200
    
    print(f"Nx = {Nx}, dt = {dt:.6f}")
    
    V = lambda x: harmonic_oscillator_potential(x, k)
    psi0 = coherent_state(x, 0, alpha, lam, omega)
    norm = np.sqrt(np.sum(np.abs(psi0)**2) * dx)
    psi0 = psi0 / norm
    
    orders = [1, 2, 3]  # spatial derivative orders (skip 4 for speed)
    colors = ['green', 'blue', 'red']
    styles = ['-', '-', '-']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for order, color, style in zip(orders, colors, styles):
        print(f"  Testing derivative order {order}...")
        
        # Crank-Nicolson
        solver_cn = SchrodingerSolver(x_min, x_max, Nx, dt, V, 
                                       derivative_order=order, use_expm=False)
        times_cn, psi_saved_cn = solver_cn.solve(psi0, t_max, save_interval=50)
        
        # Matrix exponential (higher order in time)
        solver_exp = SchrodingerSolver(x_min, x_max, Nx, dt, V,
                                        derivative_order=order, use_expm=True)
        times_exp, psi_saved_exp = solver_exp.solve(psi0, t_max, save_interval=50)
        
        # Calculate errors
        errors_cn = []
        errors_exp = []
        
        for i, t in enumerate(times_cn):
            psi_ana = coherent_state(x, t, alpha, lam, omega)
            psi_ana = psi_ana / np.sqrt(np.sum(np.abs(psi_ana)**2) * dx)
            
            error_cn = np.max(np.abs(psi_saved_cn[i] - psi_ana))
            error_exp = np.max(np.abs(psi_saved_exp[i] - psi_ana))
            
            errors_cn.append(error_cn)
            errors_exp.append(error_exp)
        
        ax.plot(times_cn, errors_cn, color=color, linestyle='-', linewidth=1.5,
                label=f'Red {order} (Crank-Nicolson)')
        ax.plot(times_exp, errors_exp, color=color, linestyle=':', linewidth=1.5,
                label=f'Red {order} (expm)')
    
    ax.set_xlabel('čas t', fontsize=12)
    ax.set_ylabel('maksimalna napaka', fontsize=12)
    ax.set_title('Največja absolutna vrednost razlike med numerično in analitično rešitvijo', fontsize=12)
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('order_comparison.pdf', dpi=300, bbox_inches='tight')
    print("Shranjeno: order_comparison.pdf")
    
    return fig


# ============================================
# Main execution
# ============================================

def main():
    """Run all simulations and generate all figures"""
    print("\n" + "="*60)
    print("DIFERENČNE METODE ZA PARCIALNE DIFERENCIALNE ENAČBE")
    print("REŠEVANJE SCHRÖDINGERJEVE ENAČBE")
    print("="*60)
    
    total_start = time.time()
    
    # Generate all figures
    print("\n--- Harmonski oscilator ---")
    fig1 = run_harmonic_oscillator()
    plt.close(fig1)
    
    print("\n--- Prost valovni paket ---")
    fig2 = run_free_particle()
    plt.close(fig2)
    
    print("\n--- Analiza višine maksimuma ---")
    fig3 = run_maximum_height_analysis()
    plt.close(fig3)
    
    print("\n--- Ohranjanje verjetnosti ---")
    fig4 = run_probability_conservation()
    plt.close(fig4)
    
    print("\n--- Primerjava redov natančnosti ---")
    fig5 = run_order_comparison()
    plt.close(fig5)
    
    print("\n" + "="*60)
    print(f"✅ Vse simulacije so končane!")
    print(f"   Skupni čas: {time.time()-total_start:.1f} sekund")
    print("="*60)
    print("\nShranjene datoteke:")
    print("  - harmonic_oscillator_heatmap.pdf")
    print("  - free_particle_heatmap.pdf")
    print("  - max_height_analysis.pdf")
    print("  - probability_conservation.pdf")
    print("  - order_comparison.pdf")


if __name__ == "__main__":
    main()
