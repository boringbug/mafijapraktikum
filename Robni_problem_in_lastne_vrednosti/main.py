import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import warnings
warnings.filterwarnings('ignore')

class QuantumWellSolver:
    def __init__(self, a=1.0, V0=100.0, N=200, hbar=1.0, m=1.0):
        self.a = a
        self.V0 = V0
        self.N = N
        self.h = a / (N + 1)
        self.hbar = hbar
        self.m = m
    
    def infinite_well_analytical(self, n_max=10):
        eigenvalues = []
        eigenfunctions = []
        x = np.linspace(-self.a/2, self.a/2, 1000)
        
        for n in range(1, n_max + 1):
            E_n = (n**2 * np.pi**2 * self.hbar**2) / (2 * self.m * self.a**2)
            psi = np.sqrt(2/self.a) * np.sin(n * np.pi * (x + self.a/2) / self.a)
            eigenvalues.append(E_n)
            eigenfunctions.append(psi)
        
        return np.array(eigenvalues), np.array(eigenfunctions), x
    
    def finite_well_transcendental(self):
        energies = []
        z0 = self.a * np.sqrt(2*self.m*self.V0) / (2*self.hbar)
        
        def f_even(z):
            return np.tan(z) - np.sqrt((z0/z)**2 - 1)
        
        def f_odd(z):
            return -1/np.tan(z) - np.sqrt((z0/z)**2 - 1)
        
        z_vals = np.linspace(0.1, z0-0.1, 1000)
        
        for i in range(len(z_vals)-1):
            if f_even(z_vals[i]) * f_even(z_vals[i+1]) < 0:
                a, b = z_vals[i], z_vals[i+1]
                for _ in range(20):
                    c = (a + b) / 2
                    if f_even(a) * f_even(c) < 0:
                        b = c
                    else:
                        a = c
                z_root = (a + b) / 2
                E = (4 * z_root**2 * self.hbar**2) / (2 * self.m * self.a**2)
                if E < self.V0:
                    energies.append(E)
        
        for i in range(len(z_vals)-1):
            if f_odd(z_vals[i]) * f_odd(z_vals[i+1]) < 0:
                a, b = z_vals[i], z_vals[i+1]
                for _ in range(20):
                    c = (a + b) / 2
                    if f_odd(a) * f_odd(c) < 0:
                        b = c
                    else:
                        a = c
                z_root = (a + b) / 2
                E = (4 * z_root**2 * self.hbar**2) / (2 * self.m * self.a**2)
                if E < self.V0:
                    energies.append(E)
        
        return np.sort(energies)
    
    def solve_finite_difference(self, V_func, n_states=10):
        N = self.N
        h = self.h
        x = np.linspace(-self.a/2, self.a/2, N+2)
        
        V = V_func(x[1:-1])
        
        # Vrnem na staro shemo (kot v originalni kodi)
        main_diag = 2/h**2 + V
        off_diag = -1/h**2 * np.ones(N-1)
        
        H = np.zeros((N, N))
        np.fill_diagonal(H, main_diag)
        np.fill_diagonal(H[1:], off_diag)
        np.fill_diagonal(H[:, 1:], off_diag)
        
        eigvals, eigvecs = np.linalg.eigh(H)
        
        padded_vecs = []
        for i in range(min(n_states, len(eigvals))):
            psi = np.zeros(N+2)
            psi[1:-1] = eigvecs[:, i]
            
            norm = np.sqrt(np.trapz(psi**2, x))
            if norm > 0:
                psi /= norm
            
            padded_vecs.append(psi)
        
        return eigvals[:n_states], np.array(padded_vecs), x
    
    def shooting_method(self, V_func, parity='even', E_guess=None, tol=1e-10):
        if E_guess is None:
            E_guess = 2.0 if parity == 'even' else 8.0
        
        def schrodinger(x, y, E):
            psi, dpsi = y
            d2psi = (2 * self.m / self.hbar**2) * (V_func(x) - E) * psi
            return [dpsi, d2psi]
        
        def boundary_condition(E):
            if parity == 'even':
                y0 = [1.0, 0.0]
            else:
                y0 = [0.0, 1.0]
            
            sol = solve_ivp(lambda x, y: schrodinger(x, y, E),
                           [0, self.a/2], y0,
                           method='RK45', max_step=0.001,
                           rtol=1e-12, atol=1e-12)
            return sol.y[0, -1]
        
        E0, E1 = E_guess * 0.9, E_guess
        f0, f1 = boundary_condition(E0), boundary_condition(E1)
        
        for _ in range(50):
            if abs(f1) < tol:
                break
            
            if f1 == f0:
                break
            
            E_new = E1 - f1 * (E1 - E0) / (f1 - f0)
            E0, E1 = E1, E_new
            f0, f1 = f1, boundary_condition(E1)
        
        x_full = np.linspace(-self.a/2, self.a/2, 1000)
        if parity == 'even':
            y0 = [1.0, 0.0]
        else:
            y0 = [0.0, 1.0]
        
        sol = solve_ivp(lambda x, y: schrodinger(x, y, E1),
                       [x_full[0], x_full[-1]], y0,
                       t_eval=x_full, method='RK45',
                       max_step=0.001, rtol=1e-12, atol=1e-12)
        
        psi_full = sol.y[0]
        norm = np.sqrt(np.trapz(psi_full**2, x_full))
        if norm > 0:
            psi_full /= norm
        
        return E1, psi_full, x_full
    
    def plot_wavefunction_comparison(self):
        def V_infinite(x):
            return np.zeros_like(x)
        
        E_anal, psi_anal, x_anal = self.infinite_well_analytical(3)
        E_num, psi_num, x_num = self.solve_finite_difference(V_infinite, 3)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i in range(3):
            ax = axes[i]
            ax.plot(x_anal, psi_anal[i], color=colors[i], linewidth=2.5, alpha=0.8, label='Analitično')
            ax.plot(x_num, psi_num[i], color=colors[i], linewidth=1.5, alpha=0.6, linestyle='--', label='Numerično')
            ax.set_title(f'Stanje n={i+1}', fontsize=12)
            ax.set_xlabel('$x$', fontsize=11)
            ax.set_ylabel('$\\psi(x)$', fontsize=11)
            ax.axvline(x=-self.a/2, color='r', linestyle=':', alpha=0.5)
            ax.axvline(x=self.a/2, color='r', linestyle=':', alpha=0.5)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.2)
            ax.set_xlim(-self.a/2*1.1, self.a/2*1.1)
            
            if i == 0:
                ax.legend(fontsize=10)
        
        plt.suptitle('Primerjava valovnih funkcij: neskončna potencialna jama',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('wavefunction_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ wavefunction_comparison.png")
    
    def plot_finite_well(self):
        def V_finite(x):
            return np.where(np.abs(x) <= self.a/2, 0, self.V0)
        
        E_num, psi_num, x = self.solve_finite_difference(V_finite, 10)
        
        bound_mask = E_num < self.V0
        E_bound = E_num[bound_mask]
        psi_bound = psi_num[bound_mask]
        
        n_states = min(4, len(E_bound))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pot = np.linspace(-self.a*1.5, self.a*1.5, 1000)
        V_pot = V_finite(x_pot)
        ax.plot(x_pot, V_pot, 'k-', linewidth=2, alpha=0.8, label='$V(x)$')
        ax.fill_between(x_pot, 0, V_pot, color='gray', alpha=0.1)
        
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, n_states))
        
        for i in range(n_states):
            offset = E_bound[i]
            scale = 0.2 * self.V0
            psi_scaled = offset + scale * psi_bound[i]
            ax.plot(x, psi_scaled, color=colors[i], linewidth=2,
                   label=f'$n={i+1},\\ E={E_bound[i]:.3f}$')
            ax.axhline(y=offset, color=colors[i], linestyle=':', alpha=0.5)
        
        ax.axhline(y=self.V0, color='k', linestyle='--', alpha=0.5, label=f'$V_0={self.V0}$')
        ax.set_title(f'Končna potencialna jama ($V_0={self.V0}$)', fontsize=14, fontweight='bold')
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$E$, $\\psi(x)$', fontsize=12)
        ax.set_xlim(-self.a*1.5, self.a*1.5)
        ax.set_ylim(-1, self.V0*1.2)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('finite_well_wavefunctions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ finite_well_wavefunctions.png")
    
    def plot_convergence_study(self):
        def V_infinite(x):
            return np.zeros_like(x)
        
        E_anal, _, _ = self.infinite_well_analytical(1)
        E_ref = E_anal[0]
        
        N_list = [10, 20, 30, 50, 70, 100, 150, 200, 300, 400, 500]
        errors = []
        times = []
        
        for N_val in N_list:
            solver = QuantumWellSolver(a=self.a, V0=self.V0, N=N_val)
            
            start = time.time()
            E_num, _, _ = solver.solve_finite_difference(V_infinite, 1)
            end = time.time()
            
            if len(E_num) > 0:
                error = abs(E_num[0] - E_ref) / E_ref
                errors.append(error)
                times.append(end - start)
            else:
                errors.append(np.nan)
                times.append(np.nan)
        
        valid = ~np.isnan(errors)
        N_valid = np.array(N_list)[valid]
        errors_valid = np.array(errors)[valid]
        times_valid = np.array(times)[valid]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        ax1.loglog(N_valid, errors_valid, 'bo-', linewidth=2, markersize=6)
        
        if len(N_valid) > 2:
            coeffs = np.polyfit(np.log(N_valid), np.log(errors_valid), 1)
            slope = coeffs[0]
            
            N_fit = np.logspace(np.log10(min(N_valid)), np.log10(max(N_valid)), 100)
            fit_curve = np.exp(coeffs[1]) * N_fit**slope
            
            # DODAM MINUS pred slope za pravilen prikaz
            ax1.loglog(N_fit, fit_curve, 'r--', linewidth=1.5,
                      label=f'Fit: $\\sim N^{{{slope:.3f}}}$')
            
            ref_error = errors_valid[0] * (N_valid[0]/N_fit)**2
            ax1.loglog(N_fit, ref_error, 'g:', linewidth=1.5,
                      label='$\\mathcal{O}(N^{-2})$')
            
            ax1.text(0.05, 0.95, f'Naklon: {slope:.3f}',
                    transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.set_title('Konvergenca napake', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Število točk $N$', fontsize=11)
        ax1.set_ylabel('Relativna napaka', fontsize=11)
        ax1.grid(True, alpha=0.2, which='both')
        ax1.legend(fontsize=10)
        
        ax2.loglog(N_valid, times_valid, 's-', color='purple',
                  linewidth=2, markersize=6)
        
        ax2.set_title('Časovna zahtevnost', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Število točk $N$', fontsize=11)
        ax2.set_ylabel('Čas [s]', fontsize=11)
        ax2.grid(True, alpha=0.2, which='both')
        
        plt.suptitle('Analiza konvergence diferenčne metode',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ convergence_analysis.png")
    
    def plot_shooting_demo(self):
        def V_infinite(x):
            return np.zeros_like(x)
        
        E_anal, _, _ = self.infinite_well_analytical(6)
        
        E_test = np.linspace(0.5, 150, 500)
        psi_ends_even = []
        psi_ends_odd = []
        
        for E in E_test:
            def rhs_even(x, y):
                psi, dpsi = y
                d2psi = (2*self.m/self.hbar**2) * (V_infinite(x) - E) * psi
                return [dpsi, d2psi]
            
            sol_even = solve_ivp(rhs_even, [0, self.a/2], [1.0, 0.0],
                                max_step=0.01, rtol=1e-9)
            psi_ends_even.append(sol_even.y[0, -1])
            
            def rhs_odd(x, y):
                psi, dpsi = y
                d2psi = (2*self.m/self.hbar**2) * (V_infinite(x) - E) * psi
                return [dpsi, d2psi]
            
            sol_odd = solve_ivp(rhs_odd, [0, self.a/2], [0.0, 1.0],
                               max_step=0.01, rtol=1e-9)
            psi_ends_odd.append(sol_odd.y[0, -1])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(E_test, psi_ends_even, 'b-', linewidth=1.5, alpha=0.8, label='Soda ($\\psi(0)=1$)')
        ax.plot(E_test, psi_ends_odd, 'r-', linewidth=1.5, alpha=0.8, label='Liha ($\\psi(0)=0$)')
        
        for i, E in enumerate(E_anal[:4]):
            color = 'b' if i % 2 == 0 else 'r'
            ax.axvline(x=E, color=color, linestyle=':', alpha=0.5)
            ax.plot(E, 0, 'o', color=color, markersize=8)
            ax.text(E, 0.3 if i%2==0 else -0.3, f'$E_{{{i+1}}}$',
                   ha='center', fontsize=9)
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlim(0, 150)
        ax.set_ylim(-2, 2)
        
        ax.set_title('Strelska metoda: $\\psi(a/2)$ kot funkcija $E$',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Energija $E$', fontsize=12)
        ax.set_ylabel('$\\psi(a/2)$', fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig('shooting_method_demo.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ shooting_method_demo.png")
    
    def plot_energy_spectra(self):
        def V_infinite(x):
            return np.zeros_like(x)
        
        def V_finite(x):
            return np.where(np.abs(x) <= self.a/2, 0, self.V0)
        
        E_inf_anal, _, _ = self.infinite_well_analytical(8)
        E_fin_num, _, _ = self.solve_finite_difference(V_finite, 15)
        bound_mask = E_fin_num < self.V0
        E_fin_bound = E_fin_num[bound_mask]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(range(1, len(E_inf_anal)+1), E_inf_anal, 'bo-',
               linewidth=2, markersize=7, label='Neskončna jama')
        
        if len(E_fin_bound) > 0:
            ax.plot(range(1, len(E_fin_bound)+1), E_fin_bound, 'rs--',
                   linewidth=2, markersize=6, label='Končna jama')
        
        ax.axhline(y=self.V0, color='k', linestyle='--', alpha=0.7,
                  label=f'$V_0={self.V0}$')
        
        ymax = max(np.max(E_inf_anal)*1.1, self.V0*1.5)
        ax.axhspan(self.V0, ymax, alpha=0.1, color='gray')
        ax.text(ax.get_xlim()[1]*0.95, (self.V0 + ymax)/2,
               'Nezavezana\nstanja', ha='right', va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title('Primerjava energijskih spektrov',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Kvantno število $n$', fontsize=12)
        ax.set_ylabel('Energija $E_n$', fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('energy_spectrum_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ energy_spectrum_comparison.png")
    
    def plot_fourth_order(self):
        N = 200
        a = self.a
        h = a / (N + 1)
        x = np.linspace(-a/2, a/2, N+2)
        
        main_diag = 6 / h**4 * np.ones(N)
        off1_diag = -4 / h**4 * np.ones(N-1)
        off2_diag = 1 / h**4 * np.ones(N-2)
        
        H = np.zeros((N, N))
        np.fill_diagonal(H, main_diag)
        np.fill_diagonal(H[1:], off1_diag)
        np.fill_diagonal(H[:, 1:], off1_diag)
        np.fill_diagonal(H[2:], off2_diag)
        np.fill_diagonal(H[:, 2:], off2_diag)
        
        eigvals, eigvecs = np.linalg.eigh(H)
        
        n_vals = np.arange(1, 7)
        E_anal = (n_vals * np.pi / a)**4
        
        print("\n=== REZULTATI 4. REDA ===")
        for i in range(3):
            ratio = eigvals[i] / E_anal[i]
            print(f"n={i+1}: E_anal={E_anal[i]:.2f}, E_num={eigvals[i]:.2f}, razmerje={ratio:.4f}")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i in range(3):
            ax = axes[i]
            psi = np.zeros(N+2)
            psi[1:-1] = eigvecs[:, i]
            
            psi_max = np.max(np.abs(psi))
            if psi_max > 0:
                psi /= psi_max
            
            ax.plot(x, psi, 'b-', linewidth=2,
                   label=f'Numerično: {eigvals[i]:.1f}')
            
            psi_anal = np.sin((i+1) * np.pi * (x + a/2) / a)
            psi_anal = psi_anal / np.max(np.abs(psi_anal))
            
            ax.plot(x, psi_anal, 'r--', linewidth=1.5, alpha=0.7,
                   label=f'Analitično: {E_anal[i]:.1f}')
            
            ax.set_title(f'Način {i+1}', fontsize=11)
            ax.set_xlabel('$x/a$', fontsize=10)
            ax.set_ylabel('$\\psi(x)$', fontsize=10)
            ax.axvline(x=-a/2, color='r', linestyle=':', alpha=0.5)
            ax.axvline(x=a/2, color='r', linestyle=':', alpha=0.5)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8)
            ax.set_xlim(-a/2*1.1, a/2*1.1)
            ax.set_ylim(-1.1, 1.1)
        
        ax = axes[3]
        n_plot = min(6, len(eigvals))
        
        ax.plot(range(1, n_plot+1), eigvals[:n_plot], 'bo-',
               linewidth=2, markersize=7, label='Numerično')
        ax.plot(range(1, n_plot+1), E_anal[:n_plot], 'rs--',
               linewidth=2, markersize=6, label='Analitično')
        
        ax.set_title('Energijski spekter', fontsize=11)
        ax.set_xlabel('Način $n$', fontsize=10)
        ax.set_ylabel('$E_n$', fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)
        ax.set_yscale('log')
        
        plt.suptitle('Problem četrtega reda: $d^4\\psi/dx^4 = E\\psi$',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('fourth_order_problem.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ fourth_order_problem.png")
    
    def plot_error_analysis(self):
        def V_infinite(x):
            return np.zeros_like(x)
        
        E_anal, _, _ = self.infinite_well_analytical(5)
        
        N_list = [20, 30, 50, 100, 200, 300, 400]
        errors = {i: [] for i in range(5)}
        
        for N_val in N_list:
            solver = QuantumWellSolver(a=self.a, V0=self.V0, N=N_val)
            E_num, _, _ = solver.solve_finite_difference(V_infinite, 5)
            
            for i in range(min(5, len(E_num))):
                err = abs(E_num[i] - E_anal[i]) / E_anal[i]
                errors[i].append(err)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, 5))
        markers = ['o', 's', '^', 'D', 'v']
        
        for i in range(5):
            ax.loglog(N_list, errors[i], marker=markers[i], color=colors[i],
                     linewidth=1.5, markersize=6, label=f'Stanje {i+1}')
        
        ax.set_title('Relativna napaka za različna stanja',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Število točk $N$', fontsize=12)
        ax.set_ylabel('Relativna napaka', fontsize=12)
        ax.grid(True, alpha=0.2, which='both')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ error_analysis.png")
    
    def plot_method_comparison(self):
        def V_infinite(x):
            return np.zeros_like(x)
        
        E_anal, _, _ = self.infinite_well_analytical(1)
        E_ref = E_anal[0]
        
        N_list = [10, 20, 50, 100, 200]
        times_diff = []
        times_shoot = []
        acc_diff = []
        acc_shoot = []
        
        for N_val in N_list:
            solver = QuantumWellSolver(a=self.a, V0=self.V0, N=N_val)
            
            start = time.time()
            E_diff, _, _ = solver.solve_finite_difference(V_infinite, 1)
            end = time.time()
            times_diff.append(end - start)
            
            if len(E_diff) > 0:
                err = abs(E_diff[0] - E_ref) / E_ref
                acc_diff.append(-np.log10(err) if err > 0 else 15)
            else:
                acc_diff.append(0)
            
            if N_val <= 100:
                start = time.time()
                try:
                    E_shoot, _, _ = solver.shooting_method(V_infinite, 'even', 2.0)
                    end = time.time()
                    times_shoot.append(end - start)
                    err = abs(E_shoot - E_ref) / E_ref
                    acc_shoot.append(-np.log10(err) if err > 0 else 15)
                except:
                    times_shoot.append(np.nan)
                    acc_shoot.append(np.nan)
            else:
                times_shoot.append(np.nan)
                acc_shoot.append(np.nan)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        ax1.plot(N_list, times_diff, 'bo-', linewidth=2, markersize=6,
                label='Diferenčna')
        
        shoot_times = [t for t in times_shoot if not np.isnan(t)]
        shoot_N = [n for n, t in zip(N_list, times_shoot) if not np.isnan(t)]
        if len(shoot_times) > 0:
            ax1.plot(shoot_N, shoot_times, 'rs--', linewidth=2, markersize=6,
                    label='Strelska')
        
        ax1.set_xlabel('Število točk $N$', fontsize=11)
        ax1.set_ylabel('Čas [s]', fontsize=11)
        ax1.set_title('Časovna zahtevnost', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.2)
        ax1.legend(fontsize=10)
        ax1.set_yscale('log')
        
        ax2.plot(N_list, acc_diff, 'bo-', linewidth=2, markersize=6,
                label='Diferenčna')
        
        shoot_acc = [a for a in acc_shoot if not np.isnan(a)]
        if len(shoot_acc) > 0:
            ax2.plot(shoot_N, shoot_acc, 'rs--', linewidth=2, markersize=6,
                    label='Strelska')
        
        ax2.set_xlabel('Število točk $N$', fontsize=11)
        ax2.set_ylabel('Natančnost ($-\\log_{10}$(napake))', fontsize=11)
        ax2.set_title('Natančnost', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.2)
        ax2.legend(fontsize=10)
        ax2.set_ylim(0, 16)
        
        plt.suptitle('Primerjava diferenčne in strelske metode',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('method_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ method_comparison.png")
    
    def generate_report_data(self):
        print("\n=== PODATKI ZA POROČILO ===")
        
        def V_infinite(x):
            return np.zeros_like(x)
        
        E_anal, _, _ = self.infinite_well_analytical(5)
        E_num, _, _ = self.solve_finite_difference(V_infinite, 5)
        
        print("\nNESKONČNA JAMA (N=200):")
        print("n\tAnalitično\tNumerično\tNapaka (%)")
        for i in range(5):
            err = abs(E_num[i] - E_anal[i]) / E_anal[i] * 100
            print(f"{i+1}\t{E_anal[i]:.6f}\t{E_num[i]:.6f}\t{err:.6f}")
        
        def V_finite(x):
            return np.where(np.abs(x) <= self.a/2, 0, self.V0)
        
        E_fin, _, _ = self.solve_finite_difference(V_finite, 10)
        bound_mask = E_fin < self.V0
        E_bound = E_fin[bound_mask]
        
        print("\nKONČNA JAMA (V0=100, N=200):")
        print("Stanje\tEnergija\tPariteta")
        parities = ['Soda', 'Liha', 'Soda', 'Liha', 'Soda']
        for i in range(min(5, len(E_bound))):
            print(f"{i+1}\t{E_bound[i]:.3f}\t{parities[i]}")
        
        N = 200
        a = self.a
        h = a / (N + 1)
        
        main_diag = 6 / h**4 * np.ones(N)
        off1_diag = -4 / h**4 * np.ones(N-1)
        off2_diag = 1 / h**4 * np.ones(N-2)
        
        H = np.zeros((N, N))
        np.fill_diagonal(H, main_diag)
        np.fill_diagonal(H[1:], off1_diag)
        np.fill_diagonal(H[:, 1:], off1_diag)
        np.fill_diagonal(H[2:], off2_diag)
        np.fill_diagonal(H[:, 2:], off2_diag)
        
        eigvals, _ = np.linalg.eigh(H)
        
        n_vals = np.arange(1, 6)
        E_anal_4th = (n_vals * np.pi / a)**4
        
        print("\nPROBLEM 4. REDA (N=200):")
        print("n\tAnalitično\tNumerično\tNapaka (%)")
        for i in range(5):
            if i < len(eigvals):
                err = abs(eigvals[i] - E_anal_4th[i]) / E_anal_4th[i] * 100
                print(f"{i+1}\t{E_anal_4th[i]:.2f}\t{eigvals[i]:.2f}\t{err:.2f}")
    
    def generate_all_plots(self):
        print("\n" + "="*60)
        print("GENERIRANJE GRAFOV")
        print("="*60)
        
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'figure.autolayout': True,
            'savefig.dpi': 150,
            'savefig.bbox': 'tight'
        })
        
        try:
            self.plot_wavefunction_comparison()
            self.plot_finite_well()
            self.plot_convergence_study()
            self.plot_shooting_demo()
            self.plot_energy_spectra()
            self.plot_fourth_order()
            self.plot_error_analysis()
            self.plot_method_comparison()
            
            self.generate_report_data()
            
            print("\n" + "="*60)
            print("USPEŠNO GENERIRANIH 8 GRAFOV:")
            print("="*60)
            print("1. wavefunction_comparison.png")
            print("2. finite_well_wavefunctions.png")
            print("3. convergence_analysis.png")
            print("4. shooting_method_demo.png")
            print("5. energy_spectrum_comparison.png")
            print("6. fourth_order_problem.png")
            print("7. error_analysis.png")
            print("8. method_comparison.png")
            print("="*60)
            
        except Exception as e:
            print(f"\nNapaka: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    solver = QuantumWellSolver(a=1.0, V0=100.0, N=200)
    solver.generate_all_plots()
