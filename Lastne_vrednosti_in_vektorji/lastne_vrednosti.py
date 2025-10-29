import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.special import factorial, hermite
import time
import os

# Nastavitev LaTeX pisave za grafe
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 1.2,  
    'lines.markersize': 4,
    'grid.linewidth': 0.3,  
})

class AnharmonicOscillator:
    def __init__(self):
        # Ustvari mapo za shranjevanje grafov
        self.output_dir = "anharmonic_oscillator_plots"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def construct_H0(self, N):
        """Konstruira nemoteni Hamiltonov operator H0."""
        return np.diag(np.arange(N) + 0.5)
    
    def construct_q_matrix(self, N):
        """Konstruira matriko koordinate q."""
        q = np.zeros((N, N))
        for i in range(N-1):
            q[i, i+1] = 0.5 * np.sqrt(i+1)
            q[i+1, i] = 0.5 * np.sqrt(i+1)
        return q
    
    def construct_q2_matrix(self, N):
        """Konstruira matriko q^2."""
        q2 = np.zeros((N, N))
        for i in range(N):
            q2[i, i] = 0.5 * (2*i + 1)
            if i + 2 < N:
                q2[i, i+2] = 0.5 * np.sqrt((i+1)*(i+2))
            if i - 2 >= 0:
                q2[i, i-2] = 0.5 * np.sqrt(i*(i-1))
        return q2
    
    def construct_q4_matrix_analytic(self, N):
        """Analitična konstrukcija matrike q^4 - najhitrejša metoda."""
        q4 = np.zeros((N, N))
        
        for j in range(N):
            # Diagonalni elementi
            if j < N:
                q4[j, j] = 0.25 * 3 * (2*j**2 + 2*j + 1)
            
            # Elementi |i-j| = 2
            if j + 2 < N:
                q4[j, j+2] = 0.25 * 2 * np.sqrt((j+1)*(j+2)*(2*j+3))
                q4[j+2, j] = q4[j, j+2]
            
            # Elementi |i-j| = 4  
            if j + 4 < N:
                q4[j, j+4] = 0.25 * np.sqrt((j+1)*(j+2)*(j+3)*(j+4))
                q4[j+4, j] = q4[j, j+4]
        
        return q4
    
    def construct_q4_matrix_recursive(self, N):
        """Konstruira q^4 z uporabo rekurzivnih relacij za Hermitove polinome."""
        q4 = np.zeros((N, N))
        
        for i in range(N):
            for j in range(max(0, i-4), min(N, i+5)):
                prefactor = 1.0 / 16.0 * np.sqrt(2**j * factorial(j) / (2**i * factorial(i)))
                
                if i == j + 4:
                    q4[i, j] = prefactor * 1.0
                elif i == j + 2:
                    q4[i, j] = prefactor * 4.0 * (2*j + 3)
                elif i == j:
                    q4[i, j] = prefactor * 12.0 * (2*j**2 + 2*j + 1)
                elif i == j - 2:
                    q4[i, j] = prefactor * 16.0 * j * (2*j**2 - 3*j + 1)
                elif i == j - 4:
                    q4[i, j] = prefactor * 16.0 * j * (j**3 - 6*j**2 + 11*j - 6)
        
        return q4
    
    def harmonic_oscillator_wf(self, n, q):
        """Valovna funkcija harmonskega oscilatorja."""
        return (1/np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))) * np.exp(-q**2/2) * hermite(n)(q)
    
    def calculate_energies(self, N, lambda_val, method='analytic'):
        """Izračuna lastne energije in vektorje."""
        H0 = self.construct_H0(N)
        
        if method == 'power':
            q = self.construct_q_matrix(N)
            q4 = q @ q @ q @ q
        elif method == 'q2_square':
            q2 = self.construct_q2_matrix(N)
            q4 = q2 @ q2
        elif method == 'recursive':
            q4 = self.construct_q4_matrix_recursive(N)
        else:  # 'analytic'
            q4 = self.construct_q4_matrix_analytic(N)
        
        H = H0 + lambda_val * q4
        eigenvalues, eigenvectors = eigh(H)
        return eigenvalues, eigenvectors
    
    def perturbative_correction(self, n, lambda_val, order=2):
        """Perturbacijski izračun do drugega reda."""
        E0 = n + 0.5
        
        if order >= 1:
            # Prvi red
            E1 = 0.75 * (2*n**2 + 2*n + 1)
        else:
            E1 = 0
        
        if order >= 2:
            # Drugi red
            E2 = - (2*n + 1) * (21 + 17*n + 17*n**2) / 8
        else:
            E2 = 0
        
        return E0 + lambda_val * E1 + lambda_val**2 * E2
    
    def save_plot(self, filename, dpi=300, bbox_inches='tight'):
        """Shrani trenutni graf v PDF."""
        full_path = os.path.join(self.output_dir, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches=bbox_inches, format='pdf')
        print(f"Graf shranjen: {full_path}")

def plot_comparison_study():
    """Primerjava različnih metod izračuna."""
    osc = AnharmonicOscillator()
    N = 50
    lambda_val = 1.0
    
    methods = ['analytic', 'q2_square', 'power', 'recursive']
    method_names = ['Analitična', '$(q^2)^2$', '$q^4$', 'Rekurzivna']
    colors = ['blue', 'red', 'green', 'orange']
    line_styles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(12, 10))
    
    # Hitrost izvajanja
    times = []
    energies_list = []
    
    for method in methods:
        start_time = time.time()
        eigenvalues, _ = osc.calculate_energies(N, lambda_val, method=method)
        end_time = time.time()
        
        times.append(end_time - start_time)
        energies_list.append(eigenvalues[:10])
        
        print(f"Metoda {method}: {end_time - start_time:.4f} s")
    
    # Prikaz hitrosti
    plt.subplot(2, 2, 1)
    bars = plt.bar(method_names, times, color=colors, alpha=0.7, linewidth=0.5, edgecolor='black')
    plt.ylabel('Čas izvajanja (s)')
    plt.title('Hitrost različnih metod ($N=50$)')
    
    # Dodaj vrednosti na stolpce
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{time_val:.3f}s', ha='center', va='bottom', fontsize=10)
    
    # Primerjava energij
    plt.subplot(2, 2, 2)
    for i, (energies, name, color, ls) in enumerate(zip(energies_list, method_names, colors, line_styles)):
        plt.plot(range(len(energies)), energies, marker='o', color=color, 
                linestyle=ls, linewidth=1.0, markersize=3, label=name, alpha=0.8)
    
    plt.xlabel('Indeks energije')
    plt.ylabel('Energija $E_n$')
    plt.title('Primerjava energij ($\\lambda=1.0$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Razlike med metodami
    plt.subplot(2, 2, 3)
    analytic_energies = energies_list[0]
    for i, (energies, name, color, ls) in enumerate(zip(energies_list[1:], method_names[1:], colors[1:], line_styles[1:])):
        differences = np.abs(energies - analytic_energies[:len(energies)])
        plt.semilogy(range(len(differences)), differences, marker='o', color=color, 
                    linestyle=ls, linewidth=1.0, markersize=3, label=name)
    
    plt.xlabel('Indeks energije')
    plt.ylabel('Absolutna razlika')
    plt.title('Razlike glede na analitično metodo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Konvergenca z N
    plt.subplot(2, 2, 4)
    N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    markers = ['o', 's', '^', 'D']
    for state_idx, state in enumerate([0, 1, 2]):
        energies_vs_N = []
        for N_val in N_values:
            eigenvalues, _ = osc.calculate_energies(N_val, lambda_val, 'analytic')
            if state < len(eigenvalues):
                energies_vs_N.append(eigenvalues[state])
        
        plt.plot(N_values[:len(energies_vs_N)], energies_vs_N, 
                marker=markers[state_idx], linestyle='-', linewidth=1.0, 
                markersize=3, label=f'$E_{{{state}}}$')
    
    plt.xlabel('Velikost matrike $N$')
    plt.ylabel('Energija $E_n$')
    plt.title('Konvergenca energij z $N$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    osc.save_plot("01_primerjava_metod.pdf")
    plt.show()

def plot_lambda_dependence():
    """Analiza odvisnosti od parametra λ."""
    osc = AnharmonicOscillator()
    N = 100  # Velika matrika za natančne rezultate
    
    lambda_values = np.linspace(0, 1, 21)
    
    plt.figure(figsize=(12, 10))
    
    # Izračun energij
    all_energies = []
    for lambd in lambda_values:
        eigenvalues, _ = osc.calculate_energies(N, lambd, 'analytic')
        all_energies.append(eigenvalues[:8])
    
    all_energies = np.array(all_energies)
    
    # Energije v odvisnosti od λ
    plt.subplot(2, 2, 1)
    markers = ['o', 's', '^', 'D', 'v', '<']
    for i in range(6):
        plt.plot(lambda_values, all_energies[:, i], 
                marker=markers[i % len(markers)], linestyle='-', linewidth=1.0,
                markersize=3, label=f'$E_{{{i}}}$')
    
    plt.xlabel('Anharmonski parameter $\\lambda$')
    plt.ylabel('Energija $E_n$')
    plt.title('Energijski nivoji v odvisnosti od $\\lambda$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Odstopanje od harmonskega oscilatorja
    plt.subplot(2, 2, 2)
    for i in range(4):
        harmonic_energies = i + 0.5
        deviations = all_energies[:, i] - harmonic_energies
        plt.plot(lambda_values, deviations, 
                marker=markers[i % len(markers)], linestyle='-', linewidth=1.0,
                markersize=3, label=f'$\\Delta E_{{{i}}}$')
    
    plt.xlabel('Anharmonski parameter $\\lambda$')
    plt.ylabel('Odstopanje od harmonskega oscilatorja')
    plt.title('Vpliv anharmonske motnje')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Relativno odstopanje
    plt.subplot(2, 2, 3)
    for i in range(4):
        harmonic_energies = i + 0.5
        relative_deviations = (all_energies[:, i] - harmonic_energies) / harmonic_energies * 100
        plt.plot(lambda_values, relative_deviations, 
                marker=markers[i % len(markers)], linestyle='-', linewidth=1.0,
                markersize=3, label=f'$E_{{{i}}}$')
    
    plt.xlabel('Anharmonski parameter $\\lambda$')
    plt.ylabel('Relativno odstopanje ($\\%$)')
    plt.title('Relativni vpliv motnje')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Primerjava s perturbacijsko teorijo
    plt.subplot(2, 2, 4)
    line_styles = ['-', '--', '-.']
    for i in range(3):
        # Numerične vrednosti
        plt.plot(lambda_values, all_energies[:, i], 
                marker=markers[i % len(markers)], linestyle=line_styles[0], linewidth=1.2,
                markersize=3, label=f'Numerično $E_{{{i}}}$')
        
        # Perturbacijska teorija
        perturbative = [osc.perturbative_correction(i, lambd, order=2) for lambd in lambda_values]
        plt.plot(lambda_values, perturbative, 
                linestyle=line_styles[1], linewidth=1.0,
                label=f'Perturbacijsko $E_{{{i}}}$')
    
    plt.xlabel('Anharmonski parameter $\\lambda$')
    plt.ylabel('Energija $E_n$')
    plt.title('Primerjava s perturbacijsko teorijo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    osc.save_plot("02_odvisnost_od_lambda.pdf")
    plt.show()

def plot_wavefunctions_comprehensive():
    """Celovita analiza lastnih funkcij."""
    osc = AnharmonicOscillator()
    
    # Koordinatna mreža
    q_range = np.linspace(-5, 5, 1000)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    lambda_values = [0, 0.1, 0.5, 1.0]
    colors = ['black', 'blue', 'green', 'red']
    line_styles = ['-', '--', '-.', ':']
    
    for state_idx in range(6):
        if state_idx >= len(axes):
            break
            
        ax = axes[state_idx]
        
        for lambd_idx, lambd in enumerate(lambda_values):
            # Izračun lastnih stanj
            eigenvalues, eigenvectors = osc.calculate_energies(100, lambd, 'analytic')
            
            # Rekonstrukcija valovne funkcije
            psi = np.zeros_like(q_range)
            for n in range(min(100, len(eigenvectors))):
                coeff = eigenvectors[n, state_idx]
                psi += coeff * osc.harmonic_oscillator_wf(n, q_range)
            
            # Normalizacija (približno)
            norm = np.sqrt(np.trapz(psi**2, q_range))
            if norm > 0:
                psi = psi / norm
            
            # Prikaz
            ax.plot(q_range, psi + lambd_idx * 2, 
                   color=colors[lambd_idx], 
                   linestyle=line_styles[lambd_idx % len(line_styles)],
                   linewidth=1.0,
                   label=f'$\\lambda={lambd}$' if state_idx == 0 else "")
        
        ax.set_xlabel('$q$')
        ax.set_ylabel(f'$\\psi_{{{state_idx}}}(q) + \\mathrm{{konst.}}$')
        ax.set_title(f'Stanje ${state_idx}$: $E = {eigenvalues[state_idx]:.4f}$')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-4, 4)
        
        if state_idx == 0:
            ax.legend(fontsize=10)
    
    plt.suptitle('Lastne valovne funkcije za različne $\\lambda$', fontsize=16)
    plt.tight_layout()
    osc.save_plot("03_valovne_funkcije.pdf")
    plt.show()

def analyze_convergence():
    """Podrobna analiza konvergence."""
    osc = AnharmonicOscillator()
    
    N_max = 200
    N_values = np.arange(10, N_max + 1, 10)
    lambda_values = [0.1, 0.5, 1.0]
    
    plt.figure(figsize=(12, 10))
    
    markers = ['o', 's', '^', 'D']
    line_styles = ['-', '--', '-.', ':']
    
    for lambd_idx, lambd in enumerate(lambda_values):
        plt.subplot(2, 2, lambd_idx + 1)
        
        # Referenčne vrednosti (z največjo N)
        eigenvalues_ref, _ = osc.calculate_energies(N_max, lambd, 'analytic')
        
        for state_idx, state in enumerate(range(4)):
            energies = []
            for N in N_values:
                eigenvalues, _ = osc.calculate_energies(N, lambd, 'analytic')
                if state < len(eigenvalues):
                    energies.append(eigenvalues[state])
            
            # Napaka glede na referenčno vrednost
            if len(energies) == len(N_values):
                errors = np.abs(energies - eigenvalues_ref[state])
                plt.loglog(N_values, errors, 
                          marker=markers[state_idx % len(markers)], 
                          linestyle=line_styles[state_idx % len(line_styles)],
                          linewidth=1.0, markersize=3, 
                          label=f'$E_{{{state}}}$')
        
        plt.xlabel('Velikost matrike $N$')
        plt.ylabel('Absolutna napaka')
        plt.title(f'Konvergenca napake za $\\lambda={lambd}$')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Analiza potrebne velikosti matrike
    plt.subplot(2, 2, 4)
    tolerance = 1e-6
    
    for lambd_idx, lambd in enumerate(lambda_values):
        eigenvalues_ref, _ = osc.calculate_energies(N_max, lambd, 'analytic')
        required_N = []
        
        for state in range(8):
            for N in range(10, N_max + 1, 5):
                eigenvalues, _ = osc.calculate_energies(N, lambd, 'analytic')
                if state < len(eigenvalues):
                    error = abs(eigenvalues[state] - eigenvalues_ref[state])
                    if error < tolerance:
                        required_N.append(N)
                        break
            else:
                required_N.append(N_max)
        
        plt.plot(range(len(required_N)), required_N, 
                marker=markers[lambd_idx % len(markers)], 
                linestyle=line_styles[lambd_idx % len(line_styles)],
                linewidth=1.0, markersize=3, 
                label=f'$\\lambda={lambd}$')
    
    plt.xlabel('Indeks energije $n$')
    plt.ylabel('Potrebna velikost $N$')
    plt.title(f'Potrebna $N$ za natančnost ${tolerance:.0e}$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    osc.save_plot("04_konvergenca.pdf")
    plt.show()

def double_well_potential():
    """Analiza potenciala z dvema minimumoma."""
    print("\n=== ANALIZA POTENCIALA Z DVEMA MINIMUMA ===")
    
    # Hamiltonian za potencial V(q) = -2q² + q⁴/10
    # To ustreza H = p²/2 - 2q² + q⁴/10
    
    N = 200
    q = np.linspace(-4, 4, 1000)
    V = -2*q**2 + q**4/10
    
    # Numerična rešitev
    osc = AnharmonicOscillator()
    
    # Konstruiramo Hamiltonovo matriko
    H0 = osc.construct_H0(N)
    q2 = osc.construct_q2_matrix(N)
    q4 = osc.construct_q4_matrix_analytic(N)
    
    # Hamiltonian: H = H0 - 2q² + 0.1q⁴
    H = H0 - 2*q2 + 0.1*q4
    
    eigenvalues, eigenvectors = eigh(H)
    
    plt.figure(figsize=(12, 8))
    
    # Potencial in energijski nivoji
    plt.subplot(1, 2, 1)
    plt.plot(q, V, 'k-', linewidth=1.5, label='$V(q) = -2q^2 + q^4/10$')
    
    # Prikaži prvih 10 energijskih nivojev
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(10):
        plt.axhline(y=eigenvalues[i], color=colors[i % len(colors)], 
                   linestyle='--', linewidth=1.0, alpha=0.7)
        plt.text(3.5, eigenvalues[i], f'$E_{{{i}}}$', va='center', ha='left', fontsize=10)
    
    plt.xlabel('$q$')
    plt.ylabel('Energija')
    plt.title('Potencial z dvema minimumoma in energijski nivoji')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-10, 2)
    
    # Valovne funkcije
    plt.subplot(1, 2, 2)
    q_range = np.linspace(-4, 4, 1000)
    
    for i in range(4):
        psi = np.zeros_like(q_range)
        for n in range(min(100, len(eigenvectors))):
            coeff = eigenvectors[n, i]
            psi += coeff * osc.harmonic_oscillator_wf(n, q_range)
        
        # Normalizacija
        norm = np.sqrt(np.trapz(psi**2, q_range))
        if norm > 0:
            psi = psi / norm
        
        plt.plot(q_range, psi + eigenvalues[i], 
                linewidth=1.2, label=f'$\\psi_{{{i}}}$, $E={eigenvalues[i]:.4f}$')
    
    plt.plot(q, V, 'k-', linewidth=1.5, alpha=0.5)
    plt.xlabel('$q$')
    plt.ylabel('Energija')
    plt.title('Prve valovne funkcije')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-10, 2)
    
    plt.tight_layout()
    osc.save_plot("05_potencial_dva_minimuma.pdf")
    plt.show()
    
    print("Prvih 10 energij za potencial z dvema minimumoma:")
    for i in range(10):
        print(f"E_{i} = {eigenvalues[i]:.8f} ħω")
    
    # Analiza kvazidegeneracije
    print("\nAnaliza kvazidegeneracije:")
    for i in range(0, 9, 2):
        if i+1 < len(eigenvalues):
            diff = eigenvalues[i+1] - eigenvalues[i]
            print(f"E_{i+1} - E_{i} = {diff:.2e} ħω")

def plot_energy_levels_comprehensive():
    """Celovit prikaz energijskih nivojev za različne λ."""
    osc = AnharmonicOscillator()
    N = 100
    
    lambda_values = [0, 0.01, 0.1, 0.5, 1.0]
    colors = ['black', 'blue', 'green', 'orange', 'red']
    markers = ['o', 's', '^', 'D', 'v']
    
    plt.figure(figsize=(10, 8))
    
    for lambd_idx, lambd in enumerate(lambda_values):
        eigenvalues, _ = osc.calculate_energies(N, lambd, 'analytic')
        
        # Prikaz prvih 10 energij
        n_levels = min(10, len(eigenvalues))
        levels = np.arange(n_levels)
        
        plt.plot(levels, eigenvalues[:n_levels], 
                marker=markers[lambd_idx % len(markers)], linestyle='-', 
                linewidth=1.0, markersize=4,
                color=colors[lambd_idx], label=f'$\\lambda = {lambd}$')
    
    plt.xlabel('Kvantno število $n$')
    plt.ylabel('Energija $E_n$ ($\\hbar\\omega$)')
    plt.title('Energijski nivoji anharmonskega oscilatorja')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    osc.save_plot("06_energijski_nivoji.pdf")
    plt.show()

def plot_matrix_structure():
    """Prikaz strukture matrik."""
    osc = AnharmonicOscillator()
    N = 20
    
    # Ustvarimo različne matrike
    q = osc.construct_q_matrix(N)
    q2 = osc.construct_q2_matrix(N)
    q4_analytic = osc.construct_q4_matrix_analytic(N)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Matrika q
    im0 = axes[0, 0].imshow(np.abs(q), cmap='viridis', aspect='equal')
    axes[0, 0].set_title('Matrika $|q|$')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Matrika q²
    im1 = axes[0, 1].imshow(np.abs(q2), cmap='viridis', aspect='equal')
    axes[0, 1].set_title('Matrika $|q^2|$')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Matrika q⁴
    im2 = axes[1, 0].imshow(np.abs(q4_analytic), cmap='viridis', aspect='equal')
    axes[1, 0].set_title('Matrika $|q^4|$ (analitična)')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Primerjava q⁴ metod
    q4_power = osc.construct_q_matrix(N)
    q4_power = q4_power @ q4_power @ q4_power @ q4_power
    
    difference = np.abs(q4_analytic - q4_power)
    im3 = axes[1, 1].imshow(difference, cmap='hot', aspect='equal')
    axes[1, 1].set_title('Razlika $|q^4_{\\mathrm{anal}} - q^4_{\\mathrm{pot}}|$')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    osc.save_plot("07_struktura_matrik.pdf")
    plt.show()

def generate_summary_table():
    """Ustvari povzetke rezultatov v tabelah."""
    osc = AnharmonicOscillator()
    
    print("\n=== POVZETEK REZULTATOV ===")
    
    # Tabela za λ=0.5
    N = 100
    lambda_val = 0.5
    eigenvalues, _ = osc.calculate_energies(N, lambda_val, 'analytic')
    
    print(f"\nEnergije za $\\lambda = {lambda_val}$:")
    print("n\tE_n\t\tΔE_n\t\tPerturbacijski")
    print("-" * 50)
    for i in range(5):
        E_harmonic = i + 0.5
        E_anharmonic = eigenvalues[i]
        delta = E_anharmonic - E_harmonic
        perturb = osc.perturbative_correction(i, lambda_val, 2)
        print(f"{i}\t{E_anharmonic:.6f}\t{delta:.6f}\t{perturb:.6f}")

def main():
    """Glavna funkcija za izvedbo celotne analize."""
    
    print("=== CELOVITA ANALIZA ANHARMONSKEGA OSCILATORJA ===")
    print(f"Grafi se shranjujejo v mapo: anharmonic_oscillator_plots/")
    
    # 1. Primerjava metod
    print("\n1. Primerjava metod izračuna...")
    plot_comparison_study()
    
    # 2. Odvisnost od λ
    print("\n2. Analiza odvisnosti od anharmonskega parametra...")
    plot_lambda_dependence()
    
    # 3. Valovne funkcije
    print("\n3. Analiza lastnih valovnih funkcij...")
    plot_wavefunctions_comprehensive()
    
    # 4. Konvergenca
    print("\n4. Analiza konvergence...")
    analyze_convergence()
    
    # 5. Potencial z dvema minimumoma
    print("\n5. Analiza potenciala z dvema minimumoma...")
    double_well_potential()
    
    # 6. Energijski nivoji
    print("\n6. Prikaz energijskih nivojev...")
    plot_energy_levels_comprehensive()
    
    # 7. Struktura matrik
    print("\n7. Analiza strukture matrik...")
    plot_matrix_structure()
    
    # 8. Povzetek
    print("\n8. Ustvarjanje povzetka...")
    generate_summary_table()
    
    print(f"\n=== ANALIZA ZAKLJUČENA ===")
    print(f"Vse grafe najdete v mapi: anharmonic_oscillator_plots/")
    print("\nSeznam shranjenih grafov:")
    print("01_primerjava_metod.pdf - Primerjava metod izračuna")
    print("02_odvisnost_od_lambda.pdf - Odvisnost energij od λ")
    print("03_valovne_funkcije.pdf - Lastne valovne funkcije")
    print("04_konvergenca.pdf - Analiza konvergence")
    print("05_potencial_dva_minimuma.pdf - Potencial z dvema minimumoma")
    print("06_energijski_nivoji.pdf - Energijski nivoji")
    print("07_struktura_matrik.pdf - Struktura matrik")
    
    print(f"\nLaTeX poročilo lahko zdaj vključi vseh 7 grafov z LaTeX pisavo!")

if __name__ == "__main__":
    main()
