import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, dst, idst
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time

# ============================================================================
# Fourierova metoda - POPRAVLJENA S STABILNO SHEMO
# ============================================================================

def solve_fourier_diffusion(L=10.0, N=256, D=1.0, sigma=1.0, t_end=1.0, dt=0.001, bc='periodic'):
    """
    Reševanje difuzijske enačbe s Fourierovo metodo.
    Uporabi implicitno shemo za stabilnost.
    """
    start_time = time.time()
    
    if bc == 'periodic':
        x = np.linspace(0, L, N, endpoint=False)
        # Začetni pogoj (Gaussova porazdelitev)
        T0 = np.exp(-(x - L/2)**2 / sigma**2)
        T0 = T0 / np.max(T0)
        
        # Fourierova transformacija
        Tk = fft(T0)
        k = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
        
        # IMPLICITNA EULERJEVA SHEMA za stabilnost
        # T_k^{n+1} = T_k^n / (1 + D*dt*k^2)
        num_steps = int(t_end / dt)
        T_history = [T0.copy()]
        
        # Preveri, da bo shema stabilna
        stability_check = np.all(np.abs(1 / (1 + D * dt * k**2)) <= 1)
        if not stability_check:
            print(f"  Opozorilo: Shema bi lahko bila nestabilna!")
        
        for step in range(num_steps):
            Tk = Tk / (1 + D * dt * k**2)
            T = np.real(ifft(Tk))
            T_history.append(T.copy())
            
    elif bc == 'dirichlet':
        # Za Dirichlet uporabimo sinusno transformacijo
        x = np.linspace(0, L, N+2)[1:-1]  # Notranje točke (brez robov)
        N_int = len(x)
        
        # Začetni pogoj
        T0 = np.exp(-(x - L/2)**2 / sigma**2)
        T0 = T0 / np.max(T0)
        
        # Sinusna transformacija (DST tip I)
        Tk = dst(T0, type=1)
        
        # Lastne vrednosti za Laplaceov operator z Dirichletovimi pogoji
        k = np.pi * np.arange(1, N_int+1) / L
        eigenvalues = -k**2
        
        # Implicitna shema
        num_steps = int(t_end / dt)
        T_history = [T0.copy()]
        
        for step in range(num_steps):
            Tk = Tk / (1 - D * dt * eigenvalues)  # eigenvalues so negativni
            T = idst(Tk, type=1) / (2 * (N_int + 1))
            T_history.append(T.copy())
    
    else:
        raise ValueError("bc mora biti 'periodic' ali 'dirichlet'")
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    return x, np.array(T_history), computation_time

# ============================================================================
# Kolokacijska metoda s kubičnimi B-zlepki
# ============================================================================

def cubic_B_spline(x, xk, dx):
    """Vrednost kubičnega B-zlepka s središčem v xk."""
    r = (x - xk) / dx
    
    if r < -2 or r >= 2:
        return 0.0
    elif r < -1:
        t = r + 2
        return t**3 / 6.0
    elif r < 0:
        return (2/3) - r**2 * (1 + r/2)
    elif r < 1:
        return (2/3) - r**2 * (1 - r/2)
    else:  # 1 <= r < 2
        t = 2 - r
        return t**3 / 6.0

def solve_collocation_diffusion(L=10.0, N=50, D=1.0, sigma=1.0, t_end=1.0, dt=0.001):
    """
    Reševanje difuzijske enačbe s kolokacijsko metodo s kubičnimi B-zlepki.
    """
    start_time = time.time()
    
    dx = L / N
    x_nodes = np.linspace(0, L, N+1)
    x_int = x_nodes[1:-1]  # Notranje točke
    N_int = len(x_int)
    
    # Konstrukcija matrik A in B
    main_diag_A = 4 * np.ones(N_int)
    off_diag_A = np.ones(N_int - 1)
    
    A = diags([off_diag_A, main_diag_A, off_diag_A], [-1, 0, 1], format='csr')
    
    # Matrika B za difuzijski operator
    main_diag_B = -2 * np.ones(N_int)
    off_diag_B = np.ones(N_int - 1)
    B = (6*D / dx**2) * diags([off_diag_B, main_diag_B, off_diag_B], [-1, 0, 1], format='csr')
    
    # Začetni pogoj
    g = np.exp(-(x_int - L/2)**2 / sigma**2)
    g = g / np.max(g)
    
    # Reševanje začetnega sistema A a0 = 6g
    a = spsolve(A, 6 * g)
    
    # Crank-Nicolson shema (implicitna, stabilna)
    I = diags([np.ones(N_int)], [0], format='csr')
    lhs_matrix = A - (dt/2) * B
    rhs_matrix = A + (dt/2) * B
    
    # LU dekompozicija za hitrejše reševanje
    from scipy.sparse.linalg import splu
    lu = splu(lhs_matrix.tocsc())
    
    num_steps = int(t_end / dt)
    T_history = []
    
    for step in range(num_steps):
        rhs = rhs_matrix @ a
        a = lu.solve(rhs)
        
        # Rekonstrukcija temperature
        T = np.zeros(N+1)
        # Notranje točke
        for j in range(1, N):
            xj = x_nodes[j]
            T_sum = 0
            # Vsaka točka vpliva na max 4 zlepke
            for k_idx in range(max(0, j-2), min(N_int, j+2)):
                # Korekcija indeksa za zlepke
                actual_k = k_idx  # a je indeksiran 0..N_int-1 za notranje točke
                xk = (actual_k + 1) * dx  # Zlepki so centrirani na vozliščih
                T_sum += a[k_idx] * cubic_B_spline(xj, xk, dx)
            T[j] = T_sum
        # Robni pogoji (homogeni Dirichlet)
        T[0] = 0
        T[-1] = 0
        T_history.append(T.copy())
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    return x_nodes, np.array(T_history), computation_time

# ============================================================================
# Analiza stabilnosti - POPRAVLJENA
# ============================================================================

def analyze_stability():
    """Analiza stabilnosti Eulerjevih shem."""
    print("\n=== ANALIZA STABILNOSTI ===")
    
    D = 1.0
    L = 10.0
    dt_values = [0.001, 0.005, 0.01, 0.05]
    N_values = [32, 64, 128]
    
    plt.figure(figsize=(14, 10))
    
    for i, dt in enumerate(dt_values):
        stability_ratios_exp = []  # Eksplicitna
        stability_ratios_imp = []  # Implicitna
        
        for N in N_values:
            # Najvišja frekvenca
            k_max = np.pi * N / L
            
            # Eksplicitna Eulerjeva shema
            stab_exp = np.abs(1 - D * dt * k_max**2)
            stability_ratios_exp.append(stab_exp)
            
            # Implicitna Eulerjeva shema
            stab_imp = np.abs(1 / (1 + D * dt * k_max**2))
            stability_ratios_imp.append(stab_imp)
            
            print(f"dt={dt:.4f}, N={N}:")
            print(f"  Eksplicitna: |1 - D*dt*k_max²| = {stab_exp:.3f}", 
                  end="")
            if stab_exp > 1:
                print("  ⚠️ Nestabilno!")
            else:
                print("  ✓ Stabilno")
                
            print(f"  Implicitna:  |1/(1 + D*dt*k_max²)| = {stab_imp:.3f} ✓ Vedno stabilno")
        
        plt.subplot(2, 2, i+1)
        plt.plot(N_values, stability_ratios_exp, 'ro-', label='Eksplicitna Euler', 
                 linewidth=2, markersize=8)
        plt.plot(N_values, stability_ratios_imp, 'bo-', label='Implicitna Euler', 
                 linewidth=2, markersize=8)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Meja stabilnosti')
        plt.xlabel('Število točk (N)', fontsize=10)
        plt.ylabel('Faktor stabilnosti', fontsize=10)
        plt.title(f'Stabilnost za dt = {dt}', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        if max(stability_ratios_exp) > 5:
            plt.ylim([0, min(5, max(stability_ratios_imp)*2)])
    
    plt.tight_layout()
    plt.savefig('stabilnost_analiza.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# Konvergenčna analiza - POPRAVLJENA
# ============================================================================

def convergence_analysis():
    """Analiza konvergence obeh metod."""
    print("\n=== ANALIZA KONVERGENCE ===")
    
    L = 10.0
    D = 1.0
    sigma = 1.0
    t_end = 0.1
    dt = 0.0005  # Manjši dt za stabilnost
    
    # Analitična rešitej za Gaussov začetni pogoj (približek za majhne čase)
    def analytical_solution(x, t, L, D, sigma):
        # Za periodične pogoje: vsota Gaussovih funkcij
        T = np.zeros_like(x)
        # Za majhen čas uporabimo samo glavno komponento
        return np.exp(-(x - L/2)**2 / (4*D*t + sigma**2)) / np.sqrt(4*np.pi*D*t + sigma**2)
    
    # Test različnih N
    N_values_fourier = [32, 64, 128, 256]
    N_values_colloc = [20, 40, 60, 80]  # Manjše vrednosti za kolokacijo
    
    errors_fourier = []
    errors_colloc = []
    times_fourier = []
    times_colloc = []
    
    for i, N_f in enumerate(N_values_fourier):
        print(f"\nAnaliza za Fourier N = {N_f}...")
        
        # Fourierova metoda
        x_f, T_f, time_f = solve_fourier_diffusion(L, N_f, D, sigma, t_end, dt, 'periodic')
        T_analytical = analytical_solution(x_f, t_end, L, D, sigma)
        T_analytical = T_analytical / np.max(T_analytical)  # Normalizacija
        
        if len(T_f) > 0:
            error_f = np.sqrt(np.mean((T_f[-1] - T_analytical)**2))
            errors_fourier.append(error_f)
            times_fourier.append(time_f)
            print(f"  Napaka: {error_f:.2e}, čas: {time_f:.3f}s")
    
    for i, N_c in enumerate(N_values_colloc):
        print(f"\nAnaliza za Kolokacija N = {N_c}...")
        
        # Kolokacijska metoda
        x_c, T_c, time_c = solve_collocation_diffusion(L, N_c, D, sigma, t_end, dt)
        
        if len(T_c) > 0:
            # Interpolacija na Fourierjeve točke za primerjavo
            from scipy.interpolate import interp1d
            # Uporabi prvo Fourierjevo mrežo za referenco
            x_ref = np.linspace(0, L, N_values_fourier[0], endpoint=False)
            T_c_interp = interp1d(x_c, T_c[-1], kind='cubic', bounds_error=False, fill_value=0)(x_ref)
            T_analytical_ref = analytical_solution(x_ref, t_end, L, D, sigma)
            T_analytical_ref = T_analytical_ref / np.max(T_analytical_ref)
            
            error_c = np.sqrt(np.mean((T_c_interp - T_analytical_ref)**2))
            errors_colloc.append(error_c)
            times_colloc.append(time_c)
            print(f"  Napaka: {error_c:.2e}, čas: {time_c:.3f}s")
    
    # Vizualizacija konvergence - POPRAVLJENA (dimenzije se ujemajo)
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    if errors_fourier:
        plt.loglog(N_values_fourier, errors_fourier, 'bo-', label='Fourierova metoda', 
                   linewidth=2, markersize=8)
        # Referenčna črta za konvergenco reda
        if len(N_values_fourier) >= 2:
            ref_x = [N_values_fourier[0], N_values_fourier[-1]]
            ref_y = [errors_fourier[0], errors_fourier[0]*(ref_x[0]/ref_x[-1])**2]
            plt.loglog(ref_x, ref_y, 'b--', label='O(N⁻²)', alpha=0.5)
    
    if errors_colloc:
        plt.loglog(N_values_colloc, errors_colloc, 'ro-', label='Kolokacijska metoda', 
                   linewidth=2, markersize=8)
    
    plt.xlabel('Število točk (N)', fontsize=12)
    plt.ylabel('RMSE napaka', fontsize=12)
    plt.title('Konvergenčna analiza', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    
    plt.subplot(1, 2, 2)
    if times_fourier and errors_fourier:
        plt.loglog(times_fourier, errors_fourier, 'bo-', label='Fourierova metoda', 
                   linewidth=2, markersize=8)
    if times_colloc and errors_colloc:
        plt.loglog(times_colloc, errors_colloc, 'ro-', label='Kolokacijska metoda', 
                   linewidth=2, markersize=8)
    
    plt.xlabel('Čas izvajanja [s]', fontsize=12)
    plt.ylabel('RMSE napaka', fontsize=12)
    plt.title('Učinkovitost metod', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('konvergencna_analiza.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Izpis rezultatov
    print("\n" + "="*50)
    print("REZULTATI KONVERGENČNE ANALIZE:")
    print("="*50)
    
    if errors_fourier:
        print("\nFourierova metoda:")
        for i, N in enumerate(N_values_fourier):
            print(f"  N = {N}: napaka = {errors_fourier[i]:.2e}, čas = {times_fourier[i]:.3f}s")
    
    if errors_colloc:
        print("\nKolokacijska metoda:")
        for i, N in enumerate(N_values_colloc):
            print(f"  N = {N}: napaka = {errors_colloc[i]:.2e}, čas = {times_colloc[i]:.3f}s")

# ============================================================================
# Glavna primerjava
# ============================================================================

def compare_methods():
    """
    Popolna primerjava Fourierove in kolokacijske metode.
    """
    print("\n" + "="*60)
    print("PRIMERJAVA FOURIEROVE IN KOLOKACIJSKE METODE")
    print("="*60)
    
    # Parametri z manjšim dt za stabilnost
    L = 10.0
    N_fourier = 128  # Manjša vrednost za stabilnost
    N_colloc = 50
    D = 1.0
    sigma = 1.0
    t_end = 0.2  # Krajši čas
    dt = 0.0005  # Manjši dt
    
    print(f"\nParametri: L={L}, D={D}, σ={sigma}, t_end={t_end}, dt={dt}")
    print(f"N_fourier={N_fourier}, N_colloc={N_colloc}")
    
    print("\n1. FOURIEROVA METODA (Periodični robni pogoji)")
    x_f_per, T_f_per, time_f_per = solve_fourier_diffusion(L, N_fourier, D, sigma, t_end, dt, 'periodic')
    print(f"  Čas izvajanja: {time_f_per:.3f}s")
    
    print("\n2. FOURIEROVA METODA (Dirichletovi robni pogoji)")
    x_f_dir, T_f_dir, time_f_dir = solve_fourier_diffusion(L, N_fourier, D, sigma, t_end, dt, 'dirichlet')
    print(f"  Čas izvajanja: {time_f_dir:.3f}s")
    
    print("\n3. KOLOKACIJSKA METODA (Implicitna)")
    x_c, T_c, time_c = solve_collocation_diffusion(L, N_colloc, D, sigma, t_end, dt)
    print(f"  Čas izvajanja: {time_c:.3f}s")
    
    # Vizualizacija
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Fourier - periodični
    ax = axes[0, 0]
    ax.plot(x_f_per, T_f_per[0], 'k--', label='Začetni pogoj', linewidth=2, alpha=0.7)
    ax.plot(x_f_per, T_f_per[-1], 'b-', label=f't = {t_end:.2f}', linewidth=2)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('T(x,t)', fontsize=11)
    ax.set_title('Fourier: Periodični robni pogoji', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Fourier - Dirichlet
    ax = axes[0, 1]
    ax.plot(x_f_dir, T_f_dir[0], 'k--', label='Začetni pogoj', linewidth=2, alpha=0.7)
    ax.plot(x_f_dir, T_f_dir[-1], 'g-', label=f't = {t_end:.2f}', linewidth=2)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('T(x,t)', fontsize=11)
    ax.set_title('Fourier: Homogeni Dirichlet', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Kolokacija
    ax = axes[0, 2]
    ax.plot(x_c, T_c[0], 'k--', label='Začetni pogoj', linewidth=2, alpha=0.7)
    ax.plot(x_c, T_c[-1], 'r-', label=f't = {t_end:.2f}', linewidth=2)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('T(x,t)', fontsize=11)
    ax.set_title('Kolokacija s kubičnimi B-zlepki', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Časovni razvoj - izbrane točke
    ax = axes[1, 0]
    if len(T_f_per) > 10:
        time_points = np.linspace(0, t_end, len(T_f_per))
        for idx, x_pos in enumerate([L/4, L/2, 3*L/4]):
            x_idx = min(int(x_pos / L * len(x_f_per)), len(x_f_per)-1)
            ax.plot(time_points, T_f_per[:, x_idx], label=f'x = {x_pos:.1f}', linewidth=2)
        ax.set_xlabel('Čas (t)', fontsize=11)
        ax.set_ylabel('T(x,t)', fontsize=11)
        ax.set_title('Fourier: Časovni razvoj (periodični)', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    if len(T_f_dir) > 10:
        time_points = np.linspace(0, t_end, len(T_f_dir))
        for idx, x_pos in enumerate([L/4, L/2, 3*L/4]):
            x_idx = min(int(x_pos / L * len(x_f_dir)), len(x_f_dir)-1)
            ax.plot(time_points, T_f_dir[:, x_idx], label=f'x = {x_pos:.1f}', linewidth=2)
        ax.set_xlabel('Čas (t)', fontsize=11)
        ax.set_ylabel('T(x,t)', fontsize=11)
        ax.set_title('Fourier: Časovni razvoj (Dirichlet)', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    ax = axes[1, 2]
    if len(T_c) > 10:
        time_points = np.linspace(0, t_end, len(T_c))
        for idx, x_pos in enumerate([L/4, L/2, 3*L/4]):
            x_idx = min(int(x_pos / L * len(x_c)), len(x_c)-1)
            ax.plot(time_points, T_c[:, x_idx], label=f'x = {x_pos:.1f}', linewidth=2)
        ax.set_xlabel('Čas (t)', fontsize=11)
        ax.set_ylabel('T(x,t)', fontsize=11)
        ax.set_title('Kolokacija: Časovni razvoj', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('primerjava_metod_detajlno.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analiza konzervativnosti
    print("\n" + "="*50)
    print("ANALIZA KONZERVATIVNOSTI IN UČINKOVITOSTI")
    print("="*50)
    
    results = []
    if len(T_f_per) > 0:
        results.append(("Fourier (periodični)", T_f_per, x_f_per, time_f_per))
    if len(T_f_dir) > 0:
        results.append(("Fourier (Dirichlet)", T_f_dir, x_f_dir, time_f_dir))
    if len(T_c) > 0:
        results.append(("Kolokacija", T_c, x_c, time_c))
    
    for name, T_hist, x_arr, comp_time in results:
        if len(T_hist) > 0 and len(x_arr) > 1:
            # Izračun energije (L2 norma)
            if len(T_hist[0]) == len(x_arr):
                E_initial = np.trapz(T_hist[0]**2, x_arr)
                E_final = np.trapz(T_hist[-1]**2, x_arr)
                
                print(f"\n{name}:")
                print(f"  Čas izračuna: {comp_time:.3f} s")
                print(f"  Začetna energija: {E_initial:.6f}")
                print(f"  Končna energija:  {E_final:.6f}")
                if E_initial > 0:
                    print(f"  Relativna sprememba:  {100*(E_final/E_initial-1):.3f}%")

# ============================================================================
# Test sinusnega začetnega pogoja
# ============================================================================

def test_sinus_initial():
    """
    Testiranje obeh metod s sinusnim začetnim pogojem.
    """
    print("\n" + "="*60)
    print("TESTIRANJE ZA SINUSNI ZAČETNI POGOJ")
    print("="*60)
    
    L = 10.0
    N_fourier = 128
    N_colloc = 50
    D = 1.0
    t_end = 0.2
    dt = 0.0005
    
    # Analitična rešitev za sinusni primer
    def analytical_solution_sin(x, t, L, D):
        return np.sin(np.pi * x / L) * np.exp(-D * (np.pi/L)**2 * t)
    
    # Fourierova metoda (ročno za sinusni primer)
    print("\nFourierova metoda za sinusni primer...")
    x_f = np.linspace(0, L, N_fourier+2)[1:-1]  # Notranje točke za Dirichlet
    T0_sin = np.sin(np.pi * x_f / L)
    
    # Transformacija in časovna integracija
    Tk = dst(T0_sin, type=1)
    k = np.pi * np.arange(1, len(x_f)+1) / L
    eigenvalues = -k**2
    
    num_steps = int(t_end / dt)
    for step in range(num_steps):
        Tk = Tk / (1 - D * dt * eigenvalues)  # Implicitna shema
    
    T_f = idst(Tk, type=1) / (2 * (len(x_f) + 1))
    
    # Kolokacijska metoda s sinusnim začetnim pogojem
    print("Kolokacijska metoda za sinusni primer...")
    L = 10.0
    N = N_colloc
    dx = L / N
    x_nodes = np.linspace(0, L, N+1)
    x_int = x_nodes[1:-1]
    N_int = len(x_int)
    
    # Matrike
    main_diag_A = 4 * np.ones(N_int)
    off_diag_A = np.ones(N_int - 1)
    A = diags([off_diag_A, main_diag_A, off_diag_A], [-1, 0, 1], format='csr')
    
    main_diag_B = -2 * np.ones(N_int)
    off_diag_B = np.ones(N_int - 1)
    B = (6*D / dx**2) * diags([off_diag_B, main_diag_B, off_diag_B], [-1, 0, 1], format='csr')
    
    # Sinusni začetni pogoj
    g = np.sin(np.pi * x_int / L)
    a = spsolve(A, 6 * g)
    
    # Crank-Nicolson
    lhs_matrix = A - (dt/2) * B
    rhs_matrix = A + (dt/2) * B
    from scipy.sparse.linalg import splu
    lu = splu(lhs_matrix.tocsc())
    
    num_steps = int(t_end / dt)
    for step in range(num_steps):
        rhs = rhs_matrix @ a
        a = lu.solve(rhs)
    
    # Rekonstrukcija
    T_c = np.zeros(N+1)
    for j in range(1, N):
        xj = x_nodes[j]
        T_sum = 0
        for k_idx in range(max(0, j-2), min(N_int, j+2)):
            xk = (k_idx + 1) * dx
            T_sum += a[k_idx] * cubic_B_spline(xj, xk, dx)
        T_c[j] = T_sum
    
    # Analitična rešitej
    T_analytic_f = analytical_solution_sin(x_f, t_end, L, D)
    T_analytic_c = analytical_solution_sin(x_nodes, t_end, L, D)
    
    # Vizualizacija
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_f, T0_sin, 'k--', label='Začetni pogoj', linewidth=2, alpha=0.7)
    plt.plot(x_f, T_f, 'b-', label='Fourier (numerično)', linewidth=2)
    plt.plot(x_f, T_analytic_f, 'g:', label='Analitično', linewidth=3, alpha=0.5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('T(x,t)', fontsize=12)
    plt.title(r'Fourierova metoda: $T(x,0) = \sin(\pi x/L)$', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_nodes, np.sin(np.pi * x_nodes / L), 'k--', label='Začetni pogoj', linewidth=2, alpha=0.7)
    plt.plot(x_nodes, T_c, 'r-', label='Kolokacija (numerično)', linewidth=2)
    plt.plot(x_nodes, T_analytic_c, 'g:', label='Analitično', linewidth=3, alpha=0.5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('T(x,t)', fontsize=12)
    plt.title(r'Kolokacijska metoda: $T(x,0) = \sin(\pi x/L)$', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sinusni_primer_detajlno.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Izračun napak
    error_fourier = np.sqrt(np.mean((T_f - T_analytic_f)**2))
    error_colloc = np.sqrt(np.mean((T_c[1:-1] - T_analytic_c[1:-1])**2))
    
    print("\n" + "="*50)
    print("REZULTATI ZA SINUSNI PRIMER:")
    print("="*50)
    print(f"\nNapake glede na analitično rešitev:")
    print(f"  Fourierova metoda (RMSE):  {error_fourier:.2e}")
    print(f"  Kolokacijska metoda (RMSE): {error_colloc:.2e}")

# ============================================================================
# Glavni program
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("9. NALOGA: Spektralne metode za začetne probleme PDE")
    print("Razširjena analiza z IMPLICITNIMI SHEMAMI za stabilnost")
    print("=" * 70)
    
    # Analiza stabilnosti
    analyze_stability()
    
    # Konvergenčna analiza (s popravki)
    try:
        convergence_analysis()
    except Exception as e:
        print(f"\nOpozorilo pri konvergenčni analizi: {e}")
        print("Nadaljujem z glavno primerjavo...")
    
    # Glavna primerjava metod
    compare_methods()
    
    # Test sinusnega začetnega pogoja
    test_sinus_initial()
    
    print("\n" + "=" * 70)
    print("Program uspešno zaključen.")
    print("Rezultati so shranjeni v datoteke:")
    print("  - primerjava_metod_detajlno.png")
    print("  - sinusni_primer_detajlno.png")
    print("  - stabilnost_analiza.png")
    print("  - konvergencna_analiza.png")
    print("=" * 70)
