import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import time
import os

# Ustvarimo mapo za shranjevanje grafov
output_dir = "grafi"
os.makedirs(output_dir, exist_ok=True)

def gaussian_initial(x, a, sigma, T0=1.0):
    """Gaussov začetni pogoj."""
    return T0 * np.exp(-(x - a/2)**2 / sigma**2)

def fourier_method_periodic(T0, a, D, dt, Nt):
    """Fourierjeva metoda s periodičnimi robnimi pogoji."""
    N = len(T0)
    dx = a / N
    
    k = np.fft.fftfreq(N, d=dx)
    Tk = fft(T0)
    
    factor = -4 * np.pi**2 * D * k**2
    
    T_history = [T0.copy()]
    Tk_current = Tk.copy()
    
    for n in range(1, Nt):
        Tk_current *= (1 + dt * factor)
        T_current = np.real(ifft(Tk_current))
        T_history.append(T_current.copy())
    
    return np.array(T_history)

def fourier_method_dirichlet(T0, a, D, dt, Nt):
    """Fourierjeva metoda z Dirichletovimi robnimi pogoji."""
    N = len(T0) - 2
    x_inner = np.linspace(0, a, N+2)[1:-1]
    dx = a / (N+1)
    
    n = np.arange(1, N+1)
    lambda_n = n * np.pi / a
    
    # Uporaba np.trapezoid namesto zastarelega np.trapz
    bn = np.zeros(N)
    for i in range(N):
        integrand = T0[1:-1] * np.sin(lambda_n[i] * x_inner)
        bn[i] = 2/a * np.trapezoid(integrand, x_inner)
    
    bn_history = [bn.copy()]
    bn_current = bn.copy()
    
    for n in range(1, Nt):
        bn_current *= np.exp(-D * lambda_n**2 * dt)
        bn_history.append(bn_current.copy())
    
    T_history = []
    for bn_t in bn_history:
        T = np.zeros(len(T0))
        T[0] = 0
        T[-1] = 0
        for i in range(N):
            T[1:-1] += bn_t[i] * np.sin(lambda_n[i] * x_inner)
        T_history.append(T)
    
    return np.array(T_history)

def collocation_method(T0, a, D, dt, Nt):
    """Kolokacijska metoda s kubičnimi B-zlepki."""
    N = len(T0) - 2
    dx = a / (N+1)
    
    diag_A = 4 * np.ones(N)
    subdiag_A = np.ones(N-1)
    A = csr_matrix(diags([subdiag_A, diag_A, subdiag_A], [-1, 0, 1], shape=(N, N)))
    
    diag_B = -2 * np.ones(N)
    subdiag_B = np.ones(N-1)
    B = (6*D/dx**2) * csr_matrix(diags([subdiag_B, diag_B, subdiag_B], [-1, 0, 1], shape=(N, N)))
    
    g = T0[1:-1]
    c0 = spsolve(A, 6*g)
    
    left_matrix = A - dt/2 * B
    right_matrix = A + dt/2 * B
    
    c_history = [c0.copy()]
    c_current = c0.copy()
    
    for n in range(1, Nt):
        c_current = spsolve(left_matrix, right_matrix.dot(c_current))
        c_history.append(c_current.copy())
    
    T_history = []
    for c_t in c_history:
        T = np.zeros(len(T0))
        T[0] = 0
        T[-1] = 0
        T[1:-1] = c_t
        T_history.append(T)
    
    return np.array(T_history)

# ===================================================================
# GRAF 1: Fourierova metoda s periodičnimi robnimi pogoji - VEČ ČASOV, DALJŠI INTERVAL
# ===================================================================
print("Graf 1: Fourierova metoda s periodičnimi robnimi pogoji...")

a = 1.0
D = 0.01
sigma = a/10
N = 128
dt = 0.001  # POVEČAN časovni korak za daljši interval
t_max = 0.2  # DALJŠI končni čas
Nt = int(t_max/dt) + 1

x = np.linspace(0, a, N)
T0 = gaussian_initial(x, a, sigma)

T_fourier_periodic = fourier_method_periodic(T0, a, D, dt, Nt)
times = np.linspace(0, t_max, Nt)

# VEČ ČASOVNIH TOČK za boljši prikaz razvoja
plt.figure(figsize=(14, 9))
time_indices = [0, Nt//50, Nt//25, Nt//15, Nt//10, Nt//6, Nt//4, Nt//3, Nt//2, 
                2*Nt//3, 3*Nt//4, 4*Nt//5, Nt-1]
colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

for idx, i in enumerate(time_indices):
    plt.plot(x, T_fourier_periodic[i], color=colors[idx], 
             label=f't = {times[i]:.4f} a²/D', linewidth=2, alpha=0.9)

plt.plot(x, T0, 'k--', label='Začetni pogoj (t=0)', linewidth=3, alpha=0.8)
plt.xlabel('x/a', fontsize=14)
plt.ylabel('T/T₀', fontsize=14)
plt.title(f'Fourierova metoda - periodični robni pogoji (t≤{t_max}a²/D)', fontsize=16)
plt.legend(fontsize=10, loc='upper right', ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/graf1_fourier_periodic.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf1_fourier_periodic.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# GRAF 2: Fourierova metoda z Dirichletovimi robnimi pogoji - VEČ ČASOV, DALJŠI INTERVAL
# ===================================================================
print("Graf 2: Fourierova metoda z Dirichletovimi robnimi pogoji...")

T_fourier_dirichlet = fourier_method_dirichlet(T0, a, D, dt, Nt)

# VEČ ČASOVNIH TOČK za boljši prikaz razvoja
plt.figure(figsize=(14, 9))
time_indices = [0, Nt//50, Nt//25, Nt//15, Nt//10, Nt//6, Nt//4, Nt//3, Nt//2, 
                2*Nt//3, 3*Nt//4, 4*Nt//5, Nt-1]
colors = plt.cm.plasma(np.linspace(0, 1, len(time_indices)))

for idx, i in enumerate(time_indices):
    plt.plot(x, T_fourier_dirichlet[i], color=colors[idx], 
             label=f't = {times[i]:.4f} a²/D', linewidth=2, alpha=0.9)

plt.plot(x, T0, 'k--', label='Začetni pogoj (t=0)', linewidth=3, alpha=0.8)
plt.xlabel('x/a', fontsize=14)
plt.ylabel('T/T₀', fontsize=14)
plt.title(f'Fourierova metoda - homogeni Dirichletovi pogoji (t≤{t_max}a²/D)', fontsize=16)
plt.legend(fontsize=10, loc='upper right', ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/graf2_fourier_dirichlet.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf2_fourier_dirichlet.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# GRAF 3: Kolokacijska metoda - VEČ ČASOV, DALJŠI INTERVAL
# ===================================================================
print("Graf 3: Kolokacijska metoda...")

T_colloc = collocation_method(T0, a, D, dt, Nt)

# VEČ ČASOVNIH TOČK za boljši prikaz razvoja
plt.figure(figsize=(14, 9))
time_indices = [0, Nt//50, Nt//25, Nt//15, Nt//10, Nt//6, Nt//4, Nt//3, Nt//2, 
                2*Nt//3, 3*Nt//4, 4*Nt//5, Nt-1]
colors = plt.cm.inferno(np.linspace(0, 1, len(time_indices)))

for idx, i in enumerate(time_indices):
    plt.plot(x, T_colloc[i], color=colors[idx], 
             label=f't = {times[i]:.4f} a²/D', linewidth=2, alpha=0.9)

plt.plot(x, T0, 'k--', label='Začetni pogoj (t=0)', linewidth=3, alpha=0.8)
plt.xlabel('x/a', fontsize=14)
plt.ylabel('T/T₀', fontsize=14)
plt.title(f'Kolokacijska metoda - kubični B-zlepki (t≤{t_max}a²/D)', fontsize=16)
plt.legend(fontsize=10, loc='upper right', ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/graf3_collocation.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf3_collocation.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# GRAF 9: Različni začetni pogoji - Delta funkcija - VEČ ČASOV, ŠE DALJŠI INTERVAL
# ===================================================================
print("Graf 9: Delta začetni pogoj z daljšim časom...")

def delta_initial(x, a, T0=1.0):
    sigma_delta = a/100
    return T0 * np.exp(-(x - a/2)**2 / sigma_delta**2) / (sigma_delta * np.sqrt(np.pi))

# ŠE DALJŠI čas za delta funkcijo
t_max_delta = 0.5  # POVEČANO na 0.5
Nt_delta = 150     # POVEČANO število korakov
dt_delta = t_max_delta / Nt_delta

T0_delta = delta_initial(x, a)
T_delta = fourier_method_dirichlet(T0_delta, a, D, dt_delta, Nt_delta)
times_delta = np.linspace(0, t_max_delta, Nt_delta)

# VEČ ČASOVNIH TOČK za boljši prikaz razvoja
plt.figure(figsize=(14, 9))
time_indices_delta = [0, Nt_delta//30, Nt_delta//20, Nt_delta//15, Nt_delta//10, 
                      Nt_delta//6, Nt_delta//4, Nt_delta//3, Nt_delta//2, 
                      2*Nt_delta//3, 3*Nt_delta//4, 4*Nt_delta//5, Nt_delta-1]
colors_delta = plt.cm.cool(np.linspace(0, 1, len(time_indices_delta)))

for idx, i in enumerate(time_indices_delta):
    plt.plot(x, T_delta[i], color=colors_delta[idx], 
             label=f't = {times_delta[i]:.4f} a²/D', linewidth=2, alpha=0.9)

plt.plot(x, T0_delta, 'k--', label='Začetni pogoj (t=0)', linewidth=3, alpha=0.8)
plt.xlabel('x/a', fontsize=14)
plt.ylabel('T/T₀', fontsize=14)
plt.title(f'Delta začetni pogoj - časovni razvoj (t≤{t_max_delta}a²/D)', fontsize=16)
plt.legend(fontsize=10, loc='upper right', ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/graf9_delta.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf9_delta.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# GRAF 10: Različni začetni pogoji - Stopničasta funkcija - VEČ ČASOV, ŠE DALJŠI INTERVAL
# ===================================================================
print("Graf 10: Stopničasti začetni pogoj z daljšim časom...")

def step_initial(x, a, T0=1.0):
    return np.where((x > a/4) & (x < 3*a/4), T0, 0)

# ŠE DALJŠI čas za stopničasto funkcijo
t_max_step = 0.6  # POVEČANO na 0.6
Nt_step = 150     # POVEČANO število korakov
dt_step = t_max_step / Nt_step

T0_step = step_initial(x, a)
T_step = fourier_method_dirichlet(T0_step, a, D, dt_step, Nt_step)
times_step = np.linspace(0, t_max_step, Nt_step)

# VEČ ČASOVNIH TOČK za boljši prikaz razvoja
plt.figure(figsize=(14, 9))
time_indices_step = [0, Nt_step//30, Nt_step//20, Nt_step//15, Nt_step//10, 
                     Nt_step//6, Nt_step//4, Nt_step//3, Nt_step//2, 
                     2*Nt_step//3, 3*Nt_step//4, 4*Nt_step//5, Nt_step-1]
colors_step = plt.cm.spring(np.linspace(0, 1, len(time_indices_step)))

for idx, i in enumerate(time_indices_step):
    plt.plot(x, T_step[i], color=colors_step[idx], 
             label=f't = {times_step[i]:.4f} a²/D', linewidth=2, alpha=0.9)

plt.plot(x, T0_step, 'k--', label='Začetni pogoj (t=0)', linewidth=3, alpha=0.8)
plt.xlabel('x/a', fontsize=14)
plt.ylabel('T/T₀', fontsize=14)
plt.title(f'Stopničasti začetni pogoj - časovni razvoj (t≤{t_max_step}a²/D)', fontsize=16)
plt.legend(fontsize=10, loc='upper right', ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/graf10_step.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf10_step.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# DODATEN GRAF 13: Kolokacijska metoda za delta funkcijo - VEČ ČASOV, DALJŠI INTERVAL
# ===================================================================
print("Dodatni graf 13: Kolokacijska metoda za delta funkcijo...")

# Uporabimo kolokacijsko metodo za delta funkcijo
T_colloc_delta = collocation_method(T0_delta, a, D, dt_delta, Nt_delta)

plt.figure(figsize=(14, 9))
time_indices_colloc_delta = [0, Nt_delta//30, Nt_delta//20, Nt_delta//15, Nt_delta//10,
                             Nt_delta//6, Nt_delta//4, Nt_delta//3, Nt_delta//2, 
                             2*Nt_delta//3, 3*Nt_delta//4, 4*Nt_delta//5, Nt_delta-1]
colors_colloc_delta = plt.cm.winter(np.linspace(0, 1, len(time_indices_colloc_delta)))

for idx, i in enumerate(time_indices_colloc_delta):
    plt.plot(x, T_colloc_delta[i], color=colors_colloc_delta[idx], 
             label=f't = {times_delta[i]:.4f} a²/D', linewidth=2, alpha=0.9)

plt.plot(x, T0_delta, 'k--', label='Začetni pogoj (t=0)', linewidth=3, alpha=0.8)
plt.xlabel('x/a', fontsize=14)
plt.ylabel('T/T₀', fontsize=14)
plt.title(f'Kolokacijska metoda - delta začetni pogoj (t≤{t_max_delta}a²/D)', fontsize=16)
plt.legend(fontsize=10, loc='upper right', ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/graf13_collocation_delta.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf13_collocation_delta.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# DODATEN GRAF 14: Kolokacijska metoda za stopničasto funkcijo - VEČ ČASOV, DALJŠI INTERVAL
# ===================================================================
print("Dodatni graf 14: Kolokacijska metoda za stopničasto funkcijo...")

# Uporabimo kolokacijsko metodo za stopničasto funkcijo
T_colloc_step = collocation_method(T0_step, a, D, dt_step, Nt_step)

plt.figure(figsize=(14, 9))
time_indices_colloc_step = [0, Nt_step//30, Nt_step//20, Nt_step//15, Nt_step//10,
                            Nt_step//6, Nt_step//4, Nt_step//3, Nt_step//2, 
                            2*Nt_step//3, 3*Nt_step//4, 4*Nt_step//5, Nt_step-1]
colors_colloc_step = plt.cm.autumn(np.linspace(0, 1, len(time_indices_colloc_step)))

for idx, i in enumerate(time_indices_colloc_step):
    plt.plot(x, T_colloc_step[i], color=colors_colloc_step[idx], 
             label=f't = {times_step[i]:.4f} a²/D', linewidth=2, alpha=0.9)

plt.plot(x, T0_step, 'k--', label='Začetni pogoj (t=0)', linewidth=3, alpha=0.8)
plt.xlabel('x/a', fontsize=14)
plt.ylabel('T/T₀', fontsize=14)
plt.title(f'Kolokacijska metoda - stopničasti začetni pogoj (t≤{t_max_step}a²/D)', fontsize=16)
plt.legend(fontsize=10, loc='upper right', ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/graf14_collocation_step.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf14_collocation_step.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# DODATEN GRAF 15: Periodični robni pogoji za delta funkcijo - VEČ ČASOV, DALJŠI INTERVAL
# ===================================================================
print("Dodatni graf 15: Periodični robni pogoji za delta funkcijo...")

# Uporabimo periodične robne pogoje za delta funkcijo
T_delta_periodic = fourier_method_periodic(T0_delta, a, D, dt_delta, Nt_delta)

plt.figure(figsize=(14, 9))
time_indices_delta_periodic = [0, Nt_delta//30, Nt_delta//20, Nt_delta//15, Nt_delta//10,
                               Nt_delta//6, Nt_delta//4, Nt_delta//3, Nt_delta//2, 
                               2*Nt_delta//3, 3*Nt_delta//4, 4*Nt_delta//5, Nt_delta-1]
colors_delta_periodic = plt.cm.hot(np.linspace(0, 1, len(time_indices_delta_periodic)))

for idx, i in enumerate(time_indices_delta_periodic):
    plt.plot(x, T_delta_periodic[i], color=colors_delta_periodic[idx], 
             label=f't = {times_delta[i]:.4f} a²/D', linewidth=2, alpha=0.9)

plt.plot(x, T0_delta, 'k--', label='Začetni pogoj (t=0)', linewidth=3, alpha=0.8)
plt.xlabel('x/a', fontsize=14)
plt.ylabel('T/T₀', fontsize=14)
plt.title(f'Periodični robni pogoji - delta začetni pogoj (t≤{t_max_delta}a²/D)', fontsize=16)
plt.legend(fontsize=10, loc='upper right', ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/graf15_delta_periodic.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf15_delta_periodic.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# GRAF 4: Primerjava metod pri t = t_max/2 - UPORABIMO NOVI t_max
# ===================================================================
print("Graf 4: Primerjava metod...")

t_half = Nt//2

plt.figure(figsize=(12, 7))
plt.plot(x, T_fourier_dirichlet[t_half], 'b-', label='Fourierova metoda', linewidth=3)
plt.plot(x, T_colloc[t_half], 'r--', label='Kolokacijska metoda', linewidth=3)
plt.xlabel('x/a', fontsize=14)
plt.ylabel('T/T₀', fontsize=14)
plt.title(f'Primerjava metod pri t = {times[t_half]:.4f} a²/D', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/graf4_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf4_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# GRAF 5: Razlika med metodama - UPORABIMO NOVI t_max
# ===================================================================
print("Graf 5: Razlika med metodama...")

diff = T_fourier_dirichlet - T_colloc

plt.figure(figsize=(12, 7))
im = plt.imshow(diff.T, aspect='auto', extent=[0, t_max, 0, a],
                origin='lower', cmap='RdBu_r', vmin=-0.005, vmax=0.005)
plt.colorbar(im, label='ΔT/T₀')
plt.xlabel('t/(a²/D)', fontsize=14)
plt.ylabel('x/a', fontsize=14)
plt.title(f'Razlika med Fourierovo in kolokacijsko metodo (t≤{t_max}a²/D)', fontsize=16)
plt.tight_layout()
plt.savefig(f'{output_dir}/graf5_difference.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf5_difference.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# GRAF 6: Stabilnost Fourierove metode - OSTANE NESPREMENJEN
# ===================================================================
print("Graf 6: Stabilnost Fourierove metode...")

dx_vals = np.logspace(-3, -1, 15)
dt_max_theory = 2/(np.pi**2) * dx_vals**2 / D * a**2

# Simulacije za preverjanje stabilnosti
dt_test_vals = []
for dx in dx_vals:
    N_test = max(10, int(a/dx))
    x_test = np.linspace(0, a, N_test)
    T0_test = gaussian_initial(x_test, a, sigma)
    
    dt_test = dt_max_theory[np.where(dx_vals == dx)[0][0]]
    stable = True
    
    # Testiramo stabilnost z večjim številom korakov
    try:
        T_test = fourier_method_periodic(T0_test, a, D, dt_test, 200)
        if np.any(np.isnan(T_test)) or np.any(np.abs(T_test) > 10):
            stable = False
    except:
        stable = False
    
    if stable:
        dt_test_vals.append(dt_test)
    else:
        dt_test_vals.append(dt_test * 0.7)

plt.figure(figsize=(12, 7))
plt.loglog(dx_vals, dt_max_theory, 'b-', linewidth=3, label='Teoretična napoved')
plt.loglog(dx_vals, dt_test_vals, 'ro', markersize=8, label='Numerične meritve')
plt.xlabel('Δx/a', fontsize=14)
plt.ylabel('Δt/(a²/D)', fontsize=14)
plt.title('Meja stabilnosti Fourierove metode (Eulerjeva shema)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{output_dir}/graf6_stability.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf6_stability.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# GRAF 7: Hitrost izvajanja - OSTANE NESPREMENJEN
# ===================================================================
print("Graf 7: Hitrost izvajanja...")

N_vals = [50, 100, 200, 500]
times_fourier = []
times_colloc = []

dt_test = 0.004  # POVEČAN časovni korak za test hitrosti
Nt_test = 50  # ZMANJŠANO število korakov zaradi daljšega Δt

for N_val in N_vals:
    x_val = np.linspace(0, a, N_val)
    T0_val = gaussian_initial(x_val, a, sigma)
    
    start = time.time()
    _ = fourier_method_periodic(T0_val, a, D, dt_test, Nt_test)
    times_fourier.append(time.time() - start)
    
    start = time.time()
    _ = collocation_method(T0_val, a, D, dt_test, Nt_test)
    times_colloc.append(time.time() - start)

plt.figure(figsize=(12, 7))
plt.plot(N_vals, times_fourier, 'bo-', linewidth=3, markersize=8, label='Fourierova metoda')
plt.plot(N_vals, times_colloc, 'ro-', linewidth=3, markersize=8, label='Kolokacijska metoda')
plt.xlabel('Število prostorskih točk N', fontsize=14)
plt.ylabel('Čas izvajanja [s]', fontsize=14)
plt.title(f'Hitrost izvajanja ({Nt_test} časovnih korakov, Δt={dt_test})', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/graf7_speed.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf7_speed.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# GRAF 8: 3D prikaz časovnega razvoja - ŠE DALJŠI INTERVAL
# ===================================================================
print("Graf 8: 3D prikaz z daljšim časom...")

# POSEBNE NASTAVITVE ZA 3D GRAF - ŠE DALJŠI ČAS
t_max_3d = 0.3  # POVEČANO na 0.3
Nt_3d = 150     # POVEČANO število korakov
dt_3d = t_max_3d / Nt_3d

# Izračunajmo razvoj za daljši čas
T_3d = fourier_method_periodic(T0, a, D, dt_3d, Nt_3d)
times_3d = np.linspace(0, t_max_3d, Nt_3d)

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Uporabimo vsak 3. časovni korak za boljšo preglednost
step = max(1, Nt_3d//75)
X, T_mesh = np.meshgrid(x, times_3d[::step])
Z = T_3d[::step]

surf = ax.plot_surface(X, T_mesh, Z, cmap='viridis', alpha=0.8, 
                      linewidth=0, antialiased=True, rstride=1, cstride=1)
ax.set_xlabel('x/a', fontsize=14, labelpad=12)
ax.set_ylabel('t/(a²/D)', fontsize=14, labelpad=12)
ax.set_zlabel('T/T₀', fontsize=14, labelpad=12)
ax.set_title(f'Časovni razvoj temperature - Fourierova metoda (t≤{t_max_3d}a²/D)', fontsize=16)
ax.view_init(elev=25, azim=-60)  # Boljši kot pogleda
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='T/T₀')
plt.tight_layout()
plt.savefig(f'{output_dir}/graf8_3d.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf8_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# DODATEN GRAF: Analiza padca amplitude Gaussove funkcije - UPORABIMO NOVI t_max
# ===================================================================
print("Dodatni graf: Analiza padca amplitude...")

# Izračunajmo maksimalno temperaturo v odvisnosti od časa
max_temp_fourier = np.max(T_fourier_periodic, axis=1)
max_temp_colloc = np.max(T_colloc, axis=1)

# Analitična napoved za neskončno palico
# Za Gaussov začetni pogoj: T_max(t) = 1/√(1 + 4Dt/σ²)
analytical_max = 1 / np.sqrt(1 + 4 * D * times / sigma**2)

plt.figure(figsize=(12, 7))
plt.plot(times, max_temp_fourier, 'b-', label='Fourierova metoda', linewidth=2)
plt.plot(times, max_temp_colloc, 'r--', label='Kolokacijska metoda', linewidth=2)
plt.plot(times, analytical_max, 'g-.', label='Analitična napoved', linewidth=2)
plt.xlabel('t/(a²/D)', fontsize=14)
plt.ylabel('Maksimalna temperatura T_max/T₀', fontsize=14)
plt.title(f'Časovni padec amplitude Gaussove temperature (t≤{t_max}a²/D)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Logaritemska skala za boljši prikaz eksponentnega padca
plt.tight_layout()
plt.savefig(f'{output_dir}/graf11_amplitude_decay.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf11_amplitude_decay.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# DODATEN GRAF: Fourierovi koeficienti v času - UPORABIMO NOVI t_max
# ===================================================================
print("Dodatni graf: Fourierovi koeficienti...")

# Izračunajmo Fourierove koeficiente za periodični primer
Tk_history = []
Tk_current = fft(T0)
k = np.fft.fftfreq(N, d=a/N)
factor = -4 * np.pi**2 * D * k**2

for n in range(Nt):
    if n > 0:
        Tk_current *= (1 + dt * factor)
    Tk_history.append(Tk_current.copy())

Tk_history = np.array(Tk_history)

# Izberemo več Fourierovih modov za prikaz
modes_to_show = [0, 1, 2, 3, 5, 8, 12, 20, 30, 40]
mode_labels = ['k=0', 'k=1', 'k=2', 'k=3', 'k=5', 'k=8', 'k=12', 'k=20', 'k=30', 'k=40']

plt.figure(figsize=(12, 7))
for idx, mode in enumerate(modes_to_show):
    if mode < len(k):
        amplitude = np.abs(Tk_history[:, mode])
        plt.plot(times, amplitude, label=mode_labels[idx], linewidth=2)

plt.xlabel('t/(a²/D)', fontsize=14)
plt.ylabel('|Tₖ(t)|', fontsize=14)
plt.title(f'Časovni razvoj Fourierovih koeficientov (t≤{t_max}a²/D)', fontsize=16)
plt.legend(fontsize=10, loc='upper right', ncol=2)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log skala za boljši prikaz eksponentnega padca
plt.tight_layout()
plt.savefig(f'{output_dir}/graf12_fourier_coefficients.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/graf12_fourier_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Vsi grafi so shranjeni v mapo '{output_dir}'")
print("\nGLAVNE SPREMEMBE:")
print(f"1. DALJŠI ČASOVNI INTERVALI:")
print(f"   - Glavni grafi (1-3): t_max = {t_max} (prej 0.02)")
print(f"   - Delta funkcija: t_max = {t_max_delta} (prej 0.3)")
print(f"   - Stopničasta funkcija: t_max = {t_max_step} (prej 0.4)")
print(f"   - 3D prikaz: t_max = {t_max_3d} (prej 0.1)")
print(f"2. VEČ ČASOVNIH TOČK:")
print(f"   - Vsi grafi prikazujejo 13 časovnih točk (prej 9)")
print(f"   - Boljši prikaz celotnega časovnega razvoja")
print(f"3. VEČ FOUREJROVIH MODOV: 10 modov namesto 6")
print(f"4. VEČJE SLIKE: Za boljšo berljivost")
print(f"5. VSE OBSTOJEČE FUNKCIONALNOSTI OHNJENE")
print(f"\nSKUPAJ: 15 GRAFOV")
print("Analiza končana!")
