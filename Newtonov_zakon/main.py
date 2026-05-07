import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.special import ellipk
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ============================================
# Mathematical Pendulum - Core Functions
# ============================================

class Pendulum:
    def __init__(self, omega0=1.0):
        self.omega0 = omega0
        
    def equation(self, state):
        theta, theta_dot = state
        return np.array([theta_dot, -self.omega0**2 * np.sin(theta)])
    
    def exact_period(self, theta0):
        if theta0 == 0:
            return 2*np.pi/self.omega0
        k = np.sin(theta0/2)**2
        if k >= 1:
            k = 0.999999
        return 4 * ellipk(k) / self.omega0
    
    def energy(self, theta, theta_dot):
        kinetic = 0.5 * theta_dot**2
        potential = self.omega0**2 * (1 - np.cos(theta))
        return kinetic + potential

# ============================================
# Numerical Methods
# ============================================

def euler_method(f, y0, t_span, h):
    t0, tf = t_span
    n_steps = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n_steps)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0
    
    for i in range(n_steps-1):
        y[i+1] = y[i] + h * f(y[i])
    
    return t, y

def heun_method(f, y0, t_span, h):
    t0, tf = t_span
    n_steps = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n_steps)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0
    
    for i in range(n_steps-1):
        k1 = f(y[i])
        k2 = f(y[i] + h * k1)
        y[i+1] = y[i] + h/2 * (k1 + k2)
    
    return t, y

def rk4_method(f, y0, t_span, h):
    t0, tf = t_span
    n_steps = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n_steps)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0
    
    for i in range(n_steps-1):
        k1 = f(y[i])
        k2 = f(y[i] + h/2 * k1)
        k3 = f(y[i] + h/2 * k2)
        k4 = f(y[i] + h * k3)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y

def verlet_method(f, y0, t_span, h):
    t0, tf = t_span
    n_steps = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n_steps)
    theta = np.zeros(n_steps)
    theta_dot = np.zeros(n_steps)
    
    theta[0] = y0[0]
    theta_dot[0] = y0[1]
    
    a0 = f([theta[0], theta_dot[0]])[1]
    theta[1] = theta[0] + h * theta_dot[0] + 0.5 * h**2 * a0
    theta_dot[1] = theta_dot[0] + h * a0
    
    for i in range(1, n_steps-1):
        theta[i+1] = 2*theta[i] - theta[i-1] + h**2 * f([theta[i], theta_dot[i]])[1]
        theta_dot[i+1] = (theta[i+1] - theta[i-1]) / (2*h)
    
    return t, np.column_stack([theta, theta_dot])

def pefrl_method(f, y0, t_span, h):
    t0, tf = t_span
    n_steps = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n_steps)
    theta = np.zeros(n_steps)
    p = np.zeros(n_steps)
    
    xi = 0.1786178958448091
    lambda_ = -0.2123418310626054
    chi = -0.06626458266981843
    
    theta[0], p[0] = y0[0], y0[1]
    
    for i in range(n_steps-1):
        q, p_curr = theta[i], p[i]
        
        p_curr = p_curr + xi * h * f([q, p_curr])[1]
        q = q + lambda_ * h * p_curr
        p_curr = p_curr + chi * h * f([q, p_curr])[1]
        q = q + (1 - 2*lambda_) * h * p_curr
        p_curr = p_curr + chi * h * f([q, p_curr])[1]
        q = q + lambda_ * h * p_curr
        p_curr = p_curr + xi * h * f([q, p_curr])[1]
        
        theta[i+1] = q
        p[i+1] = p_curr
    
    return t, np.column_stack([theta, p])

# ============================================
# Figure 1: Time evolution
# ============================================

def figure1_time_evolution():
    """Figure 1 - time evolution"""
    print("\n" + "="*60)
    print("Časovni potek")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    pendulum = Pendulum(omega0=1.0)
    t = np.linspace(0, 15, 1000)
    amplitudes = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(amplitudes)))
    
    for theta0, color in zip(amplitudes, colors):
        _, sol = rk4_method(
            lambda state: pendulum.equation(state),
            [theta0, 0], (0, 15), 0.01
        )
        ax.plot(t[:len(sol)], sol[:len(t), 0], color=color, 
                label=f'θ₀ = {theta0}', linewidth=1.5)
    
    ax.set_xlabel('Čas (s)', fontsize=12)
    ax.set_ylabel('Kot θ (rad)', fontsize=12)
    ax.set_title('Časovni potek matematičnega nihala za različne začetne amplitude', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ============================================
# Figure 2: Phase portraits
# ============================================

def figure2_phase_portraits():
    """Figure 2/3 - phase portraits"""
    print("\n" + "="*60)
    print("Fazni portreti")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pendulum = Pendulum(omega0=1.0)
    
    conditions = {
        'Majhni nihaji': (0.5, 0, 30),
        'Veliki nihaji': (2.0, 0, 30),
        'Navijanje': (0, 2.5, 50)
    }
    
    for ax, (title, (theta0, v0, t_max)) in zip(axes, conditions.items()):
        t, sol = rk4_method(
            lambda state: pendulum.equation(state),
            [theta0, v0], (0, t_max), 0.01
        )
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sol)))
        for i in range(0, len(sol)-1, 10):
            ax.plot(sol[i:i+2, 0], sol[i:i+2, 1], color=colors[i], alpha=0.7, linewidth=1)
        
        ax.plot(sol[0, 0], sol[0, 1], 'go', markersize=8, label='Začetek')
        ax.plot(sol[-1, 0], sol[-1, 1], 'ro', markersize=8, label='Konec')
        ax.set_xlabel('Kot θ (rad)', fontsize=11)
        ax.set_ylabel('Kotna hitrost (rad/s)', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('Fazni portreti matematičnega nihala', fontsize=14)
    plt.tight_layout()
    return fig

# ============================================
# Figure 4: Error analysis (only maximum error)
# ============================================

def figure4_error_analysis():
    """Error analysis - only maximum error vs step size"""
    print("\n" + "="*60)
    print("Analiza napak - maksimalna napaka")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pendulum = Pendulum(omega0=1.0)
    theta0 = 1.0
    t_max = 15
    
    # Pre-compute reference solution
    print("  Računam referenčno rešitev (visoka natančnost)...")
    t_ref = np.linspace(0, t_max, 10000)
    
    def rhs(t, state):
        return pendulum.equation(state)
    
    sol_ref = solve_ivp(rhs, (0, t_max), [theta0, 0], t_eval=t_ref, 
                        method='DOP853', rtol=1e-12, atol=1e-14)
    
    from scipy.interpolate import interp1d
    theta_ref_interp = interp1d(sol_ref.t, sol_ref.y[0], kind='cubic', 
                                 fill_value='extrapolate')
    
    methods = {
        'Eulerjeva': euler_method,
        'Heunova': heun_method,
        'RK4': rk4_method,
        'Verletova': verlet_method,
        'PEFRL': pefrl_method
    }
    colors = {'Eulerjeva': 'red', 'Heunova': 'blue', 'RK4': 'green', 
              'Verletova': 'orange', 'PEFRL': 'purple'}
    
    h_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for method_name, method in methods.items():
        print(f"  Metoda: {method_name}")
        errors = []
        for h in h_values:
            try:
                t, sol = method(
                    lambda state: pendulum.equation(state),
                    [theta0, 0], (0, t_max), h
                )
                
                t_eval = np.linspace(0, t_max, min(1000, len(sol)))
                theta_num_interp = interp1d(t, sol[:, 0], kind='linear', 
                                             fill_value='extrapolate')
                theta_num_eval = theta_num_interp(t_eval)
                theta_ref_eval = theta_ref_interp(t_eval)
                
                error = np.max(np.abs(theta_num_eval - theta_ref_eval))
                errors.append(error)
                print(f"    h={h:.4f}s: napaka={error:.2e}")
            except Exception as e:
                errors.append(np.nan)
        
        errors = np.array(errors)
        mask = ~np.isnan(errors)
        if np.any(mask):
            ax.loglog(np.array(h_values)[mask], errors[mask], 'o-', 
                     color=colors[method_name], label=method_name, 
                     linewidth=1.5, markersize=4)
    
    # Add reference slopes
    h_ref = np.array([0.001, 0.01, 0.1])
    ax.loglog(h_ref, 0.1 * h_ref**1, 'k--', alpha=0.5, label='O(h)', linewidth=1)
    ax.loglog(h_ref, 0.5 * h_ref**2, 'k-.', alpha=0.5, label='O(h²)', linewidth=1)
    ax.loglog(h_ref, 5 * h_ref**4, 'k:', alpha=0.5, label='O(h⁴)', linewidth=1)
    
    ax.set_xlabel('Velikost koraka Δt (s)', fontsize=12)
    ax.set_ylabel('Maksimalna napaka', fontsize=12)
    ax.set_title('Maksimalna napaka v odvisnosti od velikosti koraka', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================
# Figure 5: Long-term stability
# ============================================

def figure5_stability():
    """Long-term stability analysis"""
    print("\n" + "="*60)
    print("Dolgoročna stabilnost")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    pendulum = Pendulum(omega0=1.0)
    theta0 = 1.0
    h = 0.1
    
    methods = {
        'Eulerjeva': euler_method,
        'Heunova': heun_method,
        'RK4': rk4_method,
        'Verletova': verlet_method,
        'PEFRL': pefrl_method
    }
    colors = {'Eulerjeva': 'red', 'Heunova': 'blue', 'RK4': 'green', 
              'Verletova': 'orange', 'PEFRL': 'purple'}
    
    # Energy over time
    ax1 = axes[0]
    print("  Računam ohranjanje energije...")
    
    for method_name, method in methods.items():
        print(f"    {method_name}")
        t, sol = method(
            lambda state: pendulum.equation(state),
            [theta0, 0], (0, 50), h
        )
        
        energy = np.array([pendulum.energy(s[0], s[1]) for s in sol])
        rel_energy = (energy - energy[0]) / energy[0] * 100
        
        ax1.plot(t, rel_energy, color=colors[method_name], 
                label=method_name, linewidth=1.5, alpha=0.8)
    
    ax1.set_xlabel('Čas (s)', fontsize=12)
    ax1.set_ylabel('Napaka energije (%)', fontsize=12)
    ax1.set_title('Ohranjanje energije skozi čas (Δt = 0,1 s)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 1)
    
    # Period stability
    ax2 = axes[1]
    print("  Računam stabilnost periode...")
    
    for method_name, method in methods.items():
        print(f"    {method_name}")
        t, sol = method(
            lambda state: pendulum.equation(state),
            [theta0, 0], (0, 100), h
        )
        
        peaks, _ = find_peaks(sol[:, 0], height=theta0*0.5)
        
        if len(peaks) >= 10:
            periods = []
            for i in range(1, len(peaks)):
                period = t[peaks[i]] - t[peaks[i-1]]
                periods.append(period)
            
            exact_period = pendulum.exact_period(theta0)
            rel_period_errors = [(p - exact_period)/exact_period * 100 for p in periods]
            
            ax2.plot(range(1, len(rel_period_errors)+1), rel_period_errors, 
                    'o-', color=colors[method_name], label=method_name, 
                    markersize=2, linewidth=1, alpha=0.8)
    
    ax2.set_xlabel('Številka nihaja', fontsize=12)
    ax2.set_ylabel('Napaka periode (%)', fontsize=12)
    ax2.set_title('Stabilnost periode skozi več nihajev (Δt = 0,1 s)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Energy zoom for symplectic methods
    ax3 = axes[2]
    print("  Računam energijo - simplektične metode (povečava)...")
    
    for method_name in ['Verletova', 'PEFRL']:
        method = methods[method_name]
        t, sol = method(
            lambda state: pendulum.equation(state),
            [theta0, 0], (0, 50), h
        )
        
        energy = np.array([pendulum.energy(s[0], s[1]) for s in sol])
        rel_energy = (energy - energy[0]) / energy[0] * 100
        
        ax3.plot(t, rel_energy, color=colors[method_name], 
                label=method_name, linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('Čas (s)', fontsize=12)
    ax3.set_ylabel('Napaka energije (%)', fontsize=12)
    ax3.set_title('Ohranjanje energije - približ (simplektične metode)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 20)
    ax3.set_ylim(-0.02, 0.02)
    
    plt.suptitle('Dolgoročna stabilnost in ohranjanje energije', fontsize=14)
    plt.tight_layout()
    return fig

# ============================================
# Figure 9: Driven Pendulum Phase Portraits
# ============================================

def figure9_driven_pendulum():
    """Driven pendulum regimes"""
    print("\n" + "="*60)
    print("Vsiljeno nihalo - dinamični režimi")
    print("="*60)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    beta = 0.5
    regimes = [
        ('Periodično', {'v': 0.8, 'omega': 0.7}),
        ('Podvojitev periode', {'v': 1.0, 'omega': 0.7}),
        ('Perioda 3', {'v': 1.15, 'omega': 0.7}),
        ('Kaotično', {'v': 1.2, 'omega': 0.7}),
        ('Perioda 5', {'v': 1.25, 'omega': 0.68}),
        ('Navijanje', {'v': 1.5, 'omega': 0.6}),
        ('Velika amplituda', {'v': 1.4, 'omega': 0.65}),
        ('Kompleksno', {'v': 0.95, 'omega': 0.72})
    ]
    
    for idx, (title, params) in enumerate(regimes):
        ax = axes[idx]
        
        def rhs(t, state):
            return [state[1], -beta*state[1] - np.sin(state[0]) + params['v']*np.cos(params['omega']*t)]
        
        t_span = (0, 300)
        sol = solve_ivp(rhs, t_span, [0, 0], method='RK45', rtol=1e-6, t_eval=np.linspace(0, 300, 6000))
        
        n_points = min(4000, len(sol.y[0]))
        ax.plot(sol.y[0, -n_points:], sol.y[1, -n_points:], 'b-', linewidth=0.5, alpha=0.7)
        ax.set_xlabel('θ (rad)', fontsize=9)
        ax.set_ylabel('θ̇ (rad/s)', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('auto')
    
    plt.suptitle('Vsiljeno matematično nihalo - različni dinamični režimi (β = 0,5)', fontsize=14)
    plt.tight_layout()
    return fig

# ============================================
# Figure 12: Resonance curves
# ============================================

def figure12_resonance():
    """Resonance curves"""
    print("\n" + "="*60)
    print("Resonančne krivulje")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    beta = 0.5
    v_values = [0.3, 0.6, 0.82, 1.0, 1.2]
    colors = plt.cm.viridis(np.linspace(0, 1, len(v_values)))
    
    for v_amp, color in zip(v_values, colors):
        omega_range = np.linspace(0.5, 1.5, 25)
        amplitudes = []
        
        for omega in omega_range:
            def rhs(t, state):
                return [state[1], -beta*state[1] - np.sin(state[0]) + v_amp*np.cos(omega*t)]
            
            # Simulate longer to reach steady state
            sol = solve_ivp(rhs, (0, 300), [0, 0], method='RK45', rtol=1e-6)
            
            period = 2*np.pi/omega
            t_start = sol.t[-1] - 10*period
            mask = sol.t >= t_start
            if np.any(mask):
                amplitude = np.max(np.abs(sol.y[0, mask]))
                amplitudes.append(amplitude)
            else:
                amplitudes.append(0)
        
        ax.plot(omega_range, amplitudes, '-o', color=color, linewidth=2, markersize=4, label=f'v = {v_amp}')
    
    # Add harmonic oscillator for reference
    omega_harm = np.linspace(0.5, 1.5, 100)
    A_harm = [0.82 / np.sqrt((1 - omega**2)**2 + (2*beta*omega)**2) for omega in omega_harm]
    ax.plot(omega_harm, A_harm, 'k--', linewidth=2, alpha=0.7, label='Harmonični oscilator (v=0.82)')
    
    ax.set_xlabel('Vzbujevalna frekvenca ω', fontsize=12)
    ax.set_ylabel('Amplituda odziva (rad)', fontsize=12)
    ax.set_title('Resonančne krivulje za različne amplitude vzbujanja (β = 0,5)', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.5)
    
    plt.tight_layout()
    return fig

# ============================================
# Figure 13: Hysteresis (full calculation)
# ============================================

def figure13_hysteresis():
    """Hysteresis in Duffing/pendulum system"""
    print("\n" + "="*60)
    print("Histereza - računam (to lahko traja nekaj minut)...")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    beta = 0.03
    v_amp = 0.03
    
    # Forward sweep (increasing frequency)
    print("  Računam naraščajočo frekvenco...")
    omega_forward = np.linspace(0.8, 1.3, 35)
    amps_forward = []
    
    for i, omega in enumerate(omega_forward):
        print(f"    ω = {omega:.3f} ({i+1}/{len(omega_forward)})")
        
        def rhs(t, state):
            return [state[1], -beta*state[1] - np.sin(state[0]) + v_amp*np.cos(omega*t)]
        
        # Use previous amplitude as initial condition for faster convergence
        if amps_forward and amps_forward[-1] > 0:
            x0 = amps_forward[-1] * 0.8
        else:
            x0 = 0.1
        
        sol = solve_ivp(rhs, (0, 800), [x0, 0], method='RK45', rtol=1e-7, atol=1e-9)
        
        period = 2*np.pi/omega
        t_start = sol.t[-1] - 20*period
        mask = sol.t >= t_start
        if np.any(mask):
            amplitude = np.max(np.abs(sol.y[0, mask]))
            amps_forward.append(amplitude)
        else:
            amps_forward.append(0)
    
    # Backward sweep (decreasing frequency)
    print("  Računam padajočo frekvenco...")
    omega_backward = np.linspace(1.3, 0.8, 35)
    amps_backward = []
    
    for i, omega in enumerate(omega_backward):
        print(f"    ω = {omega:.3f} ({i+1}/{len(omega_backward)})")
        
        def rhs(t, state):
            return [state[1], -beta*state[1] - np.sin(state[0]) + v_amp*np.cos(omega*t)]
        
        # Start from higher amplitude for backward sweep
        if amps_backward and amps_backward[-1] > 0:
            x0 = amps_backward[-1] * 1.2
        else:
            x0 = 1.0
        
        sol = solve_ivp(rhs, (0, 800), [x0, 0], method='RK45', rtol=1e-7, atol=1e-9)
        
        period = 2*np.pi/omega
        t_start = sol.t[-1] - 20*period
        mask = sol.t >= t_start
        if np.any(mask):
            amplitude = np.max(np.abs(sol.y[0, mask]))
            amps_backward.append(amplitude)
        else:
            amps_backward.append(0)
    
    # Plot hysteresis
    ax.plot(omega_forward, amps_forward, 'r-o', linewidth=2, markersize=5, 
            label='Naraščajoča frekvenca', alpha=0.8)
    ax.plot(omega_backward, amps_backward, 'b-o', linewidth=2, markersize=5, 
            label='Padajoča frekvenca', alpha=0.8)
    
    # Fill hysteresis loop
    omega_hyst = np.concatenate([omega_forward, omega_backward[::-1]])
    amps_hyst = np.concatenate([amps_forward, amps_backward[::-1]])
    ax.fill(omega_hyst, amps_hyst, alpha=0.2, color='gray', label='Histerezna zanka')
    
    ax.set_xlabel('Vzbujevalna frekvenca ω', fontsize=12)
    ax.set_ylabel('Amplituda odziva (rad)', fontsize=12)
    ax.set_title('Histereza v sistemu Duffing/nihalo (β = 0,03, v = 0,03)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.8, 1.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    return fig

# ============================================
# Figure 14/15: van der Pol Oscillator
# ============================================

def figure14_vanderpol():
    """van der Pol oscillator"""
    print("\n" + "="*60)
    print("Van der Polov oscilator")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    omega_drive = 1.0
    v_amp = 10.0
    lambdas = [1, 5, 10, 15, 20, 25]
    
    for idx, lambd in enumerate(lambdas):
        row = idx // 3
        col = idx % 3
        ax_phase = axes[row, col]
        
        def rhs(t, state):
            x, x_dot = state
            return [x_dot, lambd*(1 - x**2)*x_dot - x + v_amp*np.cos(omega_drive*t)]
        
        print(f"  λ = {lambd}")
        
        try:
            sol = solve_ivp(rhs, (0, 300), [0.1, 0], method='BDF' if lambd > 10 else 'RK45', 
                           rtol=1e-6, atol=1e-8, t_eval=np.linspace(0, 300, 6000))
            
            n_points = min(4000, len(sol.y[0]))
            ax_phase.plot(sol.y[0, -n_points:], sol.y[1, -n_points:], 'b-', 
                         linewidth=0.8, alpha=0.7)
            ax_phase.set_xlabel('x', fontsize=10)
            ax_phase.set_ylabel('dx/dt', fontsize=10)
            ax_phase.set_title(f'λ = {lambd}', fontsize=11)
            ax_phase.grid(True, alpha=0.3)
            
        except Exception as e:
            ax_phase.text(0.5, 0.5, f'Napaka: {str(e)[:50]}', 
                         ha='center', va='center', transform=ax_phase.transAxes)
    
    plt.suptitle('Van der Polov oscilator (v = 10, ω = 1)', fontsize=14)
    plt.tight_layout()
    return fig

# ============================================
# Figure 16: Devil's Staircase (full calculation)
# ============================================

def figure16_devils_staircase():
    """Devil's staircase - full calculation"""
    print("\n" + "="*60)
    print("Hudičevo stopnišče - računam (to lahko traja nekaj minut)...")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    v_amp = 1.2
    omega_drive = 2*np.pi/10  # T = 10s
    lambda_range = np.linspace(1, 15, 50)
    
    period_ratios = []
    
    for i, lambd in enumerate(lambda_range):
        print(f"  λ = {lambd:.2f} ({i+1}/{len(lambda_range)})")
        
        def rhs(t, state):
            return van_der_pol(t, state, lambd, v_amp, omega_drive)
        
        # Long simulation to get accurate period
        t_span = (0, 2000)
        try:
            sol = solve_ivp(rhs, t_span, [0.1, 0], method='BDF' if lambd > 8 else 'RK45',
                           rtol=1e-8, atol=1e-10, t_eval=np.linspace(0, 2000, 40000))
            
            # Find peaks
            peaks, _ = find_peaks(sol.y[0], height=0.5)
            
            if len(peaks) > 20:
                # Calculate periods using last part for steady state
                peaks = peaks[-30:]  # Use last 30 peaks
                periods = np.diff(sol.t[peaks])
                avg_period = np.mean(periods[-15:])  # Average of last 15 periods
                driving_period = 2*np.pi/omega_drive
                ratio = avg_period / driving_period
                period_ratios.append(ratio)
                print(f"      Razmerje = {ratio:.4f}")
            else:
                period_ratios.append(np.nan)
                print(f"      Premalo vrhov ({len(peaks)})")
        except Exception as e:
            period_ratios.append(np.nan)
            print(f"      Napaka: {str(e)[:50]}")
    
    period_ratios = np.array(period_ratios)
    mask = ~np.isnan(period_ratios)
    
    # Plot
    ax.plot(lambda_range[mask], period_ratios[mask], 'b.-', linewidth=1.5, markersize=6, alpha=0.8)
    
    # Mark rational ratios
    rationals = [1/3, 1/2, 2/3, 1, 4/3, 3/2, 5/3, 2]
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']
    for ratio, color in zip(rationals, colors):
        ax.axhline(y=ratio, color=color, linestyle='--', alpha=0.4, linewidth=1)
        ax.text(lambda_range[-1]*0.95, ratio*1.02, f'{ratio:.1f}', 
                fontsize=9, color=color)
    
    ax.set_xlabel('Parameter λ', fontsize=12)
    ax.set_ylabel('Razmerje period (T_osc / T_vzb)', fontsize=12)
    ax.set_title('Hudičevo stopnišče pri van der Polovem oscilatorju (v = 1,2, T_vzb = 10 s)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lambda_range[0], lambda_range[-1])
    #ax.set_ylim(0.3, 2.2)
    
    plt.tight_layout()
    return fig

# Helper function for van der Pol (used in Devil's staircase)
def van_der_pol(t, state, lambd, v_amp, omega_drive):
    x, x_dot = state
    return [x_dot, lambd*(1 - x**2)*x_dot - x + v_amp*np.cos(omega_drive*t)]

# ============================================
# Main execution
# ============================================

def main():
    """Generate all figures"""
    print("\n" + "="*60)
    print("USTVARJAM SLIKE IZ POROČILA")
    print("="*60)
    
    total_start = time.time()
    
    # Generate all figures
    print("\n1/8: Ustvarjam sliko - Časovni potek...")
    start = time.time()
    fig1 = figure1_time_evolution()
    fig1.savefig('slika1_casovni_potek.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Končano v {time.time()-start:.1f}s")
    
    print("\n2/8: Ustvarjam sliko - Fazni portreti...")
    start = time.time()
    fig2 = figure2_phase_portraits()
    fig2.savefig('slika2_fazni_portreti.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Končano v {time.time()-start:.1f}s")
    
    print("\n3/8: Ustvarjam sliko - Analiza napak...")
    start = time.time()
    fig4 = figure4_error_analysis()
    fig4.savefig('slika4_analiza_napak.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"  Končano v {time.time()-start:.1f}s")
    
    print("\n4/8: Ustvarjam sliko - Dolgoročna stabilnost...")
    start = time.time()
    fig5 = figure5_stability()
    fig5.savefig('slika5_stabilnost.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print(f"  Končano v {time.time()-start:.1f}s")
    
    print("\n5/8: Ustvarjam sliko - Režimi vsiljenega nihala...")
    start = time.time()
    fig9 = figure9_driven_pendulum()
    fig9.savefig('slika9_vsiljeno_nihalo_rezimi.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig9)
    print(f"  Končano v {time.time()-start:.1f}s")
    
    print("\n6/8: Ustvarjam sliko - Resonančne krivulje...")
    start = time.time()
    fig12 = figure12_resonance()
    fig12.savefig('slika12_resonancne_krivulje.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig12)
    print(f"  Končano v {time.time()-start:.1f}s")
    
    print("\n7/8: Ustvarjam sliko - Histereza...")
    start = time.time()
    fig13 = figure13_hysteresis()
    fig13.savefig('slika13_histereza.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig13)
    print(f"  Končano v {time.time()-start:.1f}s")
    
    print("\n8/8: Ustvarjam sliko - Van der Polov oscilator...")
    start = time.time()
    fig14 = figure14_vanderpol()
    fig14.savefig('slika14_vanderpol.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig14)
    print(f"  Končano v {time.time()-start:.1f}s")
    
    print("\n9/9: Ustvarjam sliko - Hudičevo stopnišče...")
    start = time.time()
    fig16 = figure16_devils_staircase()
    fig16.savefig('slika16_hudicevo_stopnisce.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig16)
    print(f"  Končano v {time.time()-start:.1f}s")
    
    print("\n" + "="*60)
    print(f"✅ Vse slike so ustvarjene in shranjene v PDF formatu!")
    print(f"   Skupni čas: {time.time()-total_start:.1f} sekund")
    print("="*60)
    print("\nShranjene datoteke:")
    print("  - slika1_casovni_potek.pdf")
    print("  - slika2_fazni_portreti.pdf")
    print("  - slika4_analiza_napak.pdf")
    print("  - slika5_stabilnost.pdf")
    print("  - slika9_vsiljeno_nihalo_rezimi.pdf")
    print("  - slika12_resonancne_krivulje.pdf")
    print("  - slika13_histereza.pdf")
    print("  - slika14_vanderpol.pdf")
    print("  - slika16_hudicevo_stopnisce.pdf")

if __name__ == "__main__":
    main()
