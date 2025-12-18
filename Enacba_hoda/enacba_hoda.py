import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
import matplotlib
from time import perf_counter
import decimal
from decimal import Decimal, getcontext

# Set up plotting
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

# Create directory for plots
os.makedirs('diffeq_plots', exist_ok=True)

class DifferentialEquationSolver:
    def __init__(self):
        self.methods = {}
        
    def cooling_equation(self, t, T, k, T_zun):
        """Basic cooling equation: dT/dt = -k(T - T_zun)"""
        return -k * (T - T_zun)
    
    def cooling_with_heating(self, t, T, k, T_zun, A, delta):
        """Cooling equation with periodic heating: dT/dt = -k(T - T_zun) + A*sin(2π/24*(t-delta))"""
        return -k * (T - T_zun) + A * np.sin(2*np.pi/24 * (t - delta))
    
    def euler_method(self, f, t_span, y0, h, *args):
        """Basic Euler method (1st order)"""
        t0, tf = t_span
        t_values = np.arange(t0, tf + h, h)
        y_values = np.zeros(len(t_values))
        y_values[0] = y0
        
        for i in range(len(t_values)-1):
            y_values[i+1] = y_values[i] + h * f(t_values[i], y_values[i], *args)
            
        return t_values, y_values
    
    def midpoint_method(self, f, t_span, y0, h, *args):
        """Midpoint method (2nd order)"""
        t0, tf = t_span
        t_values = np.arange(t0, tf + h, h)
        y_values = np.zeros(len(t_values))
        y_values[0] = y0
        
        for i in range(len(t_values)-1):
            K1 = h * f(t_values[i], y_values[i], *args)
            K2 = h * f(t_values[i] + 0.5*h, y_values[i] + 0.5*K1, *args)
            y_values[i+1] = y_values[i] + K2
            
        return t_values, y_values
    
    def heun_method(self, f, t_span, y0, h, *args):
        """Heun's method (2nd order)"""
        t0, tf = t_span
        t_values = np.arange(t0, tf + h, h)
        y_values = np.zeros(len(t_values))
        y_values[0] = y0
        
        for i in range(len(t_values)-1):
            y_pred = y_values[i] + h * f(t_values[i], y_values[i], *args)
            y_values[i+1] = y_values[i] + 0.5 * h * (
                f(t_values[i], y_values[i], *args) + 
                f(t_values[i+1], y_pred, *args)
            )
            
        return t_values, y_values
    
    def rk4_method(self, f, t_span, y0, h, *args):
        """Runge-Kutta 4th order method"""
        t0, tf = t_span
        t_values = np.arange(t0, tf + h, h)
        y_values = np.zeros(len(t_values))
        y_values[0] = y0
        
        for i in range(len(t_values)-1):
            k1 = h * f(t_values[i], y_values[i], *args)
            k2 = h * f(t_values[i] + 0.5*h, y_values[i] + 0.5*k1, *args)
            k3 = h * f(t_values[i] + 0.5*h, y_values[i] + 0.5*k2, *args)
            k4 = h * f(t_values[i] + h, y_values[i] + k3, *args)
            
            y_values[i+1] = y_values[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
            
        return t_values, y_values
    
    def analytical_solution(self, t, T0, k, T_zun):
        """Analytical solution for basic cooling equation"""
        return T_zun + np.exp(-k * t) * (T0 - T_zun)
    
    def analyze_convergence_rates(self):
        """Detailed analysis of convergence rates for different methods"""
        T0 = 21
        T_zun = -5
        k = 0.1
        t_span = (0, 10)
        
        # Reference solution with very small step size
        t_ref, T_ref = self.rk4_method(self.cooling_equation, t_span, T0, 0.001, k, T_zun)
        T_exact_ref = T_ref[-1]
        
        h_range = np.logspace(-3, 0, 20)
        methods = [
            ('Euler', self.euler_method, 1),
            ('Midpoint', self.midpoint_method, 2), 
            ('Heun', self.heun_method, 2),
            ('RK4', self.rk4_method, 4)
        ]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot errors and determine convergence rates
        for method_name, method_func, theoretical_order in methods:
            errors = []
            valid_h = []
            
            for h in h_range:
                try:
                    t_num, T_num = method_func(self.cooling_equation, t_span, T0, h, k, T_zun)
                    if len(T_num) > 0:
                        error = abs(T_num[-1] - T_exact_ref)
                        errors.append(error)
                        valid_h.append(h)
                except:
                    continue
            
            if len(errors) > 1:
                # Fit power law to determine actual convergence rate
                log_h = np.log(valid_h)
                log_err = np.log(errors)
                coeffs = np.polyfit(log_h, log_err, 1)
                actual_order = abs(coeffs[0])  # Absolute value since error decreases with h
                
                ax1.loglog(valid_h, errors, 'o-', label=f'{method_name} (teorija: O(h^{theoretical_order}), eksperiment: O(h^{actual_order:.2f}))', markersize=4)
        
        ax1.set_xlabel('Korak h [h]', fontsize=12)
        ax1.set_ylabel('Absolutna napaka pri t=10', fontsize=12)
        ax1.set_title('Konvergenca numeričnih metod', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Compare with theoretical convergence lines
        h_theory = np.logspace(-3, 0, 10)
        ax2.loglog(h_theory, 0.1*h_theory, 'k--', alpha=0.5, label='Teoretična O(h)')
        ax2.loglog(h_theory, 0.1*h_theory**2, 'k-.', alpha=0.5, label='Teoretična O(h²)') 
        ax2.loglog(h_theory, 0.1*h_theory**4, 'k:', alpha=0.5, label='Teoretična O(h⁴)')
        
        # Test different methods for comparison
        for method_name, method_func, _ in methods:
            errors = []
            valid_h = []
            
            for h in h_range[:10]:  # Fewer points for speed
                try:
                    t_num, T_num = method_func(self.cooling_equation, t_span, T0, h, k, T_zun)
                    if len(T_num) > 0:
                        error = abs(T_num[-1] - T_exact_ref)
                        errors.append(error)
                        valid_h.append(h)
                except:
                    continue
            
            if len(errors) > 0:
                ax2.loglog(valid_h, errors, 's-', label=f'{method_name}', markersize=4)
        
        ax2.set_xlabel('Korak h [h]', fontsize=12)
        ax2.set_ylabel('Absolutna napaka', fontsize=12)
        ax2.set_title('Primerjava s teoretično konvergenco', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig('diffeq_plots/02_step_size_dependence.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('diffeq_plots/02_step_size_dependence.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_methods_basic_cooling(self):
        """Compare all methods for basic cooling equation with proper analysis"""
        # Parameters
        T0_1, T0_2 = 21, -15  # Initial temperatures
        T_zun = -5             # External temperature
        k = 0.1                # Cooling parameter
        t_span = (0, 50)       # Time span
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, T0 in enumerate([T0_1, T0_2]):
            ax = axes[0, idx]
            
            # Analytical solution
            t_analytical = np.linspace(t_span[0], t_span[1], 1000)
            T_analytical = self.analytical_solution(t_analytical, T0, k, T_zun)
            ax.plot(t_analytical, T_analytical, 'k-', linewidth=3, label='Analitična rešitev')
            
            # Numerical methods
            h = 1.0
            methods = [
                ('Euler', self.euler_method),
                ('Midpoint', self.midpoint_method),
                ('Heun', self.heun_method),
                ('RK4', self.rk4_method)
            ]
            
            colors = ['red', 'blue', 'green', 'orange']
            linestyles = ['--', '-.', ':', '-']
            
            for i, (method_name, method_func) in enumerate(methods):
                t_num, T_num = method_func(self.cooling_equation, t_span, T0, h, k, T_zun)
                ax.plot(t_num, T_num, linestyles[i], color=colors[i], alpha=0.8, 
                       linewidth=2, label=f'{method_name} (h={h})')
            
            ax.set_xlabel('Čas t [h]', fontsize=12)
            ax.set_ylabel('Temperatura T [°C]', fontsize=12)
            ax.set_title(f'Primerjava metod: T(0) = {T0}°C', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        # Error analysis for different step sizes
        ax = axes[1, 0]
        T0 = T0_1
        t_final = t_span[1]
        T_exact = self.analytical_solution(t_final, T0, k, T_zun)
        
        # Test different step sizes
        h_test = [2.0, 1.0, 0.5, 0.1]
        methods = [('Euler', self.euler_method), ('Midpoint', self.midpoint_method),
                  ('Heun', self.heun_method), ('RK4', self.rk4_method)]
        
        bar_width = 0.2
        x_pos = np.arange(len(h_test))
        
        for i, (method_name, method_func) in enumerate(methods):
            errors = []
            for h in h_test:
                t_num, T_num = method_func(self.cooling_equation, t_span, T0, h, k, T_zun)
                error = abs(T_num[-1] - T_exact)
                errors.append(error)
            
            ax.bar(x_pos + i*bar_width, errors, bar_width, label=method_name, alpha=0.8)
        
        ax.set_xlabel('Velikost koraka h [h]', fontsize=12)
        ax.set_ylabel('Absolutna napaka pri t=50', fontsize=12)
        ax.set_title('Vpliv velikosti koraka na natančnost', fontsize=14)
        ax.set_xticks(x_pos + bar_width*1.5)
        ax.set_xticklabels([f'{h}' for h in h_test])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Parameter study
        ax = axes[1, 1]
        k_values = [0.05, 0.1, 0.2, 0.5]
        T0 = T0_1
        
        for k_val in k_values:
            t_analytical = np.linspace(t_span[0], t_span[1], 1000)
            T_analytical = self.analytical_solution(t_analytical, T0, k_val, T_zun)
            ax.plot(t_analytical, T_analytical, linewidth=2, label=f'k = {k_val}')
        
        ax.set_xlabel('Čas t [h]', fontsize=12)
        ax.set_ylabel('Temperatura T [°C]', fontsize=12)
        ax.set_title('Vpliv parametra ohlajanja k', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig('diffeq_plots/01_basic_cooling_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('diffeq_plots/01_basic_cooling_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def high_precision_analysis(self):
        """Analysis using high precision arithmetic"""
        print("Analiza z visoko natančnostjo...")
        
        # Test basic cooling equation with standard precision
        T0 = 21
        T_zun = -5
        k = 0.1
        t_final = 50
        
        # Analytical solution with standard precision
        T_analytical_std = T_zun + np.exp(-k * t_final) * (T0 - T_zun)
        
        print(f"Standardna natančnost: {T_analytical_std}")
        
        # Test numerical methods with different step sizes
        h_values = [2.0, 1.0, 0.5, 0.1, 0.05, 0.01]
        methods = [('Euler', self.euler_method), ('RK4', self.rk4_method)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method_name, method_func in methods:
            errors = []
            
            for h in h_values:
                t_num, T_num = method_func(self.cooling_equation, (0, t_final), T0, h, k, T_zun)
                if len(T_num) > 0:
                    error = abs(T_num[-1] - T_analytical_std)
                    errors.append(error)
                else:
                    errors.append(np.nan)
            
            valid_mask = ~np.isnan(errors)
            ax.semilogy(np.array(h_values)[valid_mask], np.array(errors)[valid_mask], 
                       'o-', label=method_name, markersize=6)
        
        ax.set_xlabel('Korak h [h]', fontsize=12)
        ax.set_ylabel('Absolutna napaka', fontsize=12)
        ax.set_title('Vpliv velikosti koraka na natančnost', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig('diffeq_plots/05_high_precision_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('diffeq_plots/05_high_precision_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def study_periodic_heating(self):
        """Study the cooling equation with periodic heating"""
        # Parameters
        T0 = 21
        T_zun = -5
        k = 0.1
        delta = 10
        A_values = [1, 2, 5]
        t_span = (0, 120)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Compare methods for A=1
        ax = axes[0, 0]
        A = 1
        
        methods = [
            ('Euler (h=0.5)', self.euler_method, 0.5),
            ('Midpoint (h=0.5)', self.midpoint_method, 0.5),
            ('RK4 (h=0.5)', self.rk4_method, 0.5),
            ('RK4 (h=0.1)', self.rk4_method, 0.1)
        ]
        
        colors = ['red', 'blue', 'green', 'purple']
        linestyles = ['--', '-.', ':', '-']
        
        for i, (method_name, method_func, h) in enumerate(methods):
            t_num, T_num = method_func(self.cooling_with_heating, t_span, T0, h, k, T_zun, A, delta)
            ax.plot(t_num, T_num, linestyles[i], color=colors[i], linewidth=1.5, label=method_name)
        
        ax.set_xlabel('Čas t [h]', fontsize=12)
        ax.set_ylabel('Temperatura T [°C]', fontsize=12)
        ax.set_title(f'Periodično segrevanje: A={A}, δ={delta}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Study different amplitudes
        ax = axes[0, 1]
        method_func = self.rk4_method
        h = 0.1
        
        for A in A_values:
            t_num, T_num = method_func(self.cooling_with_heating, t_span, T0, h, k, T_zun, A, delta)
            ax.plot(t_num, T_num, linewidth=1.5, label=f'A = {A}')
        
        ax.set_xlabel('Čas t [h]', fontsize=12)
        ax.set_ylabel('Temperatura T [°C]', fontsize=12)
        ax.set_title('Vpliv amplitude segrevanja A', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Maximum temperature analysis
        ax = axes[1, 0]
        A = 1
        t_num, T_num = method_func(self.cooling_with_heating, t_span, T0, h, k, T_zun, A, delta)
        
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(T_num, height=15, distance=20)
        
        ax.plot(t_num, T_num, 'b-', linewidth=1.5, label='Temperatura')
        ax.plot(t_num[peaks], T_num[peaks], 'ro', markersize=6, label='Maksimumi')
        
        # Analyze periodicity
        avg_period = 24.0  # Default value
        if len(peaks) > 1:
            periods = np.diff(t_num[peaks])
            avg_period = np.mean(periods)
            
            for i, peak in enumerate(peaks[:3]):
                ax.annotate(f'{T_num[peak]:.1f}°C\n@{t_num[peak]:.0f}h', 
                           (t_num[peak], T_num[peak]), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                           fontsize=9)
        
        ax.set_xlabel('Čas t [h]', fontsize=12)
        ax.set_ylabel('Temperatura T [°C]', fontsize=12)
        ax.set_title(f'Analiza maksimumov (povp. perioda: {avg_period:.1f}h)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Steady state analysis
        ax = axes[1, 1]
        t_steady = t_num[t_num >= 72]
        T_steady = T_num[t_num >= 72]
        
        ax.plot(t_steady, T_steady, 'b-', linewidth=1.5)
        ax.set_xlabel('Čas t [h]', fontsize=12)
        ax.set_ylabel('Temperatura T [°C]', fontsize=12)
        ax.set_title('Ustaljeno stanje (t ≥ 72 h)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Calculate statistics for steady state
        T_steady_mean = np.mean(T_steady)
        T_steady_amp = (np.max(T_steady) - np.min(T_steady)) / 2
        
        ax.text(0.05, 0.95, f'Povprečje: {T_steady_mean:.2f}°C\nAmplituda: {T_steady_amp:.2f}°C', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('diffeq_plots/03_periodic_heating.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('diffeq_plots/03_periodic_heating.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed analysis
        print("\n" + "="*60)
        print("DETALJNA ANALIZA PERIODIČNEGA SEGREVANJA")
        print("="*60)
        if len(peaks) > 0:
            print(f"Število maksimumov: {len(peaks)}")
            print(f"Povprečna perioda: {avg_period:.2f} h")
            print(f"Povprečna maksimalna temperatura: {np.mean(T_num[peaks]):.2f}°C")
            print(f"Povprečna temperatura v ustaljenem stanju: {T_steady_mean:.2f}°C")
            print(f"Amplituda nihanj v ustaljenem stanju: {T_steady_amp:.2f}°C")
    
    def parameter_study(self):
        """Study influence of different parameters"""
        T0 = 21
        T_zun = -5
        t_span = (0, 72)  # 3 days
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Different amplitudes A
        ax = axes[0, 0]
        k = 0.1
        delta = 10
        A_values = [0.5, 1, 2, 5]
        
        for A in A_values:
            t_num, T_num = self.rk4_method(self.cooling_with_heating, t_span, T0, 0.1, k, T_zun, A, delta)
            ax.plot(t_num, T_num, label=f'A = {A}')
        
        ax.set_xlabel('Čas t [h]', fontsize=12)
        ax.set_ylabel('Temperatura T [°C]', fontsize=12)
        ax.set_title('Vpliv amplitude segrevanja A', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Different phase shifts delta
        ax = axes[0, 1]
        k = 0.1
        A = 1
        delta_values = [0, 6, 12, 18]
        
        for delta in delta_values:
            t_num, T_num = self.rk4_method(self.cooling_with_heating, t_span, T0, 0.1, k, T_zun, A, delta)
            ax.plot(t_num, T_num, label=f'δ = {delta} h')
        
        ax.set_xlabel('Čas t [h]', fontsize=12)
        ax.set_ylabel('Temperatura T [°C]', fontsize=12)
        ax.set_title('Vpliv faznega zamika δ', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Different k values with periodic heating
        ax = axes[1, 0]
        A = 1
        delta = 10
        k_values = [0.05, 0.1, 0.2, 0.5]
        
        for k in k_values:
            t_num, T_num = self.rk4_method(self.cooling_with_heating, t_span, T0, 0.1, k, T_zun, A, delta)
            ax.plot(t_num, T_num, label=f'k = {k}')
        
        ax.set_xlabel('Čas t [h]', fontsize=12)
        ax.set_ylabel('Temperatura T [°C]', fontsize=12)
        ax.set_title('Vpliv parametra ohlajanja k', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Temperature range vs amplitude
        ax = axes[1, 1]
        A_range = np.linspace(0.1, 5, 20)
        temp_ranges = []
        
        for A in A_range:
            t_num, T_num = self.rk4_method(self.cooling_with_heating, (48, 72), T0, 0.1, k, T_zun, A, delta)
            if len(T_num) > 10:
                temp_range = np.max(T_num) - np.min(T_num)
                temp_ranges.append(temp_range)
            else:
                temp_ranges.append(0)
        
        ax.plot(A_range, temp_ranges, 'o-', color='red', markersize=3)
        ax.set_xlabel('Amplituda A', fontsize=12)
        ax.set_ylabel('Razpon temperature [°C]', fontsize=12)
        ax.set_title('Razpon temperature v odvisnosti od A', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('diffeq_plots/04_parameter_study.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('diffeq_plots/04_parameter_study.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def computational_efficiency_analysis(self):
        """Analysis of computational efficiency"""
        print("Analiza računske učinkovitosti...")
        
        T0 = 21
        T_zun = -5
        k = 0.1
        t_span = (0, 50)
        
        methods = [
            ('Euler', self.euler_method),
            ('Midpoint', self.midpoint_method), 
            ('Heun', self.heun_method),
            ('RK4', self.rk4_method)
        ]
        
        h_values = [2.0, 1.0, 0.5, 0.1, 0.05]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time vs accuracy
        for method_name, method_func in methods:
            times = []
            errors = []
            
            for h in h_values:
                # Measure execution time
                start_time = perf_counter()
                t_num, T_num = method_func(self.cooling_equation, t_span, T0, h, k, T_zun)
                end_time = perf_counter()
                
                if len(T_num) > 0:
                    T_exact = self.analytical_solution(t_span[1], T0, k, T_zun)
                    error = abs(T_num[-1] - T_exact)
                    
                    times.append(end_time - start_time)
                    errors.append(error)
            
            if len(times) > 0:
                ax1.loglog(times, errors, 'o-', label=method_name, markersize=6)
        
        ax1.set_xlabel('Čas izvajanja [s]', fontsize=12)
        ax1.set_ylabel('Absolutna napaka', fontsize=12)
        ax1.set_title('Učinkovitost: natančnost v odvisnosti od časa', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Steps vs accuracy
        for method_name, method_func in methods:
            steps_list = []
            errors = []
            
            for h in h_values:
                t_num, T_num = method_func(self.cooling_equation, t_span, T0, h, k, T_zun)
                
                if len(T_num) > 0:
                    T_exact = self.analytical_solution(t_span[1], T0, k, T_zun)
                    error = abs(T_num[-1] - T_exact)
                    
                    steps_list.append(len(t_num))
                    errors.append(error)
            
            if len(steps_list) > 0:
                ax2.loglog(steps_list, errors, 'o-', label=method_name, markersize=6)
        
        ax2.set_xlabel('Število korakov', fontsize=12)
        ax2.set_ylabel('Absolutna napaka', fontsize=12)
        ax2.set_title('Učinkovitost: natančnost v odvisnosti od števila korakov', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig('diffeq_plots/06_computational_efficiency.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('diffeq_plots/06_computational_efficiency.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comprehensive_analysis(self):
        """Run all analyses with proper structure"""
        print("="*70)
        print("ANALIZA ENAČB HODA - NUMERIČNE METODE ZA DIFERENCIALNE ENAČBE")
        print("="*70)
        
        print("\n1. PRIMERJAVA NUMERIČNIH METOD ZA OSNOVNO ENAČBO OHLAJANJA")
        self.compare_methods_basic_cooling()
        
        print("\n2. ANALIZA KONVERGENCE IN NAPAK")
        self.analyze_convergence_rates()
        
        print("\n3. ANALIZA PERIODIČNEGA SEGREVANJA") 
        self.study_periodic_heating()
        
        print("\n4. ŠTUDIJ PARAMETROV")
        self.parameter_study()
        
        print("\n5. ANALIZA RAČUNSKE NATAČNOSTI")
        self.high_precision_analysis()
        
        print("\n6. ANALIZA RAČUNSKE UČINKOVITOSTI")
        self.computational_efficiency_analysis()
        
        print("\n" + "="*70)
        print("ANALIZA ZAKLJUČENA")
        print("="*70)

def main():
    solver = DifferentialEquationSolver()
    solver.run_comprehensive_analysis()

if __name__ == "__main__":
    main()
