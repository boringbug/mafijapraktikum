import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'

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
    
    def compare_methods_basic_cooling(self):
        """Compare all methods for basic cooling equation"""
        # Parameters
        T0_1, T0_2 = 21, -15  # Initial temperatures
        T_zun = -5             # External temperature
        k = 0.1                # Cooling parameter
        t_span = (0, 50)       # Time span
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, T0 in enumerate([T0_1, T0_2]):
            ax = axes[idx]
            
            # Analytical solution
            t_analytical = np.linspace(t_span[0], t_span[1], 1000)
            T_analytical = self.analytical_solution(t_analytical, T0, k, T_zun)
            ax.plot(t_analytical, T_analytical, 'k-', linewidth=2, label='Analitična rešitev')
            
            # Numerical methods
            h = 0.5
            methods = [
                ('Euler', self.euler_method),
                ('Midpoint', self.midpoint_method),
                ('Heun', self.heun_method),
                ('RK4', self.rk4_method)
            ]
            
            colors = ['red', 'blue', 'green', 'orange']
            for i, (method_name, method_func) in enumerate(methods):
                t_num, T_num = method_func(self.cooling_equation, t_span, T0, h, k, T_zun)
                ax.plot(t_num, T_num, '--', color=colors[i], alpha=0.8, 
                       label=f'{method_name}')
            
            ax.set_xlabel('Čas t [h]')
            ax.set_ylabel('Temperatura T [°C]')
            ax.set_title(f'Začetna temperatura T(0) = {T0}°C')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Error analysis for different step sizes
        ax = axes[2]
        T0 = T0_1
        t_final = t_span[1]
        T_exact = self.analytical_solution(t_final, T0, k, T_zun)
        
        h_range = np.logspace(-2, 0, 20)
        methods = [('Euler', self.euler_method), ('Midpoint', self.midpoint_method),
                  ('Heun', self.heun_method), ('RK4', self.rk4_method)]
        
        for method_name, method_func in methods:
            errors = []
            for h in h_range:
                t_num, T_num = method_func(self.cooling_equation, t_span, T0, h, k, T_zun)
                error = abs(T_num[-1] - T_exact)
                errors.append(error)
            
            ax.loglog(h_range, errors, 'o-', label=method_name, markersize=4)
        
        ax.set_xlabel('Korak h [h]')
        ax.set_ylabel('Absolutna napaka pri t=50')
        ax.set_title('Odvisnost napake od velikosti koraka')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Study different k values
        ax = axes[3]
        k_values = [0.05, 0.1, 0.2, 0.5]
        T0 = T0_1
        
        for k_val in k_values:
            t_analytical = np.linspace(t_span[0], t_span[1], 1000)
            T_analytical = self.analytical_solution(t_analytical, T0, k_val, T_zun)
            ax.plot(t_analytical, T_analytical, label=f'k = {k_val}')
        
        ax.set_xlabel('Čas t [h]')
        ax.set_ylabel('Temperatura T [°C]')
        ax.set_title('Vpliv parametra k na ohlajanje')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('diffeq_plots/01_basic_cooling_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('diffeq_plots/01_basic_cooling_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def step_size_analysis(self):
        """Detailed analysis of step size dependence"""
        T0 = 21
        T_zun = -5
        k = 0.1
        t_span = (0, 50)
        
        plt.figure(figsize=(10, 6))
        
        h_range = np.logspace(-3, 0, 50)
        methods = [('Euler', self.euler_method), ('Midpoint', self.midpoint_method),
                  ('Heun', self.heun_method), ('RK4', self.rk4_method)]
        
        T_exact = self.analytical_solution(t_span[1], T0, k, T_zun)
        
        for method_name, method_func in methods:
            errors = []
            for h in h_range:
                t_num, T_num = method_func(self.cooling_equation, t_span, T0, h, k, T_zun)
                if len(T_num) > 0:
                    error = abs(T_num[-1] - T_exact)
                    errors.append(error)
                else:
                    errors.append(np.nan)
            
            plt.loglog(h_range, errors, 'o-', label=method_name, markersize=3)
        
        # Add theoretical convergence lines
        h_theory = np.logspace(-3, 0, 10)
        plt.loglog(h_theory, 10*h_theory, 'k--', alpha=0.5, label='O(h)')
        plt.loglog(h_theory, 10*h_theory**2, 'k-.', alpha=0.5, label='O(h²)')
        plt.loglog(h_theory, 10*h_theory**4, 'k:', alpha=0.5, label='O(h⁴)')
        
        plt.xlabel('Korak h [h]')
        plt.ylabel('Absolutna napaka')
        plt.title('Konvergenca numeričnih metod')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('diffeq_plots/02_step_size_dependence.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('diffeq_plots/02_step_size_dependence.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def study_periodic_heating(self):
        """Study the cooling equation with periodic heating"""
        # Parameters
        T0 = 21
        T_zun = -5
        k = 0.1
        delta = 10
        A_values = [1, 2, 5]  # Different heating amplitudes
        t_span = (0, 120)     # 5 days
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Compare methods for A=1
        ax = axes[0]
        A = 1
        h = 0.1
        
        methods = [
            ('Euler', self.euler_method),
            ('Midpoint', self.midpoint_method),
            ('RK4', self.rk4_method)
        ]
        
        colors = ['red', 'blue', 'green']
        for i, (method_name, method_func) in enumerate(methods):
            t_num, T_num = method_func(self.cooling_with_heating, t_span, T0, h, k, T_zun, A, delta)
            ax.plot(t_num, T_num, color=colors[i], label=method_name, alpha=0.8)
        
        ax.set_xlabel('Čas t [h]')
        ax.set_ylabel('Temperatura T [°C]')
        ax.set_title(f'Periodično segrevanje (A={A}) - primerjava metod')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Study different amplitudes
        ax = axes[1]
        method_func = self.rk4_method
        h = 0.1
        
        for A in A_values:
            t_num, T_num = method_func(self.cooling_with_heating, t_span, T0, h, k, T_zun, A, delta)
            ax.plot(t_num, T_num, label=f'A = {A}')
        
        ax.set_xlabel('Čas t [h]')
        ax.set_ylabel('Temperatura T [°C]')
        ax.set_title('Vpliv amplitude segrevanja')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Find maximum temperatures and their times
        ax = axes[2]
        A = 1
        t_num, T_num = method_func(self.cooling_with_heating, t_span, T0, h, k, T_zun, A, delta)
        
        # Find local maxima (peaks)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(T_num, height=15, distance=20)  # Minimum distance of 20 hours
        
        ax.plot(t_num, T_num, 'b-', label='Temperatura', linewidth=1.5)
        ax.plot(t_num[peaks], T_num[peaks], 'ro', markersize=6, label='Maksimumi')
        
        # Annotate some peaks
        for i, peak in enumerate(peaks[:3]):  # First 3 peaks
            ax.annotate(f'{T_num[peak]:.1f}°C\n@{t_num[peak]:.0f}h', 
                       (t_num[peak], T_num[peak]), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax.set_xlabel('Čas t [h]')
        ax.set_ylabel('Temperatura T [°C]')
        ax.set_title('Določanje maksimalnih temperatur')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Phase analysis - last 2 days to see steady state
        ax = axes[3]
        t_steady = t_num[t_num >= 72]
        T_steady = T_num[t_num >= 72]
        
        ax.plot(t_steady, T_steady, 'b-', linewidth=1.5)
        ax.set_xlabel('Čas t [h]')
        ax.set_ylabel('Temperatura T [°C]')
        ax.set_title('Ustaljeno stanje (zadnja 2 dneva)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('diffeq_plots/03_periodic_heating.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('diffeq_plots/03_periodic_heating.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print maximum temperature information
        print("\n" + "="*50)
        print("ANALIZA MAKSIMALNIH TEMPERATUR")
        print("="*50)
        if len(peaks) > 0:
            print(f"Število maksimumov: {len(peaks)}")
            print(f"Prvih 5 maksimumov:")
            for i, peak in enumerate(peaks[:5]):
                print(f"  {i+1}. Maksimum: {T_num[peak]:.2f}°C pri t = {t_num[peak]:.1f} h")
            
            avg_max_temp = np.mean(T_num[peaks])
            print(f"\nPovprečna maksimalna temperatura: {avg_max_temp:.2f}°C")
    
    def parameter_study(self):
        """Study influence of different parameters"""
        T0 = 21
        T_zun = -5
        t_span = (0, 72)  # 3 days
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Different amplitudes A
        ax = axes[0]
        k = 0.1
        delta = 10
        A_values = [0.5, 1, 2, 5]
        
        for A in A_values:
            t_num, T_num = self.rk4_method(self.cooling_with_heating, t_span, T0, 0.1, k, T_zun, A, delta)
            ax.plot(t_num, T_num, label=f'A = {A}')
        
        ax.set_xlabel('Čas t [h]')
        ax.set_ylabel('Temperatura T [°C]')
        ax.set_title('Vpliv amplitude segrevanja A')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Different phase shifts delta
        ax = axes[1]
        k = 0.1
        A = 1
        delta_values = [0, 6, 12, 18]
        
        for delta in delta_values:
            t_num, T_num = self.rk4_method(self.cooling_with_heating, t_span, T0, 0.1, k, T_zun, A, delta)
            ax.plot(t_num, T_num, label=f'δ = {delta} h')
        
        ax.set_xlabel('Čas t [h]')
        ax.set_ylabel('Temperatura T [°C]')
        ax.set_title('Vpliv faznega zamika δ')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Different k values with periodic heating
        ax = axes[2]
        A = 1
        delta = 10
        k_values = [0.05, 0.1, 0.2, 0.5]
        
        for k in k_values:
            t_num, T_num = self.rk4_method(self.cooling_with_heating, t_span, T0, 0.1, k, T_zun, A, delta)
            ax.plot(t_num, T_num, label=f'k = {k}')
        
        ax.set_xlabel('Čas t [h]')
        ax.set_ylabel('Temperatura T [°C]')
        ax.set_title('Vpliv parametra ohlajanja k')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Combined effect - temperature range vs parameters
        ax = axes[3]
        A_range = np.linspace(0.1, 5, 20)
        k_range = [0.05, 0.1, 0.2]
        delta = 10
        
        for k in k_range:
            temp_ranges = []
            for A in A_range:
                t_num, T_num = self.rk4_method(self.cooling_with_heating, (48, 72), T0, 0.1, k, T_zun, A, delta)
                if len(T_num) > 10:
                    temp_range = np.max(T_num) - np.min(T_num)
                    temp_ranges.append(temp_range)
                else:
                    temp_ranges.append(0)
            
            ax.plot(A_range, temp_ranges, 'o-', label=f'k = {k}', markersize=3)
        
        ax.set_xlabel('Amplituda A')
        ax.set_ylabel('Razpon temperature [°C]')
        ax.set_title('Razpon temperature v odvisnosti od A in k')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('diffeq_plots/04_parameter_study.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('diffeq_plots/04_parameter_study.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comprehensive_analysis(self):
        """Run all analyses"""
        print("Začenjam analizo osnovne enačbe ohlajanja...")
        self.compare_methods_basic_cooling()
        
        print("\nAnaliza odvisnosti od koraka...")
        self.step_size_analysis()
        
        print("\nZačenjam analizo periodičnega segrevanja...")
        self.study_periodic_heating()
        
        print("\nŠtudij parametrov...")
        self.parameter_study()

def main():
    solver = DifferentialEquationSolver()
    solver.run_comprehensive_analysis()

if __name__ == "__main__":
    main()
