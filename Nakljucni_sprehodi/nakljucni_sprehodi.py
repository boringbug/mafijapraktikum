import matplotlib.pyplot as plt
import numpy as np
from numpy import array, sin, cos, pi, random, sqrt, log
import matplotlib.pyplot as plt

def power_law_random(mu, size=1):
    """
    Generate random numbers from power law distribution p(l) ∝ l^(-μ)
    for 1 < μ < 3
    """
    u = random.uniform(0, 1, size)
    # Inverse transform sampling for power law
    return (1 - u)**(1/(1 - mu))

def levy_walk(N=100, mu=2.5, constant_velocity=True):
    """
    Simulate Lévy walk with power-law step length distribution
    
    Parameters:
    - N: number of steps
    - mu: power law exponent (1 < μ < 3)
    - constant_velocity: True for Lévy walk, False for Lévy flight
    """
    # Generate random directions and step lengths
    random_angle = random.uniform(0, 2*pi, N)
    
    # Generate step lengths from power law distribution
    step_lengths = power_law_random(mu, N)
    
    # Starting position
    positions = [array([0.0, 0.0])]
    times = [0.0]
    
    current_pos = array([0.0, 0.0])
    current_time = 0.0
    
    for i in range(N):
        # Calculate displacement
        step_vec = step_lengths[i] * array([cos(random_angle[i]), sin(random_angle[i])])
        
        # Update position
        current_pos = current_pos + step_vec
        positions.append(current_pos.copy())
        
        # Update time (different for walk vs flight)
        if constant_velocity:
            # Lévy walk: time proportional to step length
            current_time += step_lengths[i]  # assuming velocity = 1
        else:
            # Lévy flight: each step takes equal time
            current_time += 1.0
            
        times.append(current_time)
    
    return array(positions), array(times)

def analyze_diffusion(mu_values, num_walks=1000, steps=1000):
    """
    Analyze variance growth for different μ values
    """
    plt.figure(figsize=(12, 8))
    
    for mu in mu_values:
        print(f"Analyzing μ = {mu}")
        
        # Store variance at different times
        time_points = np.logspace(1, np.log10(steps), 20).astype(int)
        variances = []
        
        for t in time_points:
            final_positions = []
            
            for walk in range(num_walks):
                positions, times = levy_walk(t, mu=mu, constant_velocity=True)
                final_positions.append(positions[-1])
            
            final_positions = array(final_positions)
            
            # Calculate variance (using robust method as suggested)
            x_pos = final_positions[:, 0]
            y_pos = final_positions[:, 1]
            
            # Remove outliers for better variance estimation
            def remove_outliers(data, fraction=0.05):
                sorted_data = np.sort(data)
                n_remove = int(len(data) * fraction)
                return sorted_data[n_remove:-n_remove]
            
            x_clean = remove_outliers(x_pos)
            y_clean = remove_outliers(y_pos)
            
            variance = (np.var(x_clean) + np.var(y_clean)) / 2
            variances.append(variance)
        
        # Fit power law: variance ~ time^γ
        log_t = np.log(time_points)
        log_var = np.log(variances)
        
        # Linear fit in log-log space
        coeffs = np.polyfit(log_t, log_var, 1)
        gamma = coeffs[0]
        
        plt.loglog(time_points, variances, 'o-', label=f'μ={mu}, γ={gamma:.2f}')
        
        print(f"  Estimated γ = {gamma:.3f}")
        
        # Theoretical prediction
        if 1 < mu < 2:
            theoretical_gamma = 2.0
        elif 2 < mu < 3:
            theoretical_gamma = 4 - mu
        else:
            theoretical_gamma = 1.0
            
        print(f"  Theoretical γ = {theoretical_gamma:.1f}")
    
    plt.xlabel('Time (steps)')
    plt.ylabel('Variance')
    plt.title('Anomalous Diffusion: Variance Growth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot some example walks
def plot_example_walks():
    """Plot example walks for different numbers of steps"""
    step_counts = [10, 100, 1000, 10000]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, N in enumerate(step_counts):
        positions, times = levy_walk(N, mu=2.5)
        
        axes[i].plot(positions[:, 0], positions[:, 1], alpha=0.7)
        axes[i].plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
        axes[i].plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='End')
        axes[i].set_title(f'Lévy Walk with {N} steps (μ=2.5)')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run the analysis
if __name__ == "__main__":
    # 1. Plot example walks
    print("Plotting example walks...")
    plot_example_walks()
    
    # 2. Analyze diffusion for different μ values
    print("\nAnalyzing diffusion exponents...")
    mu_values = [1.5, 2.0, 2.5, 2.8]
    analyze_diffusion(mu_values, num_walks=500, steps=1000)
