import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
from scipy.special import airy
from definicije import Ai_taylor, Ai_pos_asim, Ai_neg_asim, Bi_taylor, Bi_pos_asim, Bi_neg_asim

# Set mpmath precision
mp.mp.dps = 50  # 50 decimal digits precision

# Create x values as regular arrays for plotting
x = np.linspace(-20, 20, 1000)  # Reduced points for performance
x_pos = np.linspace(5, 40, 1000)  # Reduced upper limit for better visualization
x_neg = np.linspace(-25, -20, 1000)

x_line = np.linspace(-50, 10, 1000)
line = [1e-10 for i in x_line]

# This function calculates the absolute difference of the Ai funciton with mpmath
def calculate_with_mpmath_ai():
    """Calculate differences using mpmath for high precision"""
    
    # Convert arrays to lists and compute with mpmath
    diff_taylor = []
    for xi in x:
        # Compute with mpmath and convert to float for comparison
        mp_val = Ai_taylor(mp.mpf(str(xi)), 100)
        reference_val = mp.airyai(xi)
        diff_taylor.append(float(abs(mp_val - reference_val)))
    
    diff_pos_asim = []
    for xi in x_pos:
        mp_val = Ai_pos_asim(mp.mpf(str(xi)), 100)
        reference_val = mp.airyai(xi)
        diff_pos_asim.append(float(abs(mp_val - reference_val)))
    
    diff_neg_asim = []
    for xi in x_neg:
        mp_val = Ai_neg_asim(mp.mpf(str(xi)), 100)
        reference_val = mp.airyai(xi)
        diff_neg_asim.append(float(abs(mp_val - reference_val)))
    
    return diff_taylor, diff_pos_asim, diff_neg_asim

# This function calculates the absolute difference of the Bi funciton with mpmath
def calculate_with_mpmath_bi():
    """Calculate differences using mpmath for high precision"""
    
    # Convert arrays to lists and compute with mpmath
    diff_taylor = []
    for xi in x:
        # Compute with mpmath and convert to float for comparison
        mp_val = Bi_taylor(mp.mpf(str(xi)), 100)
        reference_val = mp.airybi(xi)
        diff_taylor.append(float(abs(mp_val- reference_val)))
    
    diff_pos_asim = []
    for xi in x_pos:
        mp_val = Bi_pos_asim(mp.mpf(str(xi)), 50)
        reference_val = mp.airybi(xi)
        diff_pos_asim.append(float(abs(mp_val - reference_val)))
    
    diff_neg_asim = []
    for xi in x_neg:
        mp_val = Bi_neg_asim(mp.mpf(str(xi)), 100)
        reference_val = mp.airybi(xi)
        diff_neg_asim.append(float(abs(mp_val - reference_val)))
    
    return diff_taylor, diff_pos_asim, diff_neg_asim

# This function calculates the realtive difference of the Ai funciton with mpmath
def calculate_with_mpmath_ai_relative():
    """Calculate differences using mpmath for high precision"""
    
    # Convert arrays to lists and compute with mpmath
    diff_taylor = []
    for xi in x:
        # Compute with mpmath and convert to float for comparison
        mp_val = Ai_taylor(mp.mpf(str(xi)), 100)
        reference_val = mp.airyai(xi)
        diff_taylor.append(float(abs((mp_val - reference_val) / reference_val)))
    
    diff_pos_asim = []
    for xi in x_pos:
        mp_val = Ai_pos_asim(mp.mpf(str(xi)), 100)
        reference_val = mp.airyai(xi)
        diff_pos_asim.append(float(abs((mp_val - reference_val) / reference_val)))
    
    diff_neg_asim = []
    for xi in x_neg:
        mp_val = Ai_neg_asim(mp.mpf(str(xi)), 100)
        reference_val = mp.airyai(xi)
        diff_neg_asim.append(float(abs((mp_val - reference_val) / reference_val)))
    
    return diff_taylor, diff_pos_asim, diff_neg_asim

# This function calculates the realtive difference of the Bi funciton with mpmath
def calculate_with_mpmath_bi_relative():
    """Calculate differences using mpmath for high precision"""
    
    # Convert arrays to lists and compute with mpmath
    diff_taylor = []
    for xi in x:
        # Compute with mpmath and convert to float for comparison
        mp_val = Bi_taylor(mp.mpf(str(xi)), 100)
        reference_val = mp.airybi(xi)
        diff_taylor.append(float(abs((mp_val - reference_val) / reference_val)))
    
    diff_pos_asim = []
    for xi in x_pos:
        mp_val = Bi_pos_asim(mp.mpf(str(xi)), 50)
        reference_val = mp.airybi(xi)
        diff_pos_asim.append(float(abs((mp_val - reference_val) / reference_val)))
    
    diff_neg_asim = []
    for xi in x_neg:
        mp_val = Bi_neg_asim(mp.mpf(str(xi)), 100)
        reference_val = mp.airybi(xi)
        diff_neg_asim.append(float(abs((mp_val - reference_val) / reference_val)))
    
    return diff_taylor, diff_pos_asim, diff_neg_asim
# Calculate the differences
print("Calculating differences with mpmath...")
diff_taylor_ai, diff_pos_asim_ai, diff_neg_asim_ai = calculate_with_mpmath_ai()
diff_taylor_bi, diff_pos_asim_bi, diff_neg_asim_bi = calculate_with_mpmath_bi()
diff_taylor_ai_rel, diff_pos_asim_ai_rel, diff_neg_asim_ai_rel = calculate_with_mpmath_ai_relative()
diff_taylor_bi_rel, diff_pos_asim_bi_rel, diff_neg_asim_bi_rel = calculate_with_mpmath_bi_relative()


# Semi-log plot for absolute differnece Ai
plt.figure(figsize=(12, 8))
plt.title("Absolutna napaka aproksimacij v semi-log Ai")
plt.semilogy(x, diff_taylor_ai, label='Maclauronova vrsta')
plt.semilogy(x_pos, diff_pos_asim_ai, label='Pozitivna asimptotska vrsta')
plt.semilogy(x_neg, diff_neg_asim_ai, label='Negativna asimptotska vrsta')
plt.plot(x_line, line, color='red', linestyle='--', label='1e-10 threshold')
plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('Razlika')
plt.savefig("ai_abs.pdf", dpi=300)
plt.show()

# Semi-log plot for absolute differnece Bi
plt.figure(figsize=(12, 8))
plt.title("Absolutna napaka aproksimacij v semi-log Bi")
plt.semilogy(x, diff_taylor_bi, label='Maclauronova vrsta')
plt.semilogy(x_pos, diff_pos_asim_bi, label='Pozitivna asimptotska vrsta')
plt.semilogy(x_neg, diff_neg_asim_bi, label='Negativna asimptotska vrsta')
plt.plot(x_line, line, color='red', linestyle='--', label='1e-10 threshold')
plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('Razlika')
plt.savefig("bi_abs.pdf", dpi=300)
plt.show()

# Semi-log plot for relative differnece Ai
plt.figure(figsize=(12, 8))
plt.title("Relativna napaka aproksimacij v semi-log Ai")
plt.semilogy(x, diff_taylor_ai_rel, label='Maclauronova vrsta')
plt.semilogy(x_pos, diff_pos_asim_ai_rel, label='Pozitivna asimptotska vrsta')
plt.semilogy(x_neg, diff_neg_asim_ai_rel, label='Negativna asimptotska vrsta')
plt.plot(x_line, line, color='red', linestyle='--', label='1e-10 threshold')
plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('Razlika')
plt.savefig("ai_rel.pdf", dpi=300)
plt.show()

# Semi-log plot for relativ differnece Bi
plt.figure(figsize=(12, 8))
plt.title("Relativna napaka aproksimacij v semi-log Bi")
plt.semilogy(x, diff_taylor_bi_rel, label='Maclauronova vrsta')
plt.semilogy(x_pos, diff_pos_asim_bi_rel, label='Pozitivna asimptotska vrsta')
plt.semilogy(x_neg, diff_neg_asim_bi_rel, label='Negativna asimptotska vrsta')
plt.plot(x_line, line, color='red', linestyle='--', label='1e-10 threshold')
plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('Razlika')
plt.savefig("bi_rel.pdf", dpi=300)
plt.show()
