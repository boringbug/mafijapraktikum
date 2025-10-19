import mpmath as mp
from mpmath import power, sqrt, exp, fabs, pi, sin, cos

# Ai(x) = a f(x) + b g(x) in Bi(x) = sqrt(3)[a f(x) - b g(x)]
# a = Ai(0) = Bi(0)/sqrt(3) in b = Ai'(0) = -Bi'(0)/sqrt(3)

# Use mpmath for the constants
a = mp.airyai(0)
b = mp.airyai(0, derivative=1)

# Defining the "helper" functions

def f(x, n):
    sum_val = mp.mpf(1.0)
    prod = mp.mpf(1.0)
    for k in range(0, n):
        prod *= (3*k + 3)*(3*k + 2)
        sum_val += power(x, 3*(k+1))/prod
    return sum_val

def g(x, n):
    sum_val = x  # 0.0 + x
    prod = mp.mpf(1.0)
    for k in range(0, n):
        prod *= (3*k + 4)*(3*k + 3)
        sum_val += power(x, 3*(k+1)+1)/prod
    return sum_val

def U(s): 
    """
    # This function is written in such a way that it will be used recursively a_{n+1} = U_n a_n
    """
    s_mp = mp.mpf(s)
    return (3*s_mp + mp.mpf(5)/2)*(3*s_mp + mp.mpf(1)/2)/(18*(s_mp+1))

def ksi(x):
    return mp.mpf(2)/3 * power(fabs(x), mp.mpf(3)/2)

def L(z, n):
    sum_val = mp.mpf(1.0)
    prod = mp.mpf(1.0)
    for s in range(0, n):
        prod *= U(s)
        sum_val += prod/power(z, s+1)
    return sum_val

def P(z, n):
    sum_val = mp.mpf(1.0)
    prod = mp.mpf(1.0)
    for s in range(0, n):
        prod *= U(2*s+1)*U(2*s)
        sum_val += power(-1, s+1) * prod/power(z, 2*(s+1))
    return sum_val

def Q(z, n):
    sum_val = U(0)/z
    prod = U(0)
    for s in range(1, n):
        prod *= U(2*s-1)*U(2*s)
        sum_val += power(-1, s) * prod/power(z, 2*s+1)
    return sum_val 

# Defining the main functions

# Taylor series
def Ai_taylor(x, n):
    return a*f(x, n) + b*g(x, n)

def Bi_taylor(x, n):
    return mp.sqrt(3)*(a*f(x, n) - b*g(x, n))

# Asymptotic series
def Ai_pos_asim(x, n):
    return exp(-ksi(x))/(2*sqrt(pi)*power(x, mp.mpf(1)/4)) * L(-ksi(x), n)

def Bi_pos_asim(x, n):
    return exp(ksi(x))/(sqrt(pi)*power(x, mp.mpf(1)/4)) * L(ksi(x), n)

# Oscillating series
def Ai_neg_asim(x, n):
    x_abs = fabs(x)
    ksi_val = ksi(x_abs)
    return (sin(ksi_val - pi/4)*Q(ksi_val, n) + cos(ksi_val - pi/4)*P(ksi_val, n))/(sqrt(pi)*power(x_abs, mp.mpf(1)/4))

def Bi_neg_asim(x, n):
    x_abs = fabs(x)
    ksi_val = ksi(x_abs)
    return (-sin(ksi_val - pi/4)*P(ksi_val, n) + cos(ksi_val - pi/4)*Q(ksi_val, n))/(sqrt(pi)*power(x_abs, mp.mpf(1)/4))
