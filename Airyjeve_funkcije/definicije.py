from numpy import pow, sqrt, exp, abs, pi, sin, cos
import mpmath as mp
from scipy.special import airy

# Ai(x) = a f(x) + b g(x) in Bi(x) = sqrt(3)[a f(x) - b g(x)]
# a = Ai(0) = Bi(0)/sqrt(3) in b = Ai'(0) = -Bi'(0)/sqrt(3)

a, b, _, _ = airy(0)

# Defining the "helper" functions

def f(x, n):
    sum = 1.0
    prod = 1.0
    for k in range(0, n):
        prod *= (3*k + 3)*(3*k + 2)
        sum += pow(x, 3*(k+1))/prod
    return sum


def g(x, n):
    sum = 0.0 + x 
    prod = 1.0
    for k in range(0, n):
        prod *= (3*k + 4)*(3*k + 3)
        sum += pow(x, 3*(k+1)+1)/prod
    return sum


def U(s): 
    """
    # This funciton is writen in such a way that it will be used recursivevly a_{n+1} = U_n a_n
    """
    return (3*s + 5/2)*(3*s + 1/2)/(18*(s+1))

def ksi(x):
    return 2/3 * pow(abs(x), 3/2)


def L(z, n):
    sum = 1.0
    prod = 1.0
    for s in range(0, n):
        prod *= U(s)
        sum += prod/pow(z, s+1)
    return sum

def P(z, n):
    sum = 1.0
    prod = 1.0
    for s in range(0, n):
        prod *= U(2*s+1)*U(2*s)
        sum += pow(-1, s+1) * prod/pow(z, 2*(s+1))
    return sum

def Q(z, n):
    sum = U(0)/z
    prod = U(0)
    for s in range(1, n):
        prod *= U(2*s-1)*U(2*s)
        sum += pow(-1, s) * prod/pow(z, 2*s+1)
    return sum 

# Defining the main functions


# Taylor series
def Ai_taylor(x, n):
    return a*f(x, n) + b*g(x, n)

def Bi_taylor(x, n):
    return sqrt(3)*(a*f(x, n) - b*g(x, n))


# Asyptotic series
def Ai_pos_asim(x, n):
    return exp(-ksi(x))/(2*sqrt(pi)*pow(x, 1/4)) * L(-ksi(x), n)

def Bi_pos_asim(x, n):
    return exp(ksi(x))/(sqrt(pi)*pow(x, 1/4)) * L(ksi(x), n)


# Oscialating series
def Ai_neg_asim(x, n):
    return (sin(ksi(x) - pi/4)*Q(ksi(x), n) + cos(ksi(x) - pi/4)*P(ksi(x), n))/(sqrt(pi)*pow(-x, 1/4))

def Bi_neg_asim(x, n):
    return (-sin(ksi(x) - pi/4)*Q(ksi(x), n) + cos(ksi(x) - pi/4)*P(ksi(x), n))/(sqrt(pi)*pow(-x, 1/4))
