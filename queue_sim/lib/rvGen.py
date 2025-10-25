""" Functions to sample from common probability distributions.

This module contains two types of routines:

1. gen functions which return a lambda. On evaluation, they return a value
from the specified distribution. Useful if you want to generate many values
using the same distribution or avoid passing in the parameter to helper functions.

Example usage:
exponential = genExp(1.5) # a function of type void -> float
random_val = exponential() # a float from the Exp(1.5) distribution

2. non-gen functions which directly return a value according to the distribution

Example usage:
random_val = Uniform(2,5) # a float distributed Uniform(2,5)
"""

import math
import random
from typing import Callable

# Generate X ~ Exponential(mu), E[X] = 1/mu
def genExp(mu : float) -> Callable[[], float] :
    return lambda: -(1/mu)*math.log(1-random.random())

# X ~ Unif(a,b)
def genUniform(a : float, b : float) -> Callable[[], float]:
    d = b - a
    return lambda: d*random.random() + a

# X ~ BoundedPareto(k, p, alpha)
def genBoundedPareto(k, p, alpha):
    C = (k**alpha)/(1-(k/p)**alpha)
    return lambda:(-random.random()/C + k**(-alpha))**(-1/alpha)

def genBernoulli(p):
    return lambda: 1 if (random.random() <= p) else 0

def Uniform(a,b):
    d = b - a
    return d*random.random() + a

def BoundedPareto(k, p, alpha):
    return genBoundedPareto(k,p,alpha)()

def Bernoulli(p):

    genBernoulli(p)()
