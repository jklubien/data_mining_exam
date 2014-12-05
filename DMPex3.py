# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 13:09:01 2014

@author: Klubien
"""

##### 3.1 #####
import numpy as np

Y = 2 * np.loadtxt("tmp.txt")
#print Y

##### 3.2 #####
from numpy import *
A = array([[1, 0], [0, 0]])

#print rank(A)

def matrixrank(a, tol=None):
    """Compute the matrix rank.
    
    >>> matrixrank(array([[1, 0] [0, 0]]))
    1
    """
    U, s, V = np.linalg.svd(a, full_matrices=False)
    return np.count_nonzero(s)
    
##### 3.3 #####
import pylab, scipy.stats

X = pylab.standard_normal((10, 10000))
s = pylab.sum(X * X, axis=0)
(n, bins, patches) = pylab.hist(s, bins=100)

x = pylab.linspace(0, max(s), 100)
y = scipy.stats.chi2.pdf(x, 10) * 10000 * pylab.diff(bins)[0]
pylab.plot(x, y, "c", linewidth=5)
pylab.show()