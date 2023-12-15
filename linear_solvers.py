"""

Autor: Felix Johannes Pätsch, Habib Eser
Datum: 14.12.23

Solves Ax = b given LU-decomposition A = p*l*u
with forward and backward substitution.

"""


import numpy as np

# pylint: disable=C0103
# # p,l,u,etc variable names not liked

def solve_lu(p, l, u, b):
    """
    Solves the linear system Ax = b via forward and backward substitution
    given the decomposition A = p * l * u.

    Parameters
    ----------
    p : numpy.ndarray
        Permutation matrix of LU decomposition.
    l : numpy.ndarray
        Lower triangular matrix of LU decomposition.
    u : numpy.ndarray
        Upper triangular matrix of LU decomposition.
    b : numpy.ndarray
        Vector of the right-hand side of the linear system.

    Returns
    -------
    x : numpy.ndarray
        Solution of the linear system.
    """
    # Permuting b according to permutation matrix p
    # p.T = p inverse
    permuted_b = np.dot(p.T, b)

    # Forward substitution to solve Ly = permuted_b
    y = np.zeros_like(permuted_b)
    for i, value in enumerate(permuted_b):
        y[i] = value - np.dot(l[i, :i], y[:i])

    # Backward substitution to solve Ux = y
    x = np.zeros_like(y)
    for i in reversed(range(len(permuted_b))):
        x[i] = (y[i] - np.dot(u[i, i+1:], x[i+1:])) / u[i, i]

    return x
