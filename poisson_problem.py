"""

Autor: Felix Johannes Pätsch, Habib Eser
Datum: 17.11.23

We implement some of the needed utilities to translate
the poisson_problem into the discrete version which uses liear algebra.

"""

import numpy as np
import matplotlib.pyplot as plt
from linear_solvers import solve_lu
from block_matrix   import BlockMatrix

# pylint: disable=invalid-name
# since most invalid names come from the specification

def rhs(d, n, f):
    """ Computes the right-hand side vector `b` for a given function `f`.
    Parameters
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is an array_like of `numpy`. The return value
        is a scalar.
    Returns
    -------
    numpy.ndarray
        Vector to the right-hand-side f.
    Raises
    ------
    ValueError
        If d < 1 or n < 2.
    """

    if d < 1 or n < 2:
        raise ValueError("Dimension d must be >= 1 and number of intervals n must be >= 2.")

    _res = []

    for i in range((n-1)**d):
        interval_positions = inv_idx(i + 1, d, n)
        interval_absolute_position = [x/n for x in interval_positions]
        _res.append(f(interval_absolute_position))

    return np.array(_res) / n ** 2

def idx(nx, n):
    """ Calculates the number of an equation in the Poisson problem for
    a given discretization point.
    Parameters
    ----------
    nx : list of int
        Coordinates of a discretization point, multiplied by n.
    n : int
        Number of intervals in each dimension.
    Return
    ------
    int
        Number of the corresponding equation in the Poisson problem.
    """
    result = 1
    for l, value in enumerate(nx):
        result += (value - 1) * (n - 1) ** l
    return result

def inv_idx(m, d, n):
    """ Calculates the coordinates of a discretization point for a
    given equation number of the Poisson problem.
    Parameters
    ----------
    m : int
        Number of an equation in the Poisson Problem
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    Return
    ------
    list of int
        Coordinates of the corresponding discretization point, multiplied by n.
    """
    coords = []
    m = m - 1
    for i in range(d, 0, -1):
        last_coord = m // (n - 1) ** (i - 1)
        coords.insert(0, last_coord + 1)
        m -= last_coord * (n - 1) ** (i - 1)
    return coords

def solve_poisson(d, n, f):
    """ Solves the discrete poisson problem in d dimensions with n
        evenly spaced sample points in each dimension:
            ∆u = -f, u = 0 on the boundary of [0,1]^d
    """

    matrix = BlockMatrix(d,n)
    p, l, u = matrix.get_lu()
    rhs_vec = rhs(d, n, f)
    return solve_lu(p, l, u, rhs_vec)

def compute_error(d, n, hat_u, u):
    """ Computes the error of the numerical solution of the Poisson problem
    with respect to the infinity-norm.

    Parameters
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intersections in each dimension
    hat_u : array_like of 'numpy'
        Finite difference approximation of the solution of the Poisson problem
        at the discretization points
    u : callable
        Solution of the Poisson problem
        The calling signature is 'u(x)'. Here 'x' is an array_like of 'numpy'.
        The return value is a scalar.

    Returns
    -------
    float
        maximal absolute error at the discretization points
    """

    u_column_vector_grid_points = rhs(d, n, lambda x: n * n * u(x))
    diff = hat_u - u_column_vector_grid_points
    return np.max(np.abs(diff))

def plot_error(u, f, d, n_range, step_size = 1):
    """
        f is solution to differential eqn
    """

    n_from = n_range[0]
    n_to   = n_range[1]

    # Define the range of N
    input_values = range(n_from, n_to, step_size)
    plot_values = []

    for n in input_values:
        u_approx = solve_poisson(d, n, f)
        err = compute_error(d, n, u_approx, u)
        plot_values.append(err)


    plt.figure(figsize=(10, 6))

    plt.plot(input_values, plot_values, 'b', label = "Label smth. smth.")

    plt.xlabel('Number of Sample points N in each dimension')
    plt.ylabel('Maximal Error of Approximation')
    plt.title('Dependency of maximal error on sample size')
    plt.legend()

    plt.show()

def sol_vector_to_matrix(v):
    """
    Converts a vector of length n^2 into an n x n matrix.

    Parameters
    ----------
    v : numpy.ndarray
        A vector of length n^2.

    Returns
    -------
    numpy.ndarray
        An n x n matrix.
    """
    n = int(np.sqrt(len(v)))  # Calculate n as the square root of the length of v
    matrix = np.reshape(v, (n, n))  # Reshape the vector into an n x n matrix
    return matrix

def plot_2d_function(g):
    """
    Plots a 2D function g on the domain [0, 1] x [0, 1].

    Parameters
    ----------
    g : callable
        A function that takes two arguments (x, y) and returns a value.
    """
    # Create a meshgrid for the domain [0, 1] x [0, 1]
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate the function g on the grid
    Z = g(X, Y)

    # Plotting
    plt.figure(figsize=(8, 6))
    #plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.contourf(X, Y, Z, levels=50, cmap='viridis', extent=[0, 1, 0, 1])
    plt.colorbar(label='Value of g(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Function Plot')
    plt.show()

def plot_2d_from_matrix(m):
    """
    Plots a 2D approximation given by a matrix m, where the values are
    assumed to be regularly spaced along [0,1]
    in both dimensions (not including the edges).

    Parameters
    ----------
    m : numpy.ndarray
        A 2D matrix representing the values of the function at regular intervals.
    """

    # The border should be 0
    rows, cols = m.shape
    new_matrix = np.zeros((rows + 2, cols + 2))
    new_matrix[1:-1, 1:-1] = m
    m = new_matrix

    # Assuming m is an n x n matrix, determine the number of intervals
    n = m.shape[0]

    # Create a meshgrid for the domain excluding the edges
    x = np.linspace(0,1, n)
    y = np.linspace(0,1, n)
    X, Y = np.meshgrid(x, y)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, m, levels=50, cmap='viridis')
    plt.colorbar(label='Approximated Value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Approximation Plot')
    plt.show()

def main():
    """ Demonstrates main functionality of this file
    """

    plot_2d_function(lambda x,y: x * np.sin(np.pi * x) * y * np.sin(np.pi * y))

    d = 2
    n = 4
    def _f(x):
        return \
            -1*(2 * np.pi * x[1] * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]) - np.pi**2 * x[0] * x[1] * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + \
            2 * np.pi * x[0] * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]) - np.pi**2 * x[1] * x[0] * np.sin(np.pi * x[1]) * np.sin(np.pi * x[0]))


    matrix = BlockMatrix(d, n)

    p, l, u = matrix.get_lu()
    u_approx = solve_lu(p, l, u, rhs(d, n, _f))
    m = sol_vector_to_matrix(u_approx)

    plot_2d_from_matrix(m)


if __name__ == "__main__":
    main()