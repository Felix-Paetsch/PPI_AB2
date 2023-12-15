"""

Autor: Felix Johannes Pätsch, Habib Eser
Datum: 17.11.23

This module visualizes the (absolute) sparcity of the matricies
used in the presented algorithm for d = 1,2,3 and
compares it to the sparcity (amt of non-zero entries) of a dense matrix.

"""

import matplotlib.pyplot as plt
from block_matrix import BlockMatrix

def plot_A_d_sparcity(): # pylint: disable=C0103
    """Plots the different sparcities of A^d for d = 1,2,3 and n from 5 to 1000
    """

    # Define the range of N
    input_values = range(5, 1000)
    plt.figure(figsize=(10, 6))

    # d = 1
    side_length_of_d1 = lambda n: (n - 1)
    non_zero_in_d1    = lambda n: n - 1 + 2*(n-2)
    ## Dense Matrix
    plt.loglog(input_values, \
        [side_length_of_d1(n) ** 2 for n in input_values], 'b--', label='d=1, Non sparse')
    ## Sparse Matrix
    plt.loglog(input_values, [non_zero_in_d1(n) for n in input_values], 'b'  , label='d=1, Sparse')

    # d = 2
    side_length_of_d2 = lambda n: (n-1)**2
    non_zero_in_d2    = lambda n: (n-1) * non_zero_in_d1(n) \
                            + 2 * (side_length_of_d2(n) - side_length_of_d1(n))
    ## Dense Matrix
    plt.loglog(input_values, \
        [side_length_of_d2(n) ** 2 for n in input_values], 'g--', label='d=2, Non sparse')
    ## Sparse Matrix
    plt.loglog(input_values, \
        [non_zero_in_d2(n)         for n in input_values]   , 'g'  , label='d=2, Sparse')

    # d = 3
    side_length_of_d3 = lambda n: (n-1)**3
    non_zero_in_d3    = lambda n: (n-1) * non_zero_in_d2(n) \
                            + 2 * (side_length_of_d3(n) - side_length_of_d2(n))
    ## Dense Matrix
    plt.loglog(input_values, \
        [side_length_of_d3(n) ** 2 for n in input_values], 'r--', label='d=3, Non sparse')
    ## Sparse Matrix
    plt.loglog(input_values, \
        [non_zero_in_d3(n)         for n in input_values] , 'r'  , label='d=3, Sparse')

    plt.xlabel('Number of Sample points N in each dimension')
    plt.ylabel('Non-zero Matrix entries')
    plt.title('Comparison of Functions on a Log Plot')
    plt.legend()

    plt.show()

def plot_lu_sparcity():
    """Plots the different sparcities for d = 1,2,3 and n from 5 to 20
    """

    # Define the range of N
    input_values = range(5, 20)
    plt.figure(figsize=(10, 6))

    # d = 1
    side_length_of_d1 = lambda n: (n - 1)
    non_zero_in_d1    = lambda n: n - 1 + 2*(n-2)
    ## LU Matrix
    plt.loglog(input_values, \
        [BlockMatrix(1,n).eval_sparsity_lu()[0] for n in input_values], 'b--', label='d=1, LU')
    ## Sparse Matrix
    plt.loglog(input_values, [non_zero_in_d1(n) for n in input_values], 'b'  , label='d=1, Sparse')

    # d = 2
    side_length_of_d2 = lambda n: (n-1)**2
    non_zero_in_d2    = lambda n: (n-1) * non_zero_in_d1(n) \
                            + 2 * (side_length_of_d2(n) - side_length_of_d1(n))
    ## LU Matrix
    plt.loglog(input_values, \
        [BlockMatrix(2,n).eval_sparsity_lu()[0] for n in input_values], 'g--', label='d=1, LU')
    ## Sparse Matrix
    plt.loglog(input_values, \
        [non_zero_in_d2(n)         for n in input_values]   , 'g'  , label='d=2, Sparse')

    # d = 3
    side_length_of_d3 = lambda n: (n-1)**3
    non_zero_in_d3    = lambda n: (n-1) * non_zero_in_d2(n) \
                            + 2 * (side_length_of_d3(n) - side_length_of_d2(n))
    ## LU Matrix
    plt.loglog(input_values, \
        [BlockMatrix(3,n).eval_sparsity_lu()[0] for n in input_values], 'r--', label='d=1, LU')
    ## Sparse Matrix
    plt.loglog(input_values, \
        [non_zero_in_d3(n)         for n in input_values] , 'r'  , label='d=3, Sparse')

    plt.xlabel('Anzahl an Gitterpunkten N in der jeweiligen Dimension')
    plt.ylabel('Nicht-Null Matrix Einträge')
    plt.title('Vergleich von Nicht-Null Einträgen in LU Zerlegung und sparse Matritzen $A^d$')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    plot_A_d_sparcity()
    plot_lu_sparcity()
