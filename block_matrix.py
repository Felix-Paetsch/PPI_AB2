"""

Autor: Felix Johannes Pätsch, Habib Eser
Datum: 17.11.23

We implement the structure BlockMatrix which will be used to
solve the discrete version of the Poisson problem.
It utilizes the sparce matrix structure from the algorithm for memory efficency.

"""

import numpy as np
import scipy.sparse as sp # pylint: disable=import-error
import scipy.linalg as la # pylint: disable=import-error

# pylint: disable=invalid-name
# since most invalid names come from the specification

class BlockMatrix:
    """ Represents block matrices arising from finite difference approximations
    of the Laplace operator.

    Parameters
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.

    Attributes
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.

    Raises
    ------
    ValueError
        If d < 1 or n < 2.
    """

    def __init__(self, d, n):
        if d < 1 or n < 2:
            raise ValueError("Dimension d must be >= 1 and number of intervals n must be >= 2.")

        self.d = d
        self.n = n

        self.matrix = self.get_sparse()

    def print_matrix_dense(self):
        """ Prints the block matrix in a non-sparse (dense) format. """
        dense_matrix = self.matrix.toarray()
        print(dense_matrix)

    def get_sparse(self):
        """ Returns the block matrix as sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            The block_matrix in a sparse data format.
        """

        return self._create_matrix_dd(self.d)

    def _create_matrix_1d(self):
        diagonals = [-np.ones(self.n-2), 2*self.d*np.ones(self.n - 1), -np.ones(self.n-2)]
        return sp.diags(diagonals, offsets=[-1, 0, 1], format='csr')

    def _create_matrix_dd(self, d):
        if d == 1:
            return self._create_matrix_1d()

        # Create the main diagonal block
        main_block = self._create_matrix_dd(d - 1)

        # Create negative identity matrices for off-diagonals
        identity_size = main_block.shape[0]
        off_diagonal_block = -sp.eye(identity_size, format='csr')

        # Assemble the main diagonal blocks
        main_diagonal = [main_block] * (self.n - 1)

        # Create the large block matrix using block_diag
        large_matrix = sp.block_diag(main_diagonal, format='csr')

        for i in range(1, self.n - 1):
            start_idx = i * identity_size
            large_matrix[
                start_idx:start_idx + identity_size,
                start_idx - identity_size:start_idx
            ] = off_diagonal_block
            large_matrix[
                start_idx - identity_size:start_idx,
                start_idx:start_idx + identity_size
            ] = off_diagonal_block

        return large_matrix

    def eval_sparsity(self):
        """ Returns the absolute and relative numbers of non-zero elements of
        the matrix. The relative quantities are with respect to the total
        number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """

        total_elements = self.matrix.shape[0] * self.matrix.shape[1]
        non_zeros = self.matrix.nnz
        relative_non_zeros = non_zeros / total_elements
        return non_zeros, relative_non_zeros

    def get_lu(self):
        """ Provides an LU-Decomposition of the represented matrix A of the
        form A = p * l * u

        Returns
        -------
        p : numpy.ndarray
            permutation matrix of LU-decomposition
        l : numpy.ndarray
            lower triangular unit diagonal matrix of LU-decomposition
        u : numpy.ndarray
            upper triangular matrix of LU-decomposition
        """
        p, l, u = la.lu(self.matrix.toarray()) # pylint: disable=W0632
        return p, l, u

    def eval_sparsity_lu(self):
        """ Returns the absolute and relative numbers of non-zero elements of
        the LU-Decomposition. The relative quantities are with respect to the
        total number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """

        P, L, U = self.get_lu()
        zeros_L = np.sum(L == 0)
        zeros_U = np.sum(U == 0)
        matrix_dim = P.shape[0]
        total_zeros = zeros_L + zeros_U - (matrix_dim ** 2 - matrix_dim)

        abs_non_zero = matrix_dim * matrix_dim - total_zeros
        rel_non_zero = abs_non_zero / (matrix_dim * matrix_dim)
        return (abs_non_zero, rel_non_zero)
