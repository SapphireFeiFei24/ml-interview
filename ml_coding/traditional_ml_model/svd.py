import numpy as np

def SVD:
    """
    Inuition:
    A = UEV^T
    U (M*M): rotates coordinates
    E (M*N): diagonal with non-negative singular values, scalar
    V (N*N): rotates coordinates

    Some math facts:
    * Relatoinship to eigen-decomposition
    * Rank: number of non-zero singular values
    * Best rank-k approx
    * Condition number
    * Sensitivity: small singular values -> unstable inverse/pseudoinverse

    Algorithms to know
    * Exact SVD(dense, small-medium matrices): numpy.linalg.svd - O(min(mn^2, m^2n)
    * Truncated/Partial SVD via eigendecomposition of A^TA or scipy.sparse.linalg.svds for sparse large matrices
    * Power iteration/Lanczos: compute top singular vector iteractively
    * Randomized SVD - fast to practical for very large matrices O(mnlogk + k^2 (m+n))
    """
    def __init__(self):
        pass

    def truncated_svd(A, k):
        """
        Return U_k, S_k, Vt_k for top-k SVD using numpy
        :param A: (M, N)
        :param k:  <= min(M, N)
        :return:
        """

        # full_matrics=False gives U shape (m,r), Vt shape(r, n) where r = min(m,n)
        # so slicing is simple
        U, s, Vt = np.linalg.svd(A, full_matrices=False) #economical SVD
        return U[:, :k], s[:k], Vt[:k, :]