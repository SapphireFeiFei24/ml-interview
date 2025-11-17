import numpy as np

class SVD:
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
        O(min(mn^2, m^n)) Good for small to medium matrices
        Return U_k, S_k, Vt_k for top-k SVD using numpy
        :param A: (M, N)
        :param k:  <= min(M, N)
        :return:
        """

        # full_matrics=False gives U shape (m,r), Vt shape(r, n) where r = min(m,n)
        # so slicing is simple
        U, s, Vt = np.linalg.svd(A, full_matrices=False) #economical SVD
        return U[:, :k], s[:k], Vt[:k, :]

    def top1_svd_power(self, A, num_iters=100, tol=1e-6):
        """
        Power method to find top singular triplet (u, sigma, v) of A
        :param A: (m, n)
        :param num_iters:
        :param tol:
        :return:
        """
        m, n = A.shape
        # init v randomly
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        last_sigma = 0.0

        for i in range(num_iters):
            # w = A^T (A v)
            Av = A @ v  # (m,)
            w = A.T @ Av  # (n,)
            norm_w = np.linalg.norm(w)
            if norm_w == 0:
                break
            v = w / norm_w
            sigma = np.linalg.norm(Av)
            if abs(sigma - last_sigma) < tol:
                break
            last_sigma = sigma

        # compute u
        Av = A @ v
        sigma = np.linalg.norm(Av)
        if sigma == 0:
            u = np.zeros(m)
        else:
            u = Av / sigma
        return u, sigma, v

    def topk_svd_subspace(self, A, k, num_iters=20):
        pass