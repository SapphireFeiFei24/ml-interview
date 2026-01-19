import numpy as np

"""
Notes:
* Computing A.T@A doubles the sensitivity of the problem.
The condition number gets squared, so numerical errors explode.
That's why algorithms like SVD and QR avoid forming A.T@A
"""
class SVD:
    """
    Intuition:
    A = UEV^T
    U (M*M): rotates coordinates
        -> Directions of output space(row space)
        -> Principal component scores
        -> Rcmd: latent user factors
    E (M*N): diagonal with non-negative singular values, scalar
        -> importance of each direction
    V (N*N): rotates coordinates
        -> rotates the input vector into a basis where A acts independently along each axis
        -> same as PCA's principal directions
        -> Rcmd: latent item factors

    Some math facts:
    * Relationship to eigen-decomposition
    * Rank: number of non-zero singular values
    * Best rank-k approx
    * Condition number
    * Sensitivity: small singular values -> unstable inverse/pseudoinverse

    Algorithms to know
    * Exact SVD(dense, small-medium matrices): numpy.linalg.svd - O(min(mn^2, m^2n)
    * Truncated/Partial SVD via eigen-decomposition of A^TA or scipy.sparse.linalg.svds for sparse large matrices
    * Power iteration/Lanczos: compute top singular vector interactively
    * Randomized SVD - fast to practical for very large matrices O(mnlogk + k^2 (m+n))
    """
    def __init__(self):
        pass

    def truncated_svd(A, k):
        """
        O(min(mn^2, m^2n)) Good for small to medium matrices
        Return U_k, S_k, Vt_k for top-k SVD using numpy
        :param A: (M, N)
        :param k:  <= min(M, N)
        :return:
        """

        # full_matrics=False gives U shape (m,r), Vt shape(r, n) where r = min(m,n)
        # so slicing is simple
        U, s, Vt = np.linalg.svd(A, full_matrices=False)  # economical SVD
        return U[:, :k], s[:k], Vt[:k, :]

    def top1_svd_power(self, A, num_iters=100, tol=1e-6):
        """
        Power method to find top singular triplet (u, sigma, v) of A
        Each iteration costs O(mn). Converges faster if spectral gap sigma1/sigma2 is large
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
        """
        Compute top-k SVD using subspace iteration (power method on block).
        O(mn* (k + p)) much cheaper when k << min(m, n)
        :param A: (m, n)
        :param k:
        :param num_iters:
        :return: U_k, S_k, Vt_k
        """

        m, n = A.shape
        # approximate top-k right singular subspace
        Y = np.random.randn(n, k)

        # iterate y <- (A.T A) Y
        for _ in range(num_iters):
            Z = A.T @ (A @ Y)  # avoid A.T @ A for stability
            # orthonormalize columns (QR)
            Y, _ = np.linalg.qr(Y)

        # project A on subspace
        B = A @ Y

        # compute SVD of small matrix B (m x k)
        Ub, s, Vtb = np.linalg.svd(B, full_matrices=False)
        U_k = Ub[:, :k]
        S_k = s[:k]
        V_k = Y @ Vtb.T[:, :k]  # shape n x k
        return U_k, S_k, V_k.T


class PCA:
    """
    Computed through SVD of the mean-centered data matrix
    """
    def __init__(self):
        self.svd = SVD()

    def reduce(self, X):
        """
        Dimensionality reduction -> map into a 2D plane
        :param X: (n_sample, n_features)
        :return: (n_sample, 2)
        """
        X_centered = X - X.mean(axis=0)
        # U: (n_sample, 2) S: (2,) Vt: (2, n_features)
        U, S, Vt = self.svd.truncated_svd(X_centered, 2)
        PCs = Vt.T
        return X_centered @ PCs


class CollaborativeFiltering:
    def __init__(self):
        self.svd = SVD()

        # demo (n_users, n_items)
        self.ratings = np.array([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [0, 0, 5, 4],
            [0, 1, 5, 4]], dtype=float)

    def _preprocess(self, ratings):
        # fill na
        mean = np.mean(ratings[ratings>0])
        rating_filled = np.where(ratings == 0, mean, ratings)
        return rating_filled

    def predict(self, ratings, top_k):
        # preprocess
        ratings = self._preprocess(ratings)

        # SVD
        U, s, Vt = self.svd.truncated_svd(ratings, top_k)
        S = np.diag(s)

        predictions = U @ S @ Vt
        return predictions

    def demo(self, top_k_rcmd):
        predictions = self.predict(self.ratings)
        # get max
        # return np.argmax(predictions, axis=1)
        # top k
        # partition
        return predictions
