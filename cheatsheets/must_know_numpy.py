import numpy
import numpy as np


def array_creation_shapes():
    print("np.array", np.array([1, 2, 3]))
    print("np.zeros", np.zeros([1, 2, 3]))
    print("np.ones", np.ones((2, 3)))
    print("np.flatten", np.ones((2, 3)).flatten())
    print("np.random.randn", np.random.randn(2,3))
    print("np.random.randint", np.random.randint(high=4, low=1, size=(3,2)))
    print("[:, new_axis]", np.ones((2, 4))[:, np.newaxis].shape)
    print("[new_axis, :]", np.ones((2, 4))[np.newaxis, :].shape)


def broad_casting():
    print(f"[array_with_scala]array:{np.ones(3)} * scalar:{5}", np.ones(3) * 5)

    origin = np.random.randn(2,3)
    mean = np.mean(origin, axis=0)
    res = origin-mean
    print(f"[array with array]array({origin.shape}) - array({mean.shape}) = result({res.shape})", res)

    origin2 = np.random.randn(4, 3)
    res = origin2[:, np.newaxis] - origin
    print(f"[array_with_different_dims] o2({origin2.shape})[:, new_axis] - o({origin.shape}) = result({res.shape})")


def vectorized_distance_computation():
    n, k = 5, 3
    vec1 = np.random.randn(n, 2)
    vec2 = np.random.randn(k, 2)
    dist = np.linalg.norm(vec1[:, None, :] - vec2[None, :, :], axis=2)
    print(f"dist: from vec1({vec1.shape}) to vec2({vec2.shape}) is ({dist.shape})", dist)


def matrix_multiplication():
    n, m, k = 3, 2, 1
    m1 = np.random.randn(n, m)
    m2 = np.random.randn(m, k)

    print(f"matmul: m1({m1.shape}) @ m2({m2.shape}) = m3({(m1 @ m2).shape})")


def linear_systems():
    A = np.random.rand(2, 2)
    b = np.random.rand(2, 2)
    print(f"solve Ax=b: A:{A} b:{b} x:{np.linalg.solve(A, b)}")
    print(f"solve Ax=b: A:{A} b:{b} x=A-1@b", np.linalg.inv(A) @ b)


def dim_reductions():
    x = np.random.normal(loc=0, scale=1, size=(2,4))
    print(f"sum of x({x.shape}) by axis 0", np.sum(x, axis=0))
    print(f"mean of x({x.shape}) by axis 1", np.mean(x, axis=1))
    print(f"max of x({x.shape} but keep origin_dim", np.max(x, axis=0, keepdims=True))
    print(f"arg of x({x.shape}) {x} by axis 0", np.argmax(x, axis=1))
    print(f"clip of {x} by 0.5", np.clip(x, min=-0.5, max=0.5))


def boolean_indexing():
    x = np.array([[1,2,3], [4,5,6]])
    y = np.array([0, 1])
    mask = (y == 1)
    print(f"mask:{mask}")
    print(f"x[mask]: {x[mask]}")
    print(f"x[~mask]: {x[~mask]}")


def concatenate():
    A = np.random.randn(2, 3)
    B = np.random.randn(2, 3)
    print(f"concat by dim0 A({A.shape}, B({B.shape})", np.concatenate([A, B], axis=0).shape)
    print(f"vstack A({A.shape}, B({B.shape})", np.vstack([A, B]).shape)
    print(f"hstack A({A.shape}, B({B.shape})", np.hstack([A, B]).shape)
    print(f"np.c_ A({A.shape}, B({B.shape})", np.c_[A, B].shape)
    print(f"np.r_ A({A.shape}, B({B.shape})", np.r_[A, B].shape)


def sort():
    input = np.random.randn(10)
    print(f"argsort for {input} by axis=0:", np.argsort(input))
    k=3
    print(f"top_{k} index:", np.argpartition(input, kth=-k)[-k:])


if __name__ == "__main__":
    array_creation_shapes()
    broad_casting()
    vectorized_distance_computation()
    matrix_multiplication()
    linear_systems()
    dim_reductions()
    boolean_indexing()
    concatenate()
    sort()