"""Problem 7.4: Two-dimensional scaled copy using TileLang (fill in the blanks).

Y = 2 * X, where X has shape (M, N). Neither M nor N is guaranteed to be
divisible by the tile dimensions, similar to Problem 2.6.

This time, first move the tile into shared memory, perform the calculation,
and then write it back. Data movement is handled by T.copy.

After completing it, compare it with Problem 2.6 and consider the blanks
related to row and column indices, boundary protection, and grid dimensions:
which concepts still have a direct equivalent here, and which are handled
automatically by T.copy?

This requires a GPU and TileLang (`uv sync --extra tilelang`).
Run the following command on the cluster:

    pytest tests/test_tilelang.py -k copy2d
"""

import tilelang
import tilelang.language as T


def make_scale2d(M, N, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func
    def scale2d(
        X: T.Buffer((M, N), dtype),
        Y: T.Buffer((M, N), dtype),
    ):
        # ===== Blank 1: Create a two-dimensional CTA grid, as in 7.3.
        # The x direction handles the N columns, while the y direction
        # handles the M rows.
        # Hint: T.ceildiv =====
        m_blocks = T.ceildiv(M, block_M)
        n_blocks = T.ceildiv(N, block_N)

        with T.Kernel(n_blocks, m_blocks, threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_N), dtype)

            # ===== Blank 2: Copy the current tile from X into shared memory.
            # Hint: T.copy(X[row starting position, column starting position],
            #              X_shared)
            # T.copy automatically handles the out-of-bounds portion. =====
            T.copy(X[by*block_M, bx*block_N], X_shared)

            for i, j in T.Parallel(block_M, block_N):
                X_shared[i, j] = X_shared[i, j] * 2.0

            # ===== Blank 3: Copy the calculated tile back to the same
            # position in Y. =====
            T.copy(X_shared, Y[by*block_M, bx*block_N])

    return scale2d