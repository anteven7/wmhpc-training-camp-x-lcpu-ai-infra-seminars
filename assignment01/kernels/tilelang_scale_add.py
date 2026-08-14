"""Problem 7.3: Scale-add using TileLang (fill in the blanks).

Y = 2 * X + 1, where X has shape (M, N).

The two blanks correspond to two basic TileLang operations.

This requires a GPU and TileLang (`uv sync --extra tilelang`).
Run the following command on the cluster:

    pytest tests/test_tilelang.py -k scale_add
"""

import tilelang
import tilelang.language as T


def make_scale_add(M, N, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func
    def scale_add(
        X: T.Buffer((M, N), dtype),
        Y: T.Buffer((M, N), dtype),
    ):
        # ===== Blank 1: Create a two-dimensional CTA grid.
        # How many blocks are needed in the x direction, which handles
        # the N columns? How many are needed in the y direction, which
        # handles the M rows?
        # Hint: T.ceildiv =====

        m_blocks = T.ceildiv(M, block_M)
        n_blocks = T.ceildiv(N, block_N)

        with T.Kernel(n_blocks, m_blocks, threads=128) as (bx, by):

            # ===== Blank 2: Iterate in parallel over every element
            # inside the block's tile.
            # Hint: T.Parallel(dimension1, dimension2) =====
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j

                if gi < M and gj < N:
                    Y[gi, gj] = X[gi, gj] * 2.0 + 1.0

    return scale_add