"""Problem 7.2: Fused elementwise operation (modification task).

scale_kernel is currently complete. Do not modify its corresponding code.

fused_kernel is currently identical to scale_kernel and is the kernel that you
need to modify.

Task: Change it so that it computes:

    z = relu(a * x + b)

where a and b are scalars.

TIP: You only need to change the computation line and pass a and b into the
kernel—the main structure remains unchanged. This is precisely the advantage
of the tile perspective :-).

After making the changes, run:

    pytest tests/test_fused_op.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def scale_kernel(x_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    z = x * 2.0
    tl.store(z_ptr + offsets, z, mask=mask)


def scale(x: torch.Tensor) -> torch.Tensor:
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    scale_kernel[grid](x, z, n, BLOCK_SIZE=BLOCK_SIZE)
    return z


# ===== Modify from this point onward =====

@triton.jit
def fused_kernel(x_ptr, a, b, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = a*x+b

    # TODO: Change this to relu(a * x + b).
    # Hint: use tl.maximum.
    z = tl.maximum(x,0.0)

    tl.store(z_ptr + offsets, z, mask=mask)


def fused(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    # TODO: Pass a and b into the kernel.
    fused_kernel[grid](x, a, b, z, n, BLOCK_SIZE=BLOCK_SIZE)

    return z