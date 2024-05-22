# Copyright (c) 2023, Tri Dao.

import math
try:
    from scipy.linalg import hadamard
except ImportError:
    hadamard = None

import torch
import torch.nn.functional as F


import slow_hadamard_transform_cuda


class HadamardTransformFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scales, scale_first):
        ctx._ht_scales = scales
        ctx._ht_scale_first = scale_first
        return slow_hadamard_transform_cuda.fast_hadamard_transform(x, scales, scale_first)

    @staticmethod
    def backward(ctx, dout):
        # The Hadamard transform matrix is symmetric, so in the backward pass we multiply by its
        # transpose, which is itself.
        return slow_hadamard_transform_cuda.fast_hadamard_transform(dout, ctx._ht_scales, ctx._ht_scale_first), None


def hadamard_transform(x, scales, scale_first):
    """
    Arguments:
        x: (..., dim)
        scales: (..., dim). Multiply the input/output by this tensor (elementwise).
        scale_first: bool. Apply the scales before or after the hadamard transform
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix.
    Equivalent to F.linear(x * scales, torch.tensor(scipy.linalg.hadamard(dim))) when scale_first = True,
               or F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scales when scale_first = False.
    If dim is not a power of 2, we implicitly pad x with zero so that dim is the next power of 2.
    """
    return HadamardTransformFn.apply(x, scales, scale_first)


class HadamardTransform12NFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scales, scale_first):
        ctx._ht_scales = scales
        ctx._ht_scale_first = scale_first
        return slow_hadamard_transform_cuda.slow_hadamard_transform_12N(x, scales, scale_first)

    @staticmethod
    def backward(ctx, dout):
        # The Hadamard transform matrix is symmetric, so in the backward pass we multiply by its
        # transpose, which is itself.
        return slow_hadamard_transform_cuda.slow_hadamard_transform_12N(dout, ctx._ht_scale, ctx._ht_scale_first), None
    

def hadamard_transform_12N(x, scales, scale_first):
    """
    Arguments:
        x: (..., dim)
        scales: (..., dim). Multiply the input/output by this tensor (elementwise).
        scale_first: bool. Apply the scales before or after the hadamard transform
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix, where dim = 12 * power of 2.
    If dim is not 12 * a power of 2, we implicitly pad x with zero so that dim is 12 * the next power of 2.
    """
    return HadamardTransform12NFn.apply(x, scales, scale_first)



class HadamardTransform20NFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scales, scale_first):
        ctx._ht_scales = scales
        ctx._ht_scale_first = scale_first
        return slow_hadamard_transform_cuda.slow_hadamard_transform_20N(x, scales, scale_first)

    @staticmethod
    def backward(ctx, dout):
        # The Hadamard transform matrix is symmetric, so in the backward pass we multiply by its
        # transpose, which is itself.
        return slow_hadamard_transform_cuda.slow_hadamard_transform_20N(dout, ctx._ht_scale, ctx._ht_scale_first), None


def hadamard_transform_20N(x, scales, scale_first):
    """
    Arguments:
        x: (..., dim)
        scales: (..., dim). Multiply the input/output by this tensor (elementwise).
        scale_first: bool. Apply the scales before or after the hadamard transform
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix, where dim = 20 * power of 2.
    If dim is not 20 * a power of 2, we implicitly pad x with zero so that dim is 20 * the next power of 2.
    """
    return HadamardTransform20NFn.apply(x, scales, scale_first)


class HadamardTransform28NFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scales, scale_first):
        ctx._ht_scales = scales
        ctx._ht_scale_first = scale_first
        return slow_hadamard_transform_cuda.slow_hadamard_transform_28N(x, scales, scale_first)

    @staticmethod
    def backward(ctx, dout):
        # The Hadamard transform matrix is symmetric, so in the backward pass we multiply by its
        # transpose, which is itself.
        return slow_hadamard_transform_cuda.slow_hadamard_transform_28N(dout, ctx._ht_scale, ctx._ht_scale_first), None


def hadamard_transform_28N(x, scales, scale_first):
    """
    Arguments:
        x: (..., dim)
        scales: (..., dim). Multiply the input/output by this tensor (elementwise).
        scale_first: bool. Apply the scales before or after the hadamard transform
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix, where dim = 28 * power of 2.
    If dim is not 28 * a power of 2, we implicitly pad x with zero so that dim is 28 * the next power of 2.
    """
    return HadamardTransform28NFn.apply(x, scales, scale_first)


class HadamardTransform40NFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scales, scale_first):
        ctx._ht_scales = scales
        ctx._ht_scale_first = scale_first
        return slow_hadamard_transform_cuda.slow_hadamard_transform_40N(x, scales, scale_first)

    @staticmethod
    def backward(ctx, dout):
        # The Hadamard transform matrix is symmetric, so in the backward pass we multiply by its
        # transpose, which is itself.
        return slow_hadamard_transform_cuda.slow_hadamard_transform_40N(dout, ctx._ht_scale, ctx._ht_scale_first), None


def hadamard_transform_40N(x, scales, scale_first):
    """
    Arguments:
        x: (..., dim)
        scales: (..., dim). Multiply the input/output by this tensor (elementwise).
        scale_first: bool. Apply the scales before or after the hadamard transform
    Returns:
        out: (..., dim)

    Multiply each row of x by the Hadamard transform matrix, where dim = 40 * power of 2.
    If dim is not 40 * a power of 2, we implicitly pad x with zero so that dim is 40 * the next power of 2.
    """
    return HadamardTransform40NFn.apply(x, scales, scale_first)


def hadamard_transform_ref(x, scales, scale_first):
    """
    x: (..., dim)
    out: (..., dim)
    """
    if hadamard is None:
        raise ImportError("Please install scipy")
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    if scale_first:
        out = F.linear(x * scales, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    else:
        out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device)) * scale
    return out[..., :dim].reshape(*x_shape)
