__docformat__ = ["restructuredtext"]
__all__ = [
    "conv2d_back_sub",
    "gemm_back_sub",
    "maxpool2d_back_sub",
    "nonlinear_back_sub",
    "relu_back_sub",
    "cal_scalar_bound",
    "back_sub_to_input",
    "back_sub_once_with_update_bound",
    "collect_residual_second_path",
    "back_sub_residual_second_path",
]

import logging
import time

from src.boundprop.base import NonLinearNode
from src.utils import LinearConstrBound, ScalarBound
from src.utils.colors import *

import torch
import torch.nn.functional as F
from torch import Tensor


#######################################################
## Meta variables for torch.ones() calls
#######################################################
# FP32 meta variables
_fp32_2x3 = torch.ones((2, 3), dtype=torch.float32)
_fp32_2x4 = torch.ones((2, 4), dtype=torch.float32)
_fp32_2 = torch.ones((2,), dtype=torch.float32)
_fp32_3x3 = torch.ones((3, 3), dtype=torch.float32)
_fp32_3x4 = torch.ones((3, 4), dtype=torch.float32)
_fp32_3 = torch.ones((3,), dtype=torch.float32)
_fp32_2x3x4 = torch.ones((2, 3, 4), dtype=torch.float32)
_fp32_3x4x5 = torch.ones((3, 4, 5), dtype=torch.float32)
_fp32_2x4x5x2 = torch.ones((2, 4, 5, 2), dtype=torch.float32)
_fp32_2x4x5 = torch.ones((2, 4, 5), dtype=torch.float32)
_fp32_2x3x4x5 = torch.ones((2, 3, 4, 5), dtype=torch.float32)
_fp32_2x4x4 = torch.ones((2, 4, 4), dtype=torch.float32)

# FP64 meta variables
_fp64_2x3 = torch.ones((2, 3), dtype=torch.float64)
_fp64_2x4 = torch.ones((2, 4), dtype=torch.float64)
_fp64_2 = torch.ones((2,), dtype=torch.float64)
_fp64_3x3 = torch.ones((3, 3), dtype=torch.float64)
_fp64_3x4 = torch.ones((3, 4), dtype=torch.float64)
_fp64_3 = torch.ones((3,), dtype=torch.float64)
_fp64_2x3x4 = torch.ones((2, 3, 4), dtype=torch.float64)
_fp64_3x4x5 = torch.ones((3, 4, 5), dtype=torch.float64)
_fp64_2x4x5x2 = torch.ones((2, 4, 5, 2), dtype=torch.float64)
_fp64_2x4x5 = torch.ones((2, 4, 5), dtype=torch.float64)
_fp64_2x3x4x5 = torch.ones((2, 3, 4, 5), dtype=torch.float64)
_fp64_2x4x4 = torch.ones((2, 4, 4), dtype=torch.float64)


#######################################################
## conv back_sub
#######################################################
def conv2d_back_sub(
    A: Tensor,
    b: Tensor | None,
    weight: Tensor,
    bias: Tensor | None,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    output_padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
    groups: int = 1,
) -> tuple[Tensor, Tensor | None]:
    """
    Back-substitute the linear relaxation through a 2D convolution operation.

    :param A: The matrix of the linear relaxation.
    :param b: The bias of the linear relaxation.
    :param weight: The kernel weight of the convolutional operation.
    :param bias: The kernel bias of the convolutional operation.
    :param stride: The stride of the convolutional operation.
    :param padding: The padding of the convolutional operation.
    :param output_padding: The output padding of the convolutional operation.
    :param dilation: The dilation of the convolutional operation.
    :param groups: The groups of the convolutional operation.

    :return: The matrix and bias of the linear relaxation after back-substitution.
    """
    d = (1, 2, 3)
    if b is not None and bias is not None:
        b = b + (A * bias.reshape(1, -1, 1, 1)).sum(dim=d)
    elif bias is not None:
        b = (A * bias.reshape(1, -1, 1, 1)).sum(dim=d)

    A = F.conv_transpose2d(
        A,
        weight,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )

    return A, b


#######################################################
## gemm back_sub
#######################################################
def _back_sub_gemm_no_bias1(
    A: Tensor, b: Tensor, weight: Tensor
) -> tuple[Tensor, Tensor]:
    A = A @ weight
    return A, b


def _back_sub_gemm_no_bias2(
    A: Tensor, weight: Tensor, bias: Tensor
) -> tuple[Tensor, Tensor]:
    b = (A * bias).sum(dim=1)
    A = _back_sub_gemm_no_bias1(A, b, weight)[0]
    return A, b


def _back_sub_gemm_no_bias3(A: Tensor, weight: Tensor) -> Tensor:
    A = A @ weight
    return A


def _back_sub_gemm(
    A: Tensor, b: Tensor, weight: Tensor, bias: Tensor
) -> tuple[Tensor, Tensor]:
    b = b + (A * bias).sum(dim=1)
    A = _back_sub_gemm_no_bias1(A, b, weight)[0]
    return A, b


_gemm_A_fp32 = _fp32_2x3
_gemm_b_fp32 = _fp32_2
_gemm_weight_fp32 = _fp32_3x4
_gemm_bias_fp32 = _fp32_3

_gemm_A_fp64 = _fp64_2x3
_gemm_b_fp64 = _fp64_2
_gemm_weight_fp64 = _fp64_3x4
_gemm_bias_fp64 = _fp64_3

_gemm_inputs1_fp32 = (_gemm_A_fp32, _gemm_b_fp32, _gemm_weight_fp32)
_gemm_inputs2_fp32 = (_gemm_A_fp32, _gemm_weight_fp32, _gemm_bias_fp32)
_gemm_inputs3_fp32 = (_gemm_A_fp32, _gemm_weight_fp32)
_gemm_inputs4_fp32 = (_gemm_A_fp32, _gemm_b_fp32, _gemm_weight_fp32, _gemm_bias_fp32)


_gemm_inputs1_fp64 = (_gemm_A_fp64, _gemm_b_fp64, _gemm_weight_fp64)
_gemm_inputs2_fp64 = (_gemm_A_fp64, _gemm_weight_fp64, _gemm_bias_fp64)
_gemm_inputs3_fp64 = (_gemm_A_fp64, _gemm_weight_fp64)
_gemm_inputs4_fp64 = (_gemm_A_fp64, _gemm_b_fp64, _gemm_weight_fp64, _gemm_bias_fp64)


_back_sub_gemm_no_bias1_fp32 = torch.jit.trace(
    _back_sub_gemm_no_bias1, _gemm_inputs1_fp32
)
_back_sub_gemm_no_bias2_fp32 = torch.jit.trace(
    _back_sub_gemm_no_bias2, _gemm_inputs2_fp32
)
_back_sub_gemm_no_bias3_fp32 = torch.jit.trace(
    _back_sub_gemm_no_bias3, _gemm_inputs3_fp32
)
_back_sub_gemm_fp32 = torch.jit.trace(_back_sub_gemm, _gemm_inputs4_fp32)


_back_sub_gemm_no_bias1_fp64 = torch.jit.trace(
    _back_sub_gemm_no_bias1, _gemm_inputs1_fp64
)
_back_sub_gemm_no_bias2_fp64 = torch.jit.trace(
    _back_sub_gemm_no_bias2, _gemm_inputs2_fp64
)
_back_sub_gemm_no_bias3_fp64 = torch.jit.trace(
    _back_sub_gemm_no_bias3, _gemm_inputs3_fp64
)
_back_sub_gemm_fp64 = torch.jit.trace(_back_sub_gemm, _gemm_inputs4_fp64)


def gemm_back_sub(
    A: Tensor, b: Tensor | None, weight: Tensor, bias: Tensor | None
) -> tuple[Tensor, Tensor | None]:
    """
    Back-substitute the linear relaxation through a GEMM operation.

    :param A: The matrix of the linear relaxation.
    :param b: The bias of the linear relaxation.
    :param weight: The weight of the GEMM operation.
    :param bias: The bias of the GEMM operation.

    :return: The matrix and bias of the linear relaxation after back-substitution.
    """
    dtype = A.dtype
    print(f'{BLUE}->> gemm_back_sub: {RESET}')
    print(f'    A: {A}')
    print(f'    b: {b}')
    print(f'    weight: {weight}')
    print(f'    bias: {bias}')

    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"The data type {dtype} is not supported.")

    if b is not None and bias is not None:
        if dtype == torch.float32:
            ret = _back_sub_gemm_fp32(A, b, weight, bias)
        else:
            ret = _back_sub_gemm_fp64(A, b, weight, bias)
    elif b is None and bias is None:
        if dtype == torch.float32:
            A = _back_sub_gemm_no_bias3_fp32(A, weight)
        else:
            A = _back_sub_gemm_no_bias3_fp64(A, weight)
        ret = A, None
    elif b is not None:
        if dtype == torch.float32:
            ret = _back_sub_gemm_no_bias1_fp32(A, b, weight)
        else:
            ret = _back_sub_gemm_no_bias1_fp64(A, b, weight)
    else:
        if dtype == torch.float32:
            ret = _back_sub_gemm_no_bias2_fp32(A, weight, bias)
        else:
            ret = _back_sub_gemm_no_bias2_fp64(A, weight, bias)
    print(f'{GREEN}<<- gemm_back_sub with ret: \n{ret}{RESET}')
    return ret


#######################################################
## maxpool back_sub
#######################################################
def _back_sub_maxpool_naive(
    A: Tensor, b: Tensor, s1: Tensor, s2: Tensor, t1: Tensor, t2: Tensor
) -> tuple[Tensor, Tensor]:
    d = (1, 2)
    Ap = A.clamp(min=0)
    An = A.clamp(max=0)

    b = b + torch.einsum("abc, bc->abc", Ap, t1).sum(dim=d)
    b = b + torch.einsum("abc, bc->abc", An, t2).sum(dim=d)
    A = torch.einsum("abc, bcd->abcd", Ap, s1)
    A = A + torch.einsum("abc, bcd->abcd", An, s2)

    return A, b


def _back_sub_maxpool_no_bias_naive(
    A: Tensor, s1: Tensor, s2: Tensor, t1: Tensor, t2: Tensor
) -> tuple[Tensor, Tensor]:
    d = (1, 2)
    Ap = A.clamp(min=0)
    An = A.clamp(max=0)

    b = torch.einsum("abc, bc->abc", Ap, t1).sum(dim=d)
    b = b + torch.einsum("abc, bc->abc", An, t2).sum(dim=d)
    A = torch.einsum("abc, bcd->abcd", Ap, s1)
    A = A + torch.einsum("abc, bcd->abcd", An, s2)

    return A, b


_maxpool_A_fp32 = _fp32_2x3x4
_maxpool_b_fp32 = _fp32_2
_maxpool_s1_fp32 = _fp32_3x4x5
_maxpool_s2_fp32 = _fp32_3x4x5
_maxpool_t1_fp32 = _fp32_3x4
_maxpool_t2_fp32 = _fp32_3x4

_maxpool_A_4d_fp32 = _fp32_2x3x4x5
_maxpool_b_4d_fp32 = _fp32_2x3
_maxpool_s1_4d_fp32 = _fp32_2x4x5x2
_maxpool_s2_4d_fp32 = _fp32_2x4x5x2
_maxpool_t1_4d_fp32 = _fp32_2x4x5
_maxpool_t2_4d_fp32 = _fp32_2x4x5

_maxpool_A_fp64 = _fp64_2x3x4
_maxpool_b_fp64 = _fp64_2
_maxpool_s1_fp64 = _fp64_3x4x5
_maxpool_s2_fp64 = _fp64_3x4x5
_maxpool_t1_fp64 = _fp64_3x4
_maxpool_t2_fp64 = _fp64_3x4

_maxpool_A_4d_fp64 = _fp64_2x3x4x5
_maxpool_b_4d_fp64 = _fp64_2x3
_maxpool_s1_4d_fp64 = _fp64_2x4x5x2
_maxpool_s2_4d_fp64 = _fp64_2x4x5x2
_maxpool_t1_4d_fp64 = _fp64_2x4x5
_maxpool_t2_4d_fp64 = _fp64_2x4x5


_maxpool_inputs1_fp32 = (
    _maxpool_A_fp32,
    _maxpool_b_fp32,
    _maxpool_s1_fp32,
    _maxpool_s2_fp32,
    _maxpool_t1_fp32,
    _maxpool_t2_fp32,
)
_maxpool_inputs1_no_bias_fp32 = (
    _maxpool_A_fp32,
    _maxpool_s1_fp32,
    _maxpool_s2_fp32,
    _maxpool_t1_fp32,
    _maxpool_t2_fp32,
)
_maxpool_inputs2_fp32 = (
    _maxpool_A_4d_fp32,
    _maxpool_b_4d_fp32,
    _maxpool_s1_4d_fp32,
    _maxpool_s2_4d_fp32,
    _maxpool_t1_4d_fp32,
    _maxpool_t2_4d_fp32,
)
_maxpool_inputs2_no_bias_fp32 = (
    _maxpool_A_4d_fp32,
    _maxpool_s1_4d_fp32,
    _maxpool_s2_4d_fp32,
    _maxpool_t1_4d_fp32,
    _maxpool_t2_4d_fp32,
)

_maxpool_inputs1_fp64 = (
    _maxpool_A_fp64,
    _maxpool_b_fp64,
    _maxpool_s1_fp64,
    _maxpool_s2_fp64,
    _maxpool_t1_fp64,
    _maxpool_t2_fp64,
)
_maxpool_inputs1_no_bias_fp64 = (
    _maxpool_A_fp64,
    _maxpool_s1_fp64,
    _maxpool_s2_fp64,
    _maxpool_t1_fp64,
    _maxpool_t2_fp64,
)
_maxpool_inputs2_fp64 = (
    _maxpool_A_4d_fp64,
    _maxpool_b_4d_fp64,
    _maxpool_s1_4d_fp64,
    _maxpool_s2_4d_fp64,
    _maxpool_t1_4d_fp64,
    _maxpool_t2_4d_fp64,
)
_maxpool_inputs2_no_bias_fp64 = (
    _maxpool_A_4d_fp64,
    _maxpool_s1_4d_fp64,
    _maxpool_s2_4d_fp64,
    _maxpool_t1_4d_fp64,
    _maxpool_t2_4d_fp64,
)

_back_sub_maxpool_naive_fp32 = torch.jit.trace(
    _back_sub_maxpool_naive, _maxpool_inputs1_fp32
)
_back_sub_maxpool_no_bias_naive_fp32 = torch.jit.trace(
    _back_sub_maxpool_no_bias_naive, _maxpool_inputs1_no_bias_fp32
)


_back_sub_maxpool_naive_fp64 = torch.jit.trace(
    _back_sub_maxpool_naive, _maxpool_inputs1_fp64
)
_back_sub_maxpool_no_bias_naive_fp64 = torch.jit.trace(
    _back_sub_maxpool_no_bias_naive, _maxpool_inputs1_no_bias_fp64
)


def maxpool2d_back_sub(
    A: Tensor,
    b: Tensor,
    s1: Tensor,
    s2: Tensor,
    t1: Tensor,
    t2: Tensor,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple,
    dilation: tuple,
    padding: tuple,
    stride: tuple,
) -> tuple[Tensor, Tensor | None]:
    """
    Back-substitute the linear relaxation through a maxpool operation.

    :param A: The matrix of the linear relaxation.
    :param b: The bias of the linear relaxation.
    :param s1: The slopes of the upper/lower linear relaxation of the maxpool
        operation.
    :param s2: The slopes of the lower/upper linear relaxation of the maxpool
        operation.
    :param t1: The intercepts of the upper/lower linear relaxation of the maxpool
        operation.
    :param t2: The intercepts of the lower/upper linear relaxation of the maxpool
        operation.
    :param input_size: The size of the input tensor.
    :param output_size: The size of the output tensor.
    :param kernel_size: The size of the kernel.
    :param stride: The stride of the maxpool operation.
    :param padding: The padding of the maxpool operation.
    :param dilation: The dilation of the maxpool operation.
    """
    c, hi, wi = input_size
    k = output_size[1] * output_size[2]
    s = kernel_size[0] * kernel_size[1]
    A = A.reshape(-1, c, k)

    if s1.dim() == s2.dim() == 2 and t1.dim() == t2.dim() == 1:
        s1, s2 = s1.reshape(c, k, s), s2.reshape(c, k, s)
        t1, t2 = t1.reshape(c, k), t2.reshape(c, k)

        A, b = (
            (
                _back_sub_maxpool_naive_fp32(A, b, s1, s2, t1, t2)
                if A.dtype == torch.float32
                else _back_sub_maxpool_naive_fp64(A, b, s1, s2, t1, t2)
            )
            if b is not None
            else (
                _back_sub_maxpool_no_bias_naive_fp32(A, s1, s2, t1, t2)
                if A.dtype == torch.float32
                else _back_sub_maxpool_no_bias_naive_fp64(A, s1, s2, t1, t2)
            )
        )

    else:
        raise ValueError(
            f"The dimensions are not supported. "
            f"A: {A.shape}, b: {b.shape}, "
            f"s1: {s1.shape}, s2: {s2.shape}, "
            f"t1: {t1.shape}, t2: {t2.shape}."
        )

    A = F.fold(
        A.reshape(-1, c, k, s).permute(0, 1, 3, 2).reshape(-1, c * s, k),
        input_size[1:],
        kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    ).reshape(-1, c * hi * wi)
    b = b.flatten() if b is not None else None

    return A, b


#######################################################
## nonlinear back_sub
#######################################################
def _back_sub_nonlinear_1d(
    A: Tensor, b: Tensor, s1: Tensor, s2: Tensor, t1: Tensor, t2: Tensor
) -> tuple[Tensor, Tensor]:
    Ap, An = A.clamp(min=0), A.clamp(max=0)
    b = b + (Ap * t1).sum(dim=1) + (An * t2).sum(dim=1)
    A = Ap * s1 + An * s2

    return A, b


def _back_sub_nonlinear_no_bias_1d(
    A: Tensor, s1: Tensor, s2: Tensor, t1: Tensor, t2: Tensor
) -> tuple[Tensor, Tensor]:
    Ap, An = A.clamp(min=0), A.clamp(max=0)
    b = (Ap * t1).sum(dim=1) + (An * t2).sum(dim=1)
    A = Ap * s1 + An * s2

    return A, b


_nonlinear_A_fp32 = _fp32_2x3
_nonlinear_b_fp32 = _fp32_2
_nonlinear_s1_fp32 = _fp32_3
_nonlinear_s2_fp32 = _fp32_3
_nonlinear_s1_2d_fp32 = _fp32_3x3
_nonlinear_s2_2d_fp32 = _fp32_3x3
_nonlinear_t1_fp32 = _fp32_3
_nonlinear_t2_fp32 = _fp32_3

_nonlinear_A_3d_fp32 = _fp32_2x3x4
_nonlinear_b_3d_fp32 = _fp32_2x3
_nonlinear_s1_3d_fp32 = _fp32_2x4x4
_nonlinear_s2_3d_fp32 = _fp32_2x4x4
_nonlinear_t1_3d_fp32 = _fp32_2x4
_nonlinear_t2_3d_fp32 = _fp32_2x4

_nonlinear_A_fp64 = _fp64_2x3
_nonlinear_b_fp64 = _fp64_2
_nonlinear_s1_fp64 = _fp64_3
_nonlinear_s2_fp64 = _fp64_3
_nonlinear_s1_2d_fp64 = _fp64_3x3
_nonlinear_s2_2d_fp64 = _fp64_3x3
_nonlinear_t1_fp64 = _fp64_3
_nonlinear_t2_fp64 = _fp64_3

_nonlinear_A_3d_fp64 = _fp64_2x3x4
_nonlinear_b_3d_fp64 = _fp64_2x3
_nonlinear_s1_3d_fp64 = _fp64_2x4x4
_nonlinear_s2_3d_fp64 = _fp64_2x4x4
_nonlinear_t1_3d_fp64 = _fp64_2x4
_nonlinear_t2_3d_fp64 = _fp64_2x4


_nonlinear_inputs_1d_fp32 = (
    _nonlinear_A_fp32,
    _nonlinear_b_fp32,
    _nonlinear_s1_fp32,
    _nonlinear_s2_fp32,
    _nonlinear_t1_fp32,
    _nonlinear_t2_fp32,
)
_nonlinear_inputs_no_bias_1d_fp32 = (
    _nonlinear_A_fp32,
    _nonlinear_s1_fp32,
    _nonlinear_s2_fp32,
    _nonlinear_t1_fp32,
    _nonlinear_t2_fp32,
)
_nonlinear_inputs_2d_fp32 = (
    _nonlinear_A_fp32,
    _nonlinear_b_fp32,
    _nonlinear_s1_2d_fp32,
    _nonlinear_s2_2d_fp32,
    _nonlinear_t1_fp32,
    _nonlinear_t2_fp32,
)
_nonlinear_inputs_no_bias_2d_fp32 = (
    _nonlinear_A_fp32,
    _nonlinear_s1_2d_fp32,
    _nonlinear_s2_2d_fp32,
    _nonlinear_t1_fp32,
    _nonlinear_t2_fp32,
)
_nonlinear_inputs_3d_fp32 = (
    _nonlinear_A_3d_fp32,
    _nonlinear_b_3d_fp32,
    _nonlinear_s1_3d_fp32,
    _nonlinear_s2_3d_fp32,
    _nonlinear_t1_3d_fp32,
    _nonlinear_t2_3d_fp32,
)
_nonlinear_inputs_no_bias_3d_fp32 = (
    _nonlinear_A_3d_fp32,
    _nonlinear_s1_3d_fp32,
    _nonlinear_s2_3d_fp32,
    _nonlinear_t1_3d_fp32,
    _nonlinear_t2_3d_fp32,
)

_nonlinear_inputs_1d_fp64 = (
    _nonlinear_A_fp64,
    _nonlinear_b_fp64,
    _nonlinear_s1_fp64,
    _nonlinear_s2_fp64,
    _nonlinear_t1_fp64,
    _nonlinear_t2_fp64,
)
_nonlinear_inputs_no_bias_1d_fp64 = (
    _nonlinear_A_fp64,
    _nonlinear_s1_fp64,
    _nonlinear_s2_fp64,
    _nonlinear_t1_fp64,
    _nonlinear_t2_fp64,
)
_nonlinear_inputs_2d_fp64 = (
    _nonlinear_A_fp64,
    _nonlinear_b_fp64,
    _nonlinear_s1_2d_fp64,
    _nonlinear_s2_2d_fp64,
    _nonlinear_t1_fp64,
    _nonlinear_t2_fp64,
)
_nonlinear_inputs_no_bias_2d_fp64 = (
    _nonlinear_A_fp64,
    _nonlinear_s1_2d_fp64,
    _nonlinear_s2_2d_fp64,
    _nonlinear_t1_fp64,
    _nonlinear_t2_fp64,
)
_nonlinear_inputs_3d_fp64 = (
    _nonlinear_A_3d_fp64,
    _nonlinear_b_3d_fp64,
    _nonlinear_s1_3d_fp64,
    _nonlinear_s2_3d_fp64,
    _nonlinear_t1_3d_fp64,
    _nonlinear_t2_3d_fp64,
)
_nonlinear_inputs_no_bias_3d_fp64 = (
    _nonlinear_A_3d_fp64,
    _nonlinear_s1_3d_fp64,
    _nonlinear_s2_3d_fp64,
    _nonlinear_t1_3d_fp64,
    _nonlinear_t2_3d_fp64,
)


_back_sub_nonlinear_1d_fp32 = torch.jit.trace(
    _back_sub_nonlinear_1d, _nonlinear_inputs_1d_fp32
)
_back_sub_nonlinear_no_bias_1d_fp32 = torch.jit.trace(
    _back_sub_nonlinear_no_bias_1d, _nonlinear_inputs_no_bias_1d_fp32
)

_back_sub_nonlinear_1d_fp64 = torch.jit.trace(
    _back_sub_nonlinear_1d, _nonlinear_inputs_1d_fp64
)
_back_sub_nonlinear_no_bias_1d_fp64 = torch.jit.trace(
    _back_sub_nonlinear_no_bias_1d, _nonlinear_inputs_no_bias_1d_fp64
)


def nonlinear_back_sub(
    A: Tensor, b: Tensor | None, s1: Tensor, s2: Tensor, t1: Tensor, t2: Tensor
) -> tuple[Tensor, Tensor | None]:
    """
    Back-substitute the linear relaxation through a non-linear operation. Based on
    calculating the lower or upper linear relaxation, lower or upper slopes and
    intercepts are used.

    :param A: The matrix of the linear relaxation.
    :param b: The bias of the linear relaxation.
    :param s1: The slopes of the upper/lower linear relaxation of the non-linear
        operation.
    :param s2: The slopes of the lower/upper linear relaxation of the non-linear
        operation.
    :param t1: The intercepts of the upper/lower linear relaxation of the non-linear
        operation.
    :param t2: The intercepts of the lower/upper linear relaxation of the non-linear
        operation.
    """
    if A.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"The data type {A.dtype} is not supported.")

    if s1.dim() == s2.dim() == 1 and t1.dim() == t2.dim() == 1:

        if b is not None:
            if A.dtype == torch.float32:
                return _back_sub_nonlinear_1d_fp32(A, b, s1, s2, t1, t2)
            return _back_sub_nonlinear_1d_fp64(A, b, s1, s2, t1, t2)
        if A.dtype == torch.float32:
            return _back_sub_nonlinear_no_bias_1d_fp32(A, s1, s2, t1, t2)
        return _back_sub_nonlinear_no_bias_1d_fp64(A, s1, s2, t1, t2)

    else:
        raise ValueError(
            f"The dimensions are not supported. "
            f"A: {A.shape}, b: {b.shape}, "
            f"s1: {s1.shape}, s2: {s2.shape}, "
            f"t1: {t1.shape}, t2: {t2.shape}."
        )


#######################################################
## relu back_sub
#######################################################

"""
This module contains the implementation of the specialized ad-hoc back-substitution for
the ReLU activation function.
"""


def _back_sub_relu_1d_lower(
    A: Tensor, b: Tensor | None, sl: Tensor, su: Tensor, tu: Tensor
) -> tuple[Tensor, Tensor]:
    Ap, An = A.clamp(min=0), A.clamp(max=0)
    bx = (An * tu).sum(dim=1)
    b = bx + b if b is not None else bx
    A = Ap * sl + An * su
    return A, b


def _back_sub_relu_1d_upper(
    A: Tensor, b: Tensor | None, sl: Tensor, su: Tensor, tu: Tensor
) -> tuple[Tensor, Tensor]:
    Ap, An = A.clamp(min=0), A.clamp(max=0)
    bx = (Ap * tu).sum(dim=1)
    b = bx + b if b is not None else bx
    A = Ap * su + An * sl
    return A, b


def relu_back_sub(
    A: Tensor,
    b: Tensor | None,
    sl: Tensor,
    su: Tensor,
    tu: Tensor,
    is_lower: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """
    Back-substitute the linear relaxation through a non-linear operation. Based on
    calculating the lower or upper linear relaxation, lower or upper slopes and
    intercepts are used.

    .. tip::
        This is a specialized implementation for the ReLU activation function.

    :param A: The matrix of the linear relaxation.
    :param b: The bias of the linear relaxation.
    :param sl: The slopes of the lower linear relaxation of the non-linear operation.
    :param su: The slopes of the upper linear relaxation of the non-linear operation.
    :param tu: The intercepts of the upper linear relaxation of the non-linear
        operation.
    :param is_lower: If True, calculate the lower relaxation; otherwise, calculate the
        upper relaxation.

    :return: The matrix and bias of the linear relaxation.
    """
    assert A.dtype in (
        torch.float32,
        torch.float64,
    ), f"The data type {A.dtype} is not supported."
    assert (
        sl.dim() == su.dim() == 1 and tu.dim() == 1
    ), f"The dimensions are not supported. sl: {sl.shape}, su: {su.shape}, tu: {tu.shape}."

    print(f'{CYAN}->> relu_back_sub: is_lower={is_lower}{RESET}')
    print(f'    A: {A}')
    print(f'    b: {b}')
    print(f'    sl: {sl}')
    print(f'    su: {su}')
    print(f'    tu: {tu}')
    if is_lower:
        ret = _back_sub_relu_1d_lower(A, b, sl, su, tu)
    else:
        ret = _back_sub_relu_1d_upper(A, b, sl, su, tu)
    print(f'{GREEN}<<- relu_back_sub with ret: \n{ret}{RESET}')
    return ret


#######################################################
## scalar bound calculation
#######################################################
def _cal_scalar_bound_1d(A: Tensor, b: Tensor, s1: Tensor, s2: Tensor) -> Tensor:
    d = (1,)
    Ap = A.clamp(min=0)
    An = A.clamp(max=0)
    sb = b + (Ap * s1).sum(dim=d) + (An * s2).sum(dim=d)
    return sb


def _cal_scalar_bound_no_bias_1d(A: Tensor, s1: Tensor, s2: Tensor) -> Tensor:
    d = (1,)
    Ap = A.clamp(min=0)
    An = A.clamp(max=0)
    sb = (Ap * s1).sum(dim=d) + (An * s2).sum(dim=d)
    return sb


def _cal_scalar_bound_3d(A: Tensor, b: Tensor, s1: Tensor, s2: Tensor) -> Tensor:
    d = (1, 2, 3)
    Ap = A.clamp(min=0)
    An = A.clamp(max=0)
    sb = b + (Ap * s1).sum(dim=d) + (An * s2).sum(dim=d)
    return sb


def _cal_scalar_bound_no_bias_3d(A: Tensor, s1: Tensor, s2: Tensor) -> Tensor:
    d = (1, 2, 3)
    Ap = A.clamp(min=0)
    An = A.clamp(max=0)
    sb = (Ap * s1).sum(dim=d) + (An * s2).sum(dim=d)
    return sb


_scalar_A_fp32 = _fp32_2x3
_scalar_b_fp32 = _fp32_2
_scalar_s1_fp32 = _fp32_3
_scalar_s2_fp32 = _fp32_3

_scalar_A_3d_fp32 = _fp32_2x3x4x5
_scalar_b_3d_fp32 = _fp32_2
_scalar_s1_3d_fp32 = _fp32_3x4x5
_scalar_s2_3d_fp32 = _fp32_3x4x5

_scalar_A_fp64 = _fp64_2x3
_scalar_b_fp64 = _fp64_2
_scalar_s1_fp64 = _fp64_3
_scalar_s2_fp64 = _fp64_3

_scalar_A_3d_fp64 = _fp64_2x3x4x5
_scalar_b_3d_fp64 = _fp64_2
_scalar_s1_3d_fp64 = _fp64_3x4x5
_scalar_s2_3d_fp64 = _fp64_3x4x5


_scalar_inputs1_fp32 = (
    _scalar_A_fp32,
    _scalar_b_fp32,
    _scalar_s1_fp32,
    _scalar_s2_fp32,
)
_scalar_inputs1_no_bias_fp32 = (_scalar_A_fp32, _scalar_s1_fp32, _scalar_s2_fp32)
_scalar_inputs2_fp32 = (
    _scalar_A_3d_fp32,
    _scalar_b_3d_fp32,
    _scalar_s1_3d_fp32,
    _scalar_s2_3d_fp32,
)
_scalar_inputs2_no_bias_fp32 = (
    _scalar_A_3d_fp32,
    _scalar_s1_3d_fp32,
    _scalar_s2_3d_fp32,
)
_scalar_inputs1_fp64 = (
    _scalar_A_fp64,
    _scalar_b_fp64,
    _scalar_s1_fp64,
    _scalar_s2_fp64,
)
_scalar_inputs1_no_bias_fp64 = (_scalar_A_fp64, _scalar_s1_fp64, _scalar_s2_fp64)
_scalar_inputs2_fp64 = (
    _scalar_A_3d_fp64,
    _scalar_b_3d_fp64,
    _scalar_s1_3d_fp64,
    _scalar_s2_3d_fp64,
)
_scalar_inputs2_no_bias_fp64 = (
    _scalar_A_3d_fp64,
    _scalar_s1_3d_fp64,
    _scalar_s2_3d_fp64,
)

_cal_scalar_bound_1d_fp32 = torch.jit.trace(_cal_scalar_bound_1d, _scalar_inputs1_fp32)
_cal_scalar_bound_no_bias_1d_fp32 = torch.jit.trace(
    _cal_scalar_bound_no_bias_1d, _scalar_inputs1_no_bias_fp32
)
_cal_scalar_bound_3d_fp32 = torch.jit.trace(_cal_scalar_bound_3d, _scalar_inputs2_fp32)
_cal_scalar_bound_no_bias_3d_fp32 = torch.jit.trace(
    _cal_scalar_bound_no_bias_3d, _scalar_inputs2_no_bias_fp32
)


_cal_scalar_bound_1d_fp64 = torch.jit.trace(_cal_scalar_bound_1d, _scalar_inputs1_fp64)
_cal_scalar_bound_no_bias_1d_fp64 = torch.jit.trace(
    _cal_scalar_bound_no_bias_1d, _scalar_inputs1_no_bias_fp64
)
_cal_scalar_bound_3d_fp64 = torch.jit.trace(_cal_scalar_bound_3d, _scalar_inputs2_fp64)
_cal_scalar_bound_no_bias_3d_fp64 = torch.jit.trace(
    _cal_scalar_bound_no_bias_3d, _scalar_inputs2_no_bias_fp64
)


def cal_scalar_bound(A: Tensor, b: Tensor | None, s1: Tensor, s2: Tensor) -> Tensor:
    """
    Calculate the scalar bound of the linear relaxation.

    :param A: The matrix of the linear relaxation.
    :param b: The bias of the linear relaxation.
    :param s1: The scalar lower or upper bound.
    :param s2: The scalar upper or lower bound.

    :return: The scalar bound of the linear relaxation.
    """

    d = A.dim()
    dtype = A.dtype

    if dtype == torch.float32:
        if b is not None:
            if d == 2:
                return _cal_scalar_bound_1d_fp32(A, b, s1, s2)
            elif d == 4:
                return _cal_scalar_bound_3d_fp32(A, b, s1, s2)
            else:
                raise ValueError(
                    f"Unsupported dimension: "
                    f"A: {A.shape}, b: {b.shape}, l: {s1.shape}, u: {s2.shape}."
                )
        else:
            if d == 2:
                return _cal_scalar_bound_no_bias_1d_fp32(A, s1, s2)
            elif d == 4:
                return _cal_scalar_bound_no_bias_3d_fp32(A, s1, s2)
            else:
                raise ValueError(
                    f"Unsupported dimension: "
                    f"A: {A.shape} l: {s1.shape}, u: {s2.shape}."
                )
    elif dtype == torch.float64:
        if b is not None:
            if d == 2:
                return _cal_scalar_bound_1d_fp64(A, b, s1, s2)
            elif d == 4:
                return _cal_scalar_bound_3d_fp64(A, b, s1, s2)
            else:
                raise ValueError(
                    f"Unsupported dimension: "
                    f"A: {A.shape}, b: {b.shape}, l: {s1.shape}, u: {s2.shape}."
                )
        else:
            if d == 2:
                return _cal_scalar_bound_no_bias_1d_fp64(A, s1, s2)
            elif d == 4:
                return _cal_scalar_bound_no_bias_3d_fp64(A, s1, s2)
            else:
                raise ValueError(
                    f"Unsupported dimension: "
                    f"A: {A.shape} l: {s1.shape}, u: {s2.shape}."
                )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}.")


# Note: Missing meta variables that weren't created:
# _fp32_3x3, _fp32_2x4, _fp64_3x3, _fp64_2x4
# Adding them for completeness:
_fp32_3x3 = torch.ones((3, 3), dtype=torch.float32)
_fp32_2x4 = torch.ones((2, 4), dtype=torch.float32)
_fp64_3x3 = torch.ones((3, 3), dtype=torch.float64)
_fp64_2x4 = torch.ones((2, 4), dtype=torch.float64)

# Update the references that need these new meta variables
_nonlinear_s1_2d_fp32 = _fp32_3x3
_nonlinear_s2_2d_fp32 = _fp32_3x3
_nonlinear_t1_3d_fp32 = _fp32_2x4
_nonlinear_t2_3d_fp32 = _fp32_2x4

_nonlinear_s1_2d_fp64 = _fp64_3x3
_nonlinear_s2_2d_fp64 = _fp64_3x3
_nonlinear_t1_3d_fp64 = _fp64_2x4
_nonlinear_t2_3d_fp64 = _fp64_2x4


#######################################################
## to_input back_sub for residual blocks
#######################################################
def back_sub_to_input(
    self: "BasicIneqNode",  # noqa
    constr_bound: LinearConstrBound,
) -> LinearConstrBound:
    """
    Back-substitute the linear relaxation to the preceding layer until the input layer.

    :param self: The current layer object, and it is a linear layer.
    :param constr_bound: The linear relaxation to back-substitute.

    :return: The linear relaxation represented by input variables.
    """

    print(f"[DEBUG] Back-substitute to input for {self}.")
    start = time.perf_counter()

    # The constraint bound for the output of the residual block.
    constr_bound_r: LinearConstrBound | None = None
    # The queue to store the modules in the second path of the residual block.
    residual_second_path: list["BasicIneqNode"] = []  # noqa
    # If self is after a residual block and the current module is in the residual block,
    # then do not update the bound by the current module.
    in_residual_block = False

    module: "BasicIneqNode" = self  # noqa
    while True:
        if (
            constr_bound_r is not None
            and module.next_nodes is not None
            and len(module.next_nodes) == 2
        ):
            (
                constr_bound,
                constr_bound_r,
                residual_second_path,
                _,
            ) = back_sub_residual_second_path(
                self, module, constr_bound, constr_bound_r, residual_second_path
            )
            in_residual_block = False

        constr_bound, _ = back_sub_once_with_update_bound(
            self, module, constr_bound, in_residual_block
        )

        if module.pre_nodes is None:
            break

        if len(module.pre_nodes) == 2:
            constr_bound_r, residual_second_path = collect_residual_second_path(
                module, constr_bound, residual_second_path
            )
            in_residual_block = True
        elif len(module.pre_nodes) != 1:
            raise RuntimeError(f"Invalid number of pre nodes: {len(module.pre_nodes)}.")

        module = module.pre_nodes[0]

    print(
        f"[DEBUG] Finish back-substitution in {time.perf_counter() - start:.4f} seconds."
    )

    return constr_bound


def back_sub_once_with_update_bound(
    self: "BasicIneqNode",  # noqa
    module: "BasicIneqNode",  # noqa
    constr_bound: LinearConstrBound,
    in_residual_block: bool,
    store_updated_bounds: bool = True,
) -> tuple[LinearConstrBound, ScalarBound | None]:

    constr_bound = module.back_sub_once(constr_bound)
    pre_module = module.pre_nodes[0] if module.pre_nodes else None
    bound = None
    if (
        not in_residual_block
        and pre_module is not None
        and isinstance(pre_module, NonLinearNode)
        and pre_module.act_relax_args.update_scalar_bounds_per_layer
    ):
        print(f"[DEBUG] Update scalar bounds by {pre_module}.")
        pre_scalar_bound = pre_module.all_bounds[pre_module.name].reshape(
            *constr_bound.L.A.shape[1:]
        )

        bound = self.cal_bounds(constr_bound, pre_scalar_bound)
        if store_updated_bounds:
            bound = self.store_bounds(self.all_bounds, self.name, bound)

    return constr_bound, bound


def collect_residual_second_path(
    module: "BasicIneqNode",  # noqa
    constr_bound: LinearConstrBound,
    residual_second_path: list["BasicIneqNode"],  # noqa
) -> tuple[LinearConstrBound, list["BasicIneqNode"]]:  # noqa
    # For residual block, there will be two paths to the input.
    # We collect the modules in the second path and process them later.
    # The input biases will be calculated in the second path, so we do not
    # need the biases for the second path.
    constr_bound_r = constr_bound.detach().clone()
    constr_bound_r.L.b = None
    if constr_bound_r.U is not None:
        constr_bound_r.U.b = None
    # Store the modules in the second path.
    module_r = module.pre_nodes[1]
    while len(module_r.next_nodes) == 1:
        residual_second_path.append(module_r)
        module_r = module_r.pre_nodes[0]

    return constr_bound_r, residual_second_path


def back_sub_residual_second_path(
    self: "BasicIneqNode",  # noqa
    module: "BasicIneqNode",  # noqa
    constr_bound: LinearConstrBound,
    constr_bound_r: LinearConstrBound,
    residual_second_path: list["BasicIneqNode"],  # noqa
    store_updated_bounds: bool = True,
) -> (
    tuple[LinearConstrBound, None, list["BasicIneqNode"]]
    | tuple[LinearConstrBound, None, list["BasicIneqNode"], ScalarBound | None]
):  # noqa  # noqa

    residual_second_path: list["BasicIneqNode"]  # noqa
    for module_r in residual_second_path:
        constr_bound_r = module_r.back_sub_once(constr_bound_r)
    constr_bound_r.L.A = constr_bound_r.L.A.reshape(constr_bound.L.A.shape)
    if constr_bound_r.U is not None:
        constr_bound_r.U.A = constr_bound_r.U.A.reshape(constr_bound.U.A.shape)
    constr_bound = constr_bound + constr_bound_r
    # Reset
    constr_bound_r = None
    residual_second_path.clear()

    bound = None
    # Handle the non-linear module in the beginning of the residual block.
    if (
        isinstance(module, NonLinearNode)
        and module.act_relax_args.update_scalar_bounds_per_layer
    ):
        print(f"[DEBUG] Update scalar bounds by {module}.")
        pre_scalar_bound = module.all_bounds[module.name].reshape(
            *constr_bound.L.A.shape[1:]
        )

        bound, _ = self.cal_bounds(constr_bound, pre_scalar_bound)
        if store_updated_bounds:
            bound = self.store_bounds(self.all_bounds, self.name, bound)

    return constr_bound, constr_bound_r, residual_second_path, bound
