__docformat__ = ["restructuredtext"]
__all__ = ["gemm_back_sub"]

import torch
from torch import Tensor


def _back_sub_gemm_no_bias1(A: Tensor, b: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    A = A @ weight
    return A, b


def _back_sub_gemm_no_bias2(A: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    b = (A * bias).sum(dim=1)
    A = _back_sub_gemm_no_bias1(A, b, weight)[0]
    return A, b


def _back_sub_gemm_no_bias3(A: Tensor, weight: Tensor) -> Tensor:
    A = A @ weight

    return A


def _back_sub_gemm(A: Tensor, b: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    b = b + (A * bias).sum(dim=1)
    A = _back_sub_gemm_no_bias1(A, b, weight)[0]
    return A, b


_A_fp32 = torch.rand((2, 3), dtype=torch.float32)
_b_fp32 = torch.rand((2,), dtype=torch.float32)
_weight_fp32 = torch.rand((3, 4), dtype=torch.float32)
_bias_fp32 = torch.rand((3,), dtype=torch.float32)

_A_fp64 = torch.rand((2, 3), dtype=torch.float64)
_b_fp64 = torch.rand((2,), dtype=torch.float64)
_weight_fp64 = torch.rand((3, 4), dtype=torch.float64)
_bias_fp64 = torch.rand((3,), dtype=torch.float64)

_example_inputs1_fp32 = (_A_fp32, _b_fp32, _weight_fp32)
_example_inputs2_fp32 = (_A_fp32, _weight_fp32, _bias_fp32)
_example_inputs3_fp32 = (_A_fp32, _weight_fp32)
_example_inputs4_fp32 = (_A_fp32, _b_fp32, _weight_fp32, _bias_fp32)


_example_inputs1_fp64 = (_A_fp64, _b_fp64, _weight_fp64)
_example_inputs2_fp64 = (_A_fp64, _weight_fp64, _bias_fp64)
_example_inputs3_fp64 = (_A_fp64, _weight_fp64)
_example_inputs4_fp64 = (_A_fp64, _b_fp64, _weight_fp64, _bias_fp64)


_back_sub_gemm_no_bias1_fp32 = torch.jit.trace(_back_sub_gemm_no_bias1, _example_inputs1_fp32)
_back_sub_gemm_no_bias2_fp32 = torch.jit.trace(_back_sub_gemm_no_bias2, _example_inputs2_fp32)
_back_sub_gemm_no_bias3_fp32 = torch.jit.trace(_back_sub_gemm_no_bias3, _example_inputs3_fp32)
_back_sub_gemm_fp32 = torch.jit.trace(_back_sub_gemm, _example_inputs4_fp32)


_back_sub_gemm_no_bias1_fp64 = torch.jit.trace(_back_sub_gemm_no_bias1, _example_inputs1_fp64)
_back_sub_gemm_no_bias2_fp64 = torch.jit.trace(_back_sub_gemm_no_bias2, _example_inputs2_fp64)
_back_sub_gemm_no_bias3_fp64 = torch.jit.trace(_back_sub_gemm_no_bias3, _example_inputs3_fp64)
_back_sub_gemm_fp64 = torch.jit.trace(_back_sub_gemm, _example_inputs4_fp64)


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

    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"The data type {dtype} is not supported.")

    if b is not None and bias is not None:
        # return _back_sub_gemm(A, b, weight, bias)
        if dtype == torch.float32:
            return _back_sub_gemm_fp32(A, b, weight, bias)
        else:
            return _back_sub_gemm_fp64(A, b, weight, bias)
    elif b is None and bias is None:
        # return _back_sub_gemm_no_bias3(A, weight), None
        if dtype == torch.float32:
            A = _back_sub_gemm_no_bias3_fp32(A, weight)
        else:
            A = _back_sub_gemm_no_bias3_fp64(A, weight)
        return A, None
    elif b is not None:
        # return _back_sub_gemm_no_bias1(A, b, weight)
        if dtype == torch.float32:
            return _back_sub_gemm_no_bias1_fp32(A, b, weight)
        else:
            return _back_sub_gemm_no_bias1_fp64(A, b, weight)
    else:
        # return _back_sub_gemm_no_bias2(A, weight, bias)
        if dtype == torch.float32:
            return _back_sub_gemm_no_bias2_fp32(A, weight, bias)
        else:
            return _back_sub_gemm_no_bias2_fp64(A, weight, bias)
