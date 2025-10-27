__docformat__ = ["restructuredtext"]
__all__ = ["nonlinear_back_sub"]

import torch
from torch import Tensor


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


_A_fp32 = torch.ones((2, 3), dtype=torch.float32)
_b_fp32 = torch.ones((2,), dtype=torch.float32)
_s1_fp32 = torch.ones((3,), dtype=torch.float32)
_s2_fp32 = torch.ones((3,), dtype=torch.float32)
_s1_2d_fp32 = torch.ones((3, 3), dtype=torch.float32)
_s2_2d_fp32 = torch.ones((3, 3), dtype=torch.float32)
_t1_fp32 = torch.ones((3,), dtype=torch.float32)
_t2_fp32 = torch.ones((3,), dtype=torch.float32)

_A_3d_fp32 = torch.ones((2, 3, 4), dtype=torch.float32)
_b_3d_fp32 = torch.ones((2, 3), dtype=torch.float32)
_s1_3d_fp32 = torch.ones((2, 4, 4), dtype=torch.float32)
_s2_3d_fp32 = torch.ones((2, 4, 4), dtype=torch.float32)
_t1_3d_fp32 = torch.ones((2, 4), dtype=torch.float32)
_t2_3d_fp32 = torch.ones((2, 4), dtype=torch.float32)

_A_fp64 = torch.ones((2, 3), dtype=torch.float64)
_b_fp64 = torch.ones((2,), dtype=torch.float64)
_s1_fp64 = torch.ones((3,), dtype=torch.float64)
_s2_fp64 = torch.ones((3,), dtype=torch.float64)
_s1_2d_fp64 = torch.ones((3, 3), dtype=torch.float64)
_s2_2d_fp64 = torch.ones((3, 3), dtype=torch.float64)
_t1_fp64 = torch.ones((3,), dtype=torch.float64)
_t2_fp64 = torch.ones((3,), dtype=torch.float64)

_A_3d_fp64 = torch.ones((2, 3, 4), dtype=torch.float64)
_b_3d_fp64 = torch.ones((2, 3), dtype=torch.float64)
_s1_3d_fp64 = torch.ones((2, 4, 4), dtype=torch.float64)
_s2_3d_fp64 = torch.ones((2, 4, 4), dtype=torch.float64)
_t1_3d_fp64 = torch.ones((2, 4), dtype=torch.float64)
_t2_3d_fp64 = torch.ones((2, 4), dtype=torch.float64)


_example_input_1d_fp32 = (_A_fp32, _b_fp32, _s1_fp32, _s2_fp32, _t1_fp32, _t2_fp32)
_example_input_no_bias_1d_fp32 = (_A_fp32, _s1_fp32, _s2_fp32, _t1_fp32, _t2_fp32)
_example_input_2d_fp32 = (_A_fp32, _b_fp32, _s1_2d_fp32, _s2_2d_fp32, _t1_fp32, _t2_fp32)
_example_input_no_bias_2d_fp32 = (_A_fp32, _s1_2d_fp32, _s2_2d_fp32, _t1_fp32, _t2_fp32)
_example_input_3d_fp32 = (
    _A_3d_fp32,
    _b_3d_fp32,
    _s1_3d_fp32,
    _s2_3d_fp32,
    _t1_3d_fp32,
    _t2_3d_fp32,
)
_example_input_no_bias_3d_fp32 = (
    _A_3d_fp32,
    _s1_3d_fp32,
    _s2_3d_fp32,
    _t1_3d_fp32,
    _t2_3d_fp32,
)

_example_input_1d_fp64 = (_A_fp64, _b_fp64, _s1_fp64, _s2_fp64, _t1_fp64, _t2_fp64)
_example_input_no_bias_1d_fp64 = (_A_fp64, _s1_fp64, _s2_fp64, _t1_fp64, _t2_fp64)
_example_input_2d_fp64 = (_A_fp64, _b_fp64, _s1_2d_fp64, _s2_2d_fp64, _t1_fp64, _t2_fp64)
_example_input_no_bias_2d_fp64 = (_A_fp64, _s1_2d_fp64, _s2_2d_fp64, _t1_fp64, _t2_fp64)
_example_input_3d_fp64 = (
    _A_3d_fp64,
    _b_3d_fp64,
    _s1_3d_fp64,
    _s2_3d_fp64,
    _t1_3d_fp64,
    _t2_3d_fp64,
)
_example_input_no_bias_3d_fp64 = (
    _A_3d_fp64,
    _s1_3d_fp64,
    _s2_3d_fp64,
    _t1_3d_fp64,
    _t2_3d_fp64,
)


_back_sub_nonlinear_1d_fp32 = torch.jit.trace(_back_sub_nonlinear_1d, _example_input_1d_fp32)
_back_sub_nonlinear_no_bias_1d_fp32 = torch.jit.trace(
    _back_sub_nonlinear_no_bias_1d, _example_input_no_bias_1d_fp32
)

_back_sub_nonlinear_1d_fp64 = torch.jit.trace(_back_sub_nonlinear_1d, _example_input_1d_fp64)
_back_sub_nonlinear_no_bias_1d_fp64 = torch.jit.trace(
    _back_sub_nonlinear_no_bias_1d, _example_input_no_bias_1d_fp64
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
