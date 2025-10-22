__docformat__ = ["restructuredtext"]
__all__ = ["cal_scalar_bound"]

import torch
from torch import Tensor


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


_A_fp32 = torch.rand((2, 3), dtype=torch.float32)
_b_fp32 = torch.rand((2,), dtype=torch.float32)
_s1_fp32 = torch.rand((3,), dtype=torch.float32)
_s2_fp32 = torch.rand((3,), dtype=torch.float32)

_A_3d_fp32 = torch.rand((2, 3, 4, 5), dtype=torch.float32)
_b_3d_fp32 = torch.rand((2,), dtype=torch.float32)
_s1_3d_fp32 = torch.rand((3, 4, 5), dtype=torch.float32)
_s2_3d_fp32 = torch.rand((3, 4, 5), dtype=torch.float32)

_A_fp64 = torch.rand((2, 3), dtype=torch.float64)
_b_fp64 = torch.rand((2,), dtype=torch.float64)
_s1_fp64 = torch.rand((3,), dtype=torch.float64)
_s2_fp64 = torch.rand((3,), dtype=torch.float64)

_A_3d_fp64 = torch.rand((2, 3, 4, 5), dtype=torch.float64)
_b_3d_fp64 = torch.rand((2,), dtype=torch.float64)
_s1_3d_fp64 = torch.rand((3, 4, 5), dtype=torch.float64)
_s2_3d_fp64 = torch.rand((3, 4, 5), dtype=torch.float64)


_example_inputs1_fp32 = (_A_fp32, _b_fp32, _s1_fp32, _s2_fp32)
_example_inputs1_no_bias_fp32 = (_A_fp32, _s1_fp32, _s2_fp32)
_example_inputs2_fp32 = (_A_3d_fp32, _b_3d_fp32, _s1_3d_fp32, _s2_3d_fp32)
_example_inputs2_no_bias_fp32 = (_A_3d_fp32, _s1_3d_fp32, _s2_3d_fp32)
_example_inputs1_fp64 = (_A_fp64, _b_fp64, _s1_fp64, _s2_fp64)
_example_inputs1_no_bias_fp64 = (_A_fp64, _s1_fp64, _s2_fp64)
_example_inputs2_fp64 = (_A_3d_fp64, _b_3d_fp64, _s1_3d_fp64, _s2_3d_fp64)
_example_inputs2_no_bias_fp64 = (_A_3d_fp64, _s1_3d_fp64, _s2_3d_fp64)

_cal_scalar_bound_1d_fp32 = torch.jit.trace(_cal_scalar_bound_1d, _example_inputs1_fp32)
_cal_scalar_bound_no_bias_1d_fp32 = torch.jit.trace(
    _cal_scalar_bound_no_bias_1d, _example_inputs1_no_bias_fp32
)
_cal_scalar_bound_3d_fp32 = torch.jit.trace(_cal_scalar_bound_3d, _example_inputs2_fp32)
_cal_scalar_bound_no_bias_3d_fp32 = torch.jit.trace(
    _cal_scalar_bound_no_bias_3d, _example_inputs2_no_bias_fp32
)


_cal_scalar_bound_1d_fp64 = torch.jit.trace(_cal_scalar_bound_1d, _example_inputs1_fp64)
_cal_scalar_bound_no_bias_1d_fp64 = torch.jit.trace(
    _cal_scalar_bound_no_bias_1d, _example_inputs1_no_bias_fp64
)
_cal_scalar_bound_3d_fp64 = torch.jit.trace(_cal_scalar_bound_3d, _example_inputs2_fp64)
_cal_scalar_bound_no_bias_3d_fp64 = torch.jit.trace(
    _cal_scalar_bound_no_bias_3d, _example_inputs2_no_bias_fp64
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
                    f"Unsupported dimension: " f"A: {A.shape} l: {s1.shape}, u: {s2.shape}."
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
                    f"Unsupported dimension: " f"A: {A.shape} l: {s1.shape}, u: {s2.shape}."
                )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}.")
