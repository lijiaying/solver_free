"""
This module contains the implementation of the specialized ad-hoc back-substitution for
the ReLU activation function.
"""

__docformat__ = ["restructuredtext"]
__all__ = ["relu_back_sub"]

import torch
from torch import Tensor


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
    assert A.dtype in (torch.float32, torch.float64), f"The data type {A.dtype} is not supported."
    assert sl.dim() == su.dim() == 1 and tu.dim() == 1, f"The dimensions are not supported. sl: {sl.shape}, su: {su.shape}, tu: {tu.shape}."
    if is_lower:
        return _back_sub_relu_1d_lower(A, b, sl, su, tu)

    return _back_sub_relu_1d_upper(A, b, sl, su, tu)
