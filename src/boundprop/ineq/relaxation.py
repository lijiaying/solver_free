__docformat__ = ["restructuredtext"]
__all__ = [
    "cal_relaxation_maxpool2d",
    "cal_relaxation_relu",
    "cal_relaxation_sigmoid",
    "cal_relaxation_tanh",
]

from typing import Callable
import torch
from torch import Tensor

from src.utils import (
    RelaxMode,
    DEEPPOLY,
    CROWN,
    STM_SN,
    sigmoid,
    dsigmoid,
    tanh,
    dtanh,
)


#######################################################
## maxpool relaxation
#######################################################


def _cal_relaxation_maxpool2d(
    l: Tensor, u: Tensor, l_max: Tensor, l_argmax: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dtype, device = l.dtype, l.device
    n, s = l.shape

    k_l = torch.zeros((n, s), dtype=dtype, device=device)
    b_l = torch.zeros((n,), dtype=dtype, device=device)
    k_u = k_l.clone()
    b_u = b_l.clone()
    idx = torch.arange(n, device=device)

    # Non-zero case. All the lower bounds and upper bounds are zero.
    mask_z = (l == 0).all(dim=1) & (u == 0).all(dim=1)

    # Lower relaxation
    # Lower bound is y > x_i, where x_i has the maximum lower bound.
    mask_l = ~mask_z
    k_l[idx[mask_l], l_argmax[mask_l]] = 1.0

    # Upper relaxation
    # (1) Trivial case. one lower bound is larger than the other upper bounds.
    # Set the maximum upper bound to negative infinite to exclude it from the
    # max operation.
    mask_t = ~(mask | mask_z)
    k_u[idx[mask_t], l_argmax[mask_t]] = 1.0

    # (2) Nontrivial cases.
    # Reference: CROWN and DeepPoly
    # y <= u_max
    b_u[mask] = u.max(dim=1)[0][mask]

    return k_l, k_u, b_l, b_u


def cal_relaxation_maxpool2d(
    l: Tensor,
    u: Tensor,
    mode: RelaxMode,
    l_max: Tensor,
    l_argmax: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n, s = l.shape
    l_max = l_max.flatten()
    l_argmax = l_argmax.flatten()
    mask = mask.flatten()

    dtype = l.dtype
    if dtype == torch.float32:
        k_l, k_u, b_l, b_u = _cal_relaxation_maxpool2d(l, u, l_max, l_argmax, mask)
    elif dtype == torch.float64:
        k_l, k_u, b_l, b_u = _cal_relaxation_maxpool2d(l, u, l_max, l_argmax, mask)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    k_l = k_l.reshape((n, s))
    k_u = k_u.reshape((n, s))
    b_l = b_l.flatten()
    b_u = b_u.flatten()

    return k_l, k_u, b_l, b_u


#######################################################
## relu relaxation
#######################################################


def _cal_relaxation_relu(l: Tensor, u: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Upper bound
    # y <= x
    k_u = torch.where(l >= 0, 1.0, 0.0)

    # y <= u / (u - l) (x - l)
    mask_u = (l < 0) & (u > 0)
    k_u = torch.where(mask_u, u / (u - l), k_u)
    b_u = torch.where(mask_u, -l * k_u, 0)

    # Lower bound
    # y >= x (when u is large) 
    # or y >= 0 (when l is large)
    k_l = torch.where(u > -l, 1.0, 0.0)
    b_l = torch.zeros_like(l)

    return k_l, k_u, b_l, b_u


def cal_relaxation_relu(
    l: Tensor,
    u: Tensor,
    mode: RelaxMode,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _cal_relaxation_relu(l, u)


#######################################################
## sshape relaxation
#######################################################
def _cal_relaxation_sshape_deeppoly(
    l: Tensor,
    u: Tensor,
    yl: Tensor,
    yu: Tensor,
    kl: Tensor,
    ku: Tensor,
    klu: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mask0: Tensor = u == l
    mask1: Tensor = (u <= 0) & ~mask0
    mask2: Tensor = (l >= 0) & ~mask0
    mask3: Tensor = ~(mask0 | mask1 | mask2)

    k_l = torch.where(mask0, kl, torch.zeros_like(kl))
    k_u = k_l.clone()

    k_l = torch.where(mask1, kl, k_l)
    k_u = torch.where(mask1, klu, k_u)

    k_l = torch.where(mask2, klu, k_l)
    k_u = torch.where(mask2, ku, k_u)

    k_l = torch.where(mask3, torch.minimum(kl, ku), k_l)
    k_u = torch.where(mask3, k_l, k_u)

    b_l = -k_l * l + yl
    b_u = -k_u * u + yu

    return k_l, k_u, b_l, b_u


def _cal_relaxation_sshape(
    l: Tensor,
    u: Tensor,
    yl: Tensor,
    yu: Tensor,
    kl: Tensor,
    ku: Tensor,
    klu: Tensor,
    m: Tensor,
    ym: Tensor,
    km: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mask0: Tensor = u == l
    mask1: Tensor = (ku >= klu) & ~mask0
    mask2: Tensor = (kl >= klu) & ~mask0
    mask3: Tensor = ~(mask0 | mask1 | mask2)

    k_l = torch.where(mask0, kl, 0.0)
    k_u = k_l.clone()
    b_l = torch.where(mask0, yl - k_l * l, 0.0)
    b_u = b_l.clone()

    k_l = torch.where(mask1, km, k_l)
    k_u = torch.where(mask1, klu, k_u)
    b_l = torch.where(mask1, ym - k_l * m, b_l)
    b_u = torch.where(mask1, yl - k_u * l, b_u)

    k_l = torch.where(mask2, klu, k_l)
    k_u = torch.where(mask2, km, k_u)
    b_l = torch.where(mask2, yu - k_l * u, b_l)
    b_u = torch.where(mask2, ym - k_u * m, b_u)

    k_l = torch.where(mask3, kl, k_l)
    k_u = torch.where(mask3, ku, k_u)
    b_l = torch.where(mask3, yl - k_l * l, b_l)
    b_u = torch.where(mask3, yu - k_u * u, b_u)

    return k_l, k_u, b_l, b_u


def cal_relaxation_sshape(
    l: Tensor,
    u: Tensor,
    f: Callable,
    df: Callable,
    mode: RelaxMode,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    yl, yu, kl, ku = f(l), f(u), df(l), df(u)

    klu = (yu - yl) / (u - l)

    if mode == DEEPPOLY:
        return _cal_relaxation_sshape_deeppoly(l, u, yl, yu, kl, ku, klu)

    m = (l + u) * 0.5
    ym = f(m)
    km = df(m)

    if mode in {CROWN, STM_SN}:
        return _cal_relaxation_sshape(l, u, yl, yu, kl, ku, klu, m, ym, km)

    raise ValueError(f"Unsupported mode: {mode}")


def cal_relaxation_sigmoid(
    l: Tensor,
    u: Tensor,
    mode: RelaxMode,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return cal_relaxation_sshape(l, u, sigmoid, dsigmoid, mode)


def cal_relaxation_tanh(
    l: Tensor, u: Tensor, mode: RelaxMode
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return cal_relaxation_sshape(l, u, tanh, dtanh, mode)
