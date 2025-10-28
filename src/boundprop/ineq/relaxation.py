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

    sl = torch.zeros((n, s), dtype=dtype, device=device)
    tl = torch.zeros((n,), dtype=dtype, device=device)
    su = sl.clone()
    tu = tl.clone()
    idx = torch.arange(n, device=device)

    # Non-zero case. All the lower bounds and upper bounds are zero.
    mask_z = (l == 0).all(dim=1) & (u == 0).all(dim=1)

    # Lower relaxation
    # Lower bound is y > x_i, where x_i has the maximum lower bound.
    mask_l = ~mask_z
    sl[idx[mask_l], l_argmax[mask_l]] = 1.0

    # Upper relaxation
    # (1) Trivial case. one lower bound is larger than the other upper bounds.
    # Set the maximum upper bound to negative infinite to exclude it from the
    # max operation.
    mask_t = ~(mask | mask_z)
    su[idx[mask_t], l_argmax[mask_t]] = 1.0

    # (2) Nontrivial cases.
    # Reference: CROWN and DeepPoly
    # y <= u_max
    tu[mask] = u.max(dim=1)[0][mask]

    return sl, su, tl, tu


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
        sl, su, tl, tu = _cal_relaxation_maxpool2d(l, u, l_max, l_argmax, mask)
    elif dtype == torch.float64:
        sl, su, tl, tu = _cal_relaxation_maxpool2d(l, u, l_max, l_argmax, mask)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    sl = sl.reshape((n, s))
    su = su.reshape((n, s))
    tl = tl.flatten()
    tu = tu.flatten()

    return sl, su, tl, tu


#######################################################
## relu relaxation
#######################################################


def _cal_relaxation_relu(l: Tensor, u: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Upper bound
    # y <= x
    su = torch.where(l >= 0, 1.0, 0.0)

    # y <= u / (u - l) (x - l)
    mask_u = (l < 0) & (u > 0)
    su = torch.where(mask_u, u / (u - l), su)
    tu = torch.where(mask_u, -l * su, 0)

    # Lower bound
    # y >= 0 or y >= x
    sl = torch.where(u >= -l, 1.0, 0.0)
    tl = torch.zeros_like(l)

    return sl, su, tl, tu


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

    sl = torch.where(mask0, kl, torch.zeros_like(kl))
    su = sl.clone()

    sl = torch.where(mask1, kl, sl)
    su = torch.where(mask1, klu, su)

    sl = torch.where(mask2, klu, sl)
    su = torch.where(mask2, ku, su)

    sl = torch.where(mask3, torch.minimum(kl, ku), sl)
    su = torch.where(mask3, sl, su)

    tl = -sl * l + yl
    tu = -su * u + yu

    return sl, su, tl, tu


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

    sl = torch.where(mask0, kl, 0.0)
    su = sl.clone()
    tl = torch.where(mask0, yl - sl * l, 0.0)
    tu = tl.clone()

    sl = torch.where(mask1, km, sl)
    su = torch.where(mask1, klu, su)
    tl = torch.where(mask1, ym - sl * m, tl)
    tu = torch.where(mask1, yl - su * l, tu)

    sl = torch.where(mask2, klu, sl)
    su = torch.where(mask2, km, su)
    tl = torch.where(mask2, yu - sl * u, tl)
    tu = torch.where(mask2, ym - su * m, tu)

    sl = torch.where(mask3, kl, sl)
    su = torch.where(mask3, ku, su)
    tl = torch.where(mask3, yl - sl * l, tl)
    tu = torch.where(mask3, yu - su * u, tu)

    return sl, su, tl, tu


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
