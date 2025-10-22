"""
This is the base class for calculating the convex hull of an activation function.
The core idea is based on
`ReLU Hull Approximation <https://dl.acm.org/doi/pdf/10.1145/3632917>`__
"""

__docformat__ = "restructuredtext"
__all__ = [
    "ActHull",
    "ReLULikeHull",
    "SShapeHull",
    "ReLUHull",
    "SigmoidHull",
    "TanhHull",
    "get_wrongs"
]

import os
import time
from abc import ABC # , abstractmethod
import sys
from typing import List
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import *

import cdd
import numpy as np
from numpy import ndarray

from ..utils import *
from ..utils.exceptions import Degenerated

_TOL = 1e-4
_MIN_BOUNDS_RANGE = 0.05
_MIN_DLP_ANGLE = 0.1
"""
 The minimum angle between two lines of the DLP function.
 Given two slopes m1 and m2, the angle between them is calculated by arctan(abs((m1 - m2) / (1 + m1 * m2))), 
 but we only use abs((m1 - m2) / (1 + m1 * m2)) to estimate.
"""

Mupper_wrong = 0
Mlower_wrong = 0

def get_wrongs(reset: bool = False) -> tuple[int, int]:
    global Mupper_wrong, Mlower_wrong
    u, l = Mupper_wrong, Mlower_wrong
    if reset:
        Mupper_wrong = 0
        Mlower_wrong = 0
    return u, l

class ActHull(ABC):
    """
    Calculate the convex hull of the activation function.

    :param S: Whether to calculate single-neuron constraints.
    :param M: Whether to calculate M.

    .. tip::
        M here means those constraints that cannot obtained
        by trivial methods or the properties of the activation function.

    :param dtype_cdd: The data type used in pycddlib library.

    .. tip::
        Even though the precision is important when calculating the convex hull,
        we suggest using "float" instead of "fraction" because the calculation is faster
        and can be accepted in most cases. If there is a numerical error, we will raise
        an exception and use "fraction" to recalculate the convex hull.
    """

    __slots__ = [
        "_add_S",
        "_add_M",
        "_dtype_cdd",
    ]

    def __init__(
        self,
        S: bool = False,
        M: bool = True,
        dtype_cdd: CDDNumberType = "float",
    ):
        assert S or M, "At least one of S and M should be True."

        self._add_S = S
        self._add_M = M
        self._dtype_cdd = dtype_cdd

    def cal_hull(
        self,
        constrs: ndarray | None = None,
        lower: ndarray | None = None,
        upper: ndarray | None = None,
    ) -> ndarray:
        """
        Calculate the convex hull of an activation function.
        There are two options:

        1. Calculate S with given input lower and upper bounds. (Two arguments: lower and upper)
        2. Calculate M with given input constraints and input lower and upper bounds. (Three arguments: constrs, lower_bounds, and upper_bounds).
        Some functions require the lower and upper bounds of the input variables, and we suggest to provide them.

        .. tip::
            The input bounds are used to generate a set of constraints that are consistent with the input bounds.

        .. tip::
            The datatype of numpy array is float64 in this function to ensure the precision of the calculation.


        :param constrs: The input constraints.
        :param lower: The lower bounds of the input variables.
        :param upper: The upper bounds of the input variables.
        :return: The constraints defining the convex hull.
        """

        self._check_input_bounds(lower, upper)
        self._check_constrs(constrs)
        self._check_inputs(constrs, lower, upper)

        d = None
        l = u = None
        c_i = c_l = c_u = None
        # Convert the data type to float64 to ensure the precision of the calculation.
        # Make a copy to avoid changing the original data.
        if constrs is not None:
            c_i = np.array(constrs, dtype=np.float64)
            d = c_i.shape[1] - 1

        if lower is not None:
            l = np.array(lower, dtype=np.float64)
            c_l = self._build_bounds_constraints(l, is_lower=True)
            d = l.size

        if upper is not None:
            u = np.array(upper, dtype=np.float64)
            c_u = self._build_bounds_constraints(u, is_lower=False)
            d = u.size

        c = np.empty((0, 1 + d), dtype=np.float64)
        if c_i is not None:
            c = np.vstack((c, c_i))
        if c_l is not None:
            c = np.vstack((c, c_l))
        if c_u is not None:
            c = np.vstack((c, c_u))
        c = np.ascontiguousarray(c)

        try:
            """
            The bounds need update if we use update scalar bounds per layer of DeepPoly. This will cause degenerated input polytope.

            There are two cases:
            (1) One of the input dimension has the same lower and upper bounds, which will throw a Degenerated exception.
            (2) The number of vertices is fewer than the dimension, which will call a Degenerated exception.

            We will first recalculate the vertices with the fractional number if there is an exception. 
            If there is still an exception, we will accept the degenerated input polytope.
            """
            v, dtype_cdd = self._cal_vertices_with_exception(c, l, u, self.dtype_cdd)
            new_l = np.min(v, axis=0)[1:]
            new_u = np.max(v, axis=0)[1:]
            self._check_degenerated_input_polytope(v, new_l, new_u)
            l = new_l
            u = new_u
        except Degenerated:
            v, dtype_cdd = self.cal_vertices(c, "fraction")
            l = np.min(v, axis=0)[1:]
            u = np.max(v, axis=0)[1:]
        except Exception as e:
            raise e

        # if self._add_S and not self._add_M:
        #     SS = self._cal_hull_with_S(l, u)
        #     soundness_check(SS, v, self._f)
        #     print()
        #     return SS

        # el
        if self._add_M:
            return self._cal_hull_with_M(c, l, u)


    def _build_bounds_constraints(self, s: ndarray, is_lower: bool = True) -> ndarray:
        """
        Build the constraints based on the lower or upper bounds of the input variables.

        :param s: The lower or upper bounds of the input variables.

        :return: The constraints of input variables.
        """
        n = s.size

        c = np.zeros((n, n + 1), dtype=s.dtype)
        c[:, 0] = -s if is_lower else s
        idx_row = np.arange(n)
        idx_col = np.arange(1, n + 1)
        c[idx_row, idx_col] = 1.0 if is_lower else -1.0

        return c

    def cal_vertices(
        self,
        c: ndarray,
        dtype_cdd: CDDNumberType,
    ) -> tuple[ndarray, CDDNumberType]:
        """
        Calculate the vertices of a polytope from the constraints.

        .. attention::
            The datatype of cdd is important because the precision may cause an error when calculating the vertices. 
            Sometimes float number is not enough to calculate the vertices, and we need to use the fractional number to
            calculate the vertices.

        .. tip::
            The result of the vertices may have repeated vertices, which is rooted in the algorithm of the pycddlib library.
            Considering removing the repeated vertices is not necessary, we just keep the repeated vertices, and it is not efficient due to the large number of vertices.

        :param c: The constraints of the polytope.
        :param dtype_cdd: The data type used in pycddlib library.

        :return: The vertices of the polytope.
        """
        h_repr = cdd.Matrix(c, number_type=dtype_cdd)
        h_repr.rep_type = cdd.RepType.INEQUALITY

        p = cdd.Polyhedron(h_repr)
        v_repr = p.get_generators()
        v = np.asarray(v_repr, dtype=np.float64)

        return v, dtype_cdd

    def _cal_hull_with_S(
        self, l: ndarray | None, u: ndarray | None
    ) -> ndarray:
        assert l is not None and u is not None, "The lower and upper bounds of the input variables should be provided."
        return self.compute_S(l, u)

    def _cal_hull_with_M(
        self,
        c: ndarray,
        l: ndarray | None = None,
        u: ndarray | None = None,
    ) -> ndarray | None | tuple[ndarray | None, ndarray, ndarray]:
        assert c is not None, "The input constraints should be provided."
        try:
            """
            The bounds need update if we use update scalar bounds per layer of DeepPoly. This will cause degenerated input polytope.

            There are two cases:
            (1) One of the input dimension has the same lower and upper bounds, which will throw a Degenerated exception.
            (2) The number of vertices is fewer than the dimension, which will call a Degenerated exception.

            We will first recalculate the vertices with the fractional number if there is an exception. 
            If there is still an exception, we will accept the degenerated input polytope.
            """
            v, dtype_cdd = self._cal_vertices_with_exception(c, l, u, self.dtype_cdd)
            new_l = np.min(v, axis=0)[1:]
            new_u = np.max(v, axis=0)[1:]
            self._check_degenerated_input_polytope(v, new_l, new_u)
            l = new_l
            u = new_u
        except Degenerated:
            v, dtype_cdd = self.cal_vertices(c, "fraction")
            l = np.min(v, axis=0)[1:]
            u = np.max(v, axis=0)[1:]
        except Exception as e:
            raise e

        if np.min(np.abs(u - l)) < _MIN_BOUNDS_RANGE and len(v) > 2:
            # We don't want to remove trivial cases for the maxpool function (one vertex
            # and one piece).
            return None

        # Update input bounds constraints
        d = l.shape[0]
        c[-2 * d : -d, 0] = -l
        c[-d:, 0] = u

        cc, dtype_cdd = self._cal_constrs_with_exception(c, v, l, u, dtype_cdd)

        # ====================CHECK====================
        # Check if all vertices satisfy the constraints.
        # v_y = self._f(v[:, 1:])
        # vertices = np.hstack((v, v_y))
        # check = np.matmul(cc, vertices.T)
        # if not np.all(check >= -_TOL):
        #     raise RuntimeError("Not all vertices satisfy the constraints.")
        return cc

    def _cal_vertices_with_exception(
        self,
        c: ndarray,
        l: ndarray | None = None,
        u: ndarray | None = None,
        dtype_cdd: CDDNumberType = "float",
    ) -> tuple[ndarray, CDDNumberType]:  # noqa
        d = c.shape[1] - 1
        v = None
        try:
            # Maybe a bug caused by float number and the fractional number will be used.
            v, dtype_cdd = self.cal_vertices(c, dtype_cdd)
            self._check_vertices(d, v)
        except Exception:  # noqa
            try:
                # Change to use the fractional number to calculate the vertices.
                dtype_cdd = "fraction"
                v, dtype_cdd = self.cal_vertices(c, dtype_cdd)  # type: ignore
                self._check_vertices(d, v)
            except Exception as e:
                # This happens when there is an unexpected error.
                self._record_and_raise_exception(e, c, v, l, u)
        return v, dtype_cdd

    def _cal_constrs_with_exception(
        self,
        c: ndarray,
        v: ndarray,
        l: ndarray | None = None,
        u: ndarray | None = None,
        dtype_cdd: CDDNumberType = "float",
    ) -> tuple[ndarray, CDDNumberType]:
        try:
            # Maybe a bug caused by float number and the fractional number will be used.
            output_constrs, dtype_cdd = self.cal_constrs(c, v, l, u, dtype_cdd)
        except Exception:  # noqa
            try:
                output_constrs, dtype_cdd = self.cal_constrs(c, v, l, u, "fraction")
            except Exception as e:
                # Normally, there should not be any error.
                # For debugging, we check and record the error.
                self._record_and_raise_exception(e, c, v, l, u)
        return output_constrs, dtype_cdd

    def cal_constrs(
        self,
        c: ndarray,
        v: ndarray,
        l: ndarray | None = None,
        u: ndarray | None = None,
        dtype_cdd: CDDNumberType = "float",
    ) -> tuple[ndarray, CDDNumberType]:
        """
        Calculate the convex hull of the activation function with a single order of input variables.

        :param c: The constraints of the input polytope.
        :param v: The vertices of the input polytope.
        :param l: The lower bounds of the input variables.
        :param u: The upper bounds of the input variables.
        :param dtype_cdd: The data type used in pycddlib library.

        :return: The constraints defining the convex hull.
        """
        pass

    def compute_S(
        self,
        l: ndarray,
        u: ndarray,
    ) -> ndarray:
        """
        Calculate S of the convex hull.

        .. tip::
            S can be calculated directly with the input lower and upper bounds.

        :param l: The lower bounds of the input variables.
        :param u: The upper bounds of the input variables.
        """
        pass

    def compute_M(
        self,
        c: ndarray,
        v: ndarray,
        l: ndarray | None = None,
        u: ndarray | None = None,
    ) -> ndarray:
        """
        Calculate M of the convex hull.

        .. tip::
            M are calculated based on the input constraints
            and vertices. The lower and upper bounds of the input variables are used to
            check the correctness of the input constraints and vertices. Specifically,
            we can get the lower and upper bounds of the calculated vertices and check
            whether they are consistent with the given input bounds.

        :param c: The constraints of the input polytope.
        :param v: The vertices of the input polytope.
        :param l: The lower bounds of the input variables.
        :param u: The upper bounds of the input variables.

        :return: The constraints defining the convex hull.
        """
        pass

    def _compute_M_with_one_y(self, *args, **kwargs):
        """Calculate the multi-neuron constraint with extending one output dimension."""
        pass

    def _construct_dlp(self, *args, **kwargs):
        """Construct a double-linear-piece (DLP) function as the lower or upper bound of
        the activation function."""
        pass

    def _f(self, x: ndarray | float) -> ndarray | float:
        pass

    def _df(self, x: ndarray | float) -> ndarray | float:
        pass

    def _check_inputs(self, c: ndarray | None, l: ndarray | None, u: ndarray | None):
        if c is not None and l is not None and u is not None:
            assert c.shape[1] - 1 == l.size == u.size, f"The dimensions of the input constraints, lower bounds, and upper bounds should be the same but {c.shape[1] - 1}, {l.size}, and {u.size} are provided."
        assert not (c is None and l is None and u is None), "At least the input constraints, or lower bounds and upper bounds should be provided."

    def _check_constrs(self, c: ndarray | None):
        if c is not None:
            d = c.shape[1] - 1
            assert c.shape[0] >= d + 1, f"n(constraints) should >= n-D + 1. Otherwise, polytope is unbounded. The shape of the input constraints is {c.shape}."

    def _check_input_bounds(self, l: ndarray | None, u: ndarray | None):
        if l is not None and u is not None:
            assert l.ndim == u.ndim == 1, f"The lower and upper bounds of the input variables should be 1-dimensional arrays but {l.ndim} and {u.ndim} are provided."
            assert l.size == u.size, f"The lower and upper bounds of the input variables should have the same size but {l.size} and {u.size} are provided."
            assert np.all(l <= u), f"The lower bounds should be less than the upper bounds but {l} and {u} are provided."

    def _check_vertices(self, dim: int, v: ndarray):  # noqa
        assert len(v) != 0, "Zero vertices. The input polytope is infeasible."
        assert np.all(v[:, 0] == 1.0), f"Unbounded polytope. First column of the vertices should be 1 but {v[:, 0]} are provided."

    def _check_degenerated_input_polytope(self, v: ndarray, l: ndarray, u: ndarray):
        d = v.shape[1] - 1
        if len(v) < d + 1: raise Degenerated(f"The {d}-d input polytope should not be with only {len(v)} vertices.")
        if np.any(np.isclose(l, u)): raise Degenerated(f"The input polytope is degenerated because one of the input dimension has the same lower and upper bounds.")

    def _record_and_raise_exception(
        self, e: Exception, c: ndarray, v: ndarray, l: ndarray | None, u: ndarray | None
    ):
        current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        os.makedirs(".temp", exist_ok=True)
        error_log = f".temp/acthull_{current_time}.log"
        with open(error_log, "w") as f:
            f.write(f"{self.__class__.__name__}\n")
            f.write(f"Exception: {e}\n")
            f.write(f"Created time: {current_time}\n")
            if c is not None:
                f.write(f"Input constraints shape: {c.shape}\n")
                f.write(f"Input constraints: {c.tolist()}\n")
            if l is not None:
                f.write(f"Input constraints lower bounds: {l.tolist()}\n")
            if u is not None:
                f.write(f"Input constraints upper bounds: {u.tolist()}\n")
            f.write(f"Input vertices shape: {v.shape}\n")
            if v is not None and len(v) > 0:
                v_l = np.min(v, axis=0)[1:]
                v_u = np.max(v, axis=0)[1:]
                f.write(f"Input vertices: {v.tolist()}\n")
                f.write(f"Input vertices lower bounds: {v_l.tolist()}\n")
                f.write(f"Input vertices upper bounds: {v_u.tolist()}\n")
        raise RuntimeError(f"Error: {e}. Please check the log: {error_log}")

    @property
    def dtype_cdd(self) -> CDDNumberType:
        """The data type used in pycddlib library."""
        return self._dtype_cdd


def compute_M_with_one_y_dlp(
    idx: int,
    c: ndarray,
    v: ndarray,
    aux_lines: ndarray,
    aux_point: float | None,
    is_convex: bool = True,
) -> tuple[ndarray, ndarray]:
    """
    Calculate M for one specified input dimension of the
    convex hull for the DLP (double linear pieces) function.

    :param idx: The index of the input dimension.
    :param c: The constraints of input polytope.
    :param v: The vertices of input polytope.
    :param aux_point: The auxiliary point that is the intersection of the two linear
        pieces.
    :param aux_lines: The auxiliary line where the two linear pieces are located.
        Each row represents a line. If there is only one line, it is the trivial case
        (linear function), which means we do not need a DLP function. If there are two
        lines, the first line is the left line and the second line is the right line.
    :param is_convex: Whether the DLP function is defined by max or min. If it is max,
        then the function is convex and is_convex is True; otherwise, it is False.

    :return: The output constraints and vertices of the convex hull after extending
        one specified output dimension.

    :raises RuntimeError: If the auxiliary point is not provided and the auxiliary lines
        should have only one line to represent the trivial case (linear function).
    :raises ArithmeticError: If the vertices should not all greater/smaller than the
        auxiliary point or divided by zero during calculation.
    """
    pad_width = ((0, 0), (0, 1))  # Extend one output dimension that is a new column.

    if aux_point is None:
        # The auxiliary point is not provided. The auxiliary lines should have only one
        # line to represent the trivial case (linear function).
        # Only need to extend the constraints and vertices in output space.
        if aux_lines.shape[0] != 1:
            raise RuntimeError(
                "The auxiliary point is not provided and the auxiliary "
                "lines should have only one line to represent the "
                "trivial case (linear function)."
            )

        line = aux_lines[[-1]]
        v = np.hstack((v, np.matmul(v, line[:, :-1].T)))
        c = np.pad(c, pad_width)
        c = np.vstack((c, line)) if is_convex else np.vstack((c, -line))
        return c, v

    v = np.pad(v, pad_width)
    c = np.pad(c, pad_width)

    vc = v[:, idx + 1]
    mask_vl, mask_vr = (vc < aux_point), (vc > aux_point)

    if not np.any(mask_vl) or not np.any(mask_vr):
        raise RuntimeError(
            "The vertices should not all greater/smaller than the auxiliary point."
        )

    ll, lr = aux_lines[[[0], [1]]] if is_convex else aux_lines[[[1], [0]]]

    vl, vr = v[mask_vl], v[mask_vr]
    v[mask_vl, -1], v[mask_vr, -1] = np.matmul(vl, ll.T).T, np.matmul(vr, lr.T).T
    vl, vr = v[mask_vl], v[mask_vr]
    d1, d2 = np.matmul(c, vr.T), np.matmul(c, vl.T)
    h1, h2 = np.matmul(ll, vr.T), np.matmul(lr, vl.T)

    if np.any(h1 == 0) or np.any(h2 == 0):
        raise RuntimeError("Zero values will be in denominators.")

    # if is_convex:
    #     assert np.all(h1 <= 0), f"{h1}"
    #     assert np.all(h2 <= 0), f"{h2}"
    # else:
    #     assert np.all(h1 >= 0), f"{h1}"
    #     assert np.all(h2 >= 0), f"{h2}"

    d1 /= h1
    d2 /= h2

    if is_convex:
        beta1 = np.max(d1, axis=1, keepdims=True)
        beta2 = np.max(d2, axis=1, keepdims=True)
    else:
        beta1 = np.min(d1, axis=1, keepdims=True)
        beta2 = np.min(d2, axis=1, keepdims=True)

    # beta1 = np.maximum(beta1, 0)
    # beta2 = np.maximum(beta2, 0)

    c -= beta1 * ll
    c -= beta2 * lr

    # assert np.all(c @ vl.T >= -_TOL)

    return c, v


class ReLULikeHull(ActHull, ABC):
    """
    This is the base class for the ReLU-like activation functions to calculate the
    convex hull.
    """

    def cal_constrs(
        self,
        c: ndarray,
        v: ndarray,
        l: ndarray | None = None,
        u: ndarray | None = None,
        dtype_cdd: CDDNumberType = "float",
    ) -> tuple[ndarray, CDDNumberType]:

        d = c.shape[1] - 1
        c = np.array(c, dtype=np.float64)
        l = np.array(l, dtype=np.float64)
        u = np.array(u, dtype=np.float64)
        cc = np.empty((0, 1 + 2 * d), dtype=np.float64)

        if self._add_S:
            c1 = self.compute_S(l, u)
            cc = np.vstack((cc, c1))

        if self._add_M:
            c2 = self.compute_M(c, v, l, u)
            cc = np.vstack((cc, c2))

        return cc, dtype_cdd

    def compute_M(
        self,
        c: ndarray,
        v: ndarray,
        l: ndarray | None = None,
        u: ndarray | None = None,
    ) -> ndarray:
        d = c.shape[1] - 1

        for i in range(d):
            lines, point = self._construct_dlp(i, d, l[i], u[i])
            c, v = self._compute_M_with_one_y(i, c, v, lines, point, True)

        return c

    def _compute_M_with_one_y(
        self,
        idx: int,
        c: ndarray,
        v: ndarray,
        dlp_lines: ndarray,
        dlp_point: float,
        is_convex: bool,
    ) -> tuple[ndarray, ndarray]:
        return compute_M_with_one_y_dlp(
            idx, c, v, dlp_lines, dlp_point, is_convex=is_convex
        )

    def _construct_dlp(self, *args, **kwargs):
        pass


class ReLUHull(ReLULikeHull):
    """
    This is to calculate the convex hull for the rectified linear unit (ReLU)
    activation function.

    .. tip::
        This is an ad hoc implementation for ReLU to obtain the convex hull
        considering high efficiency and accuracy based on the two linear pieces (
        :math:`y=x` and :math:`y=0`) of ReLU.
    """

    _lower_constraints: dict[int, ndarray] = {}

    def compute_S(self, l: ndarray, u: ndarray) -> ndarray:
        """
        Calculate S of the convex hull for ReLU.
        We use *triangle relaxation* to calculate S of ReLU.

        :param l: The lower bounds of the input variables.
        :param u: The upper bounds of the input variables.
        :return: S of the convex hull.
        """
        assert np.all(l<0) and np.all(u>0), f"l<0, u>0 must always hold for non-trivial case. {l}, {u}"

        d = l.shape[0]
        c = np.zeros((d, 2 * d + 1), dtype=np.float64)

        idx_r = np.arange(d)  # The index of the rows
        idx_x = np.arange(1, d + 1)  # The index of the input variables
        idx_y = np.arange(d + 1, 2 * d + 1)  # The index of the output variables

        # For the upper faces.
        # The output constraints have the form of -u*l + u*x - (u-l)*y >= 0.
        c[:, 0] = -u * l
        c[idx_r, idx_x] = u
        c[idx_r, idx_y] = -(u - l)

        # For the lower faces.
        if self._lower_constraints.get(d) is None:
            # The output constraints have the form of y >= 0.
            c_l1 = np.zeros((d, 2 * d + 1), dtype=np.float64)
            c_l1[idx_r, idx_y] = 1.0

            # The output constraints have the form of y >= x.
            c_l2 = np.zeros((d, 2 * d + 1), dtype=np.float64)
            c_l2[idx_r, idx_x] = -1.0
            c_l2[idx_r, idx_y] = 1.0

            cl = np.vstack((c_l1, c_l2))
            self._lower_constraints[d] = cl
        else:
            cl = self._lower_constraints[d]

        c = np.vstack((c, cl))

        return c

    def compute_M(
        self,
        c: ndarray,
        v: ndarray,
        l: ndarray | None = None,
        u: ndarray | None = None,
    ) -> ndarray:
        """
        Calculate M of the convex hull.

        :param c: The constraints of the input polytope.
        :param v: The vertices of the input polytope.
        :param l: The lower bounds of the input variables.
        :param u: The upper bounds of the input variables.

        :return: M of the convex hull.
        """
        dim = c.shape[1] - 1

        v = np.transpose(v)
        mask_xp, mask_xn = (v > _TOL), (v < -_TOL)
        if not np.any(mask_xp) and not np.any(mask_xn):
            raise RuntimeError("The vertices should not all positive or all negative.")

        y = np.maximum(v, 0)  # The vertices coordinate in output space.
        cv = np.matmul(c, v)

        s = (c.shape[0], 1)
        beta1 = np.zeros(s, dtype=np.float64)
        beta2 = np.zeros(s, dtype=np.float64)

        for i in range(1, dim + 1):
            mask_xp_i, mask_xn_i = mask_xp[i], mask_xn[i]

            if np.any(mask_xp_i):
                temp = cv[:, mask_xp_i] / v[i, mask_xp_i]
                beta1[:, 0] = -np.min(temp, axis=1)

            if np.any(mask_xn_i):
                temp = cv[:, mask_xn_i] / v[i, mask_xn_i]
                beta2[:, 0] = np.max(temp, axis=1)

            # Eliminate tiny positive values
            # beta1 = np.minimum(beta1, 0)
            # beta2 = np.minimum(beta2, 0)

            c = np.hstack((c, beta1 + beta2))
            c[:, [i]] -= beta2

            v = np.vstack((v, y[i]))
            cv += np.outer(c[:, -1], y[i]) + np.outer(-beta2, v[i])

            beta1.fill(0.0)
            beta2.fill(0.0)

        return c

    def _compute_M_with_one_y(
        self,
        idx: int,
        c: ndarray,
        v: ndarray,
        dlp_lines: ndarray,
        dlp_point: float,
        is_convex: bool,
    ) -> tuple[ndarray, ndarray]:
        raise NotSupported()

    def _construct_dlp(self, *args, **kwargs):
        raise NotSupported()

    def _f(self, x: ndarray | float) -> ndarray | float:
        return relu(x)

    def _df(self, x: ndarray | float) -> ndarray | float:
        return drelu(x)


class SShapeHull(ActHull, ABC):
    """
    This is the base class for the S-shaped functions to calculate the convex hull.

    The S-shaped functions include the sigmoid, hyperbolic tangent, etc.

    .. tip::
        Overall, to calculate the convex hull of the S-shaped functions, we
        construct two *double-linear-piece* (DLP) functions as the upper and lower
        bounds of the activation function. We take the upper constraints of the upper
        DLP function and the lower constraints of the lower DLP function as the M.

    .. tip::
        The constraints construction of the S-shaped functions is based on some tangent lines of the activation function. 
        The tangent lines are calculated in an iterative way, resulting it is slower than the ReLU-like activation functions.

        Refer to the paper:
        `Efficient Neural Network Verification via Adaptive Refinement and Adversarial Search
        <https://ecai2020.eu/papers/384_paper.pdf>`__:cite:`henriksen_efficient_2020`
        for numerically calculating the tangent lines of Sigmoid and Tanh functions.
    """

    def cal_constrs(
        self,
        c: ndarray,
        v: ndarray,
        l: ndarray | None = None,
        u: ndarray | None = None,
        dtype_cdd: CDDNumberType = "float",
    ) -> tuple[ndarray, CDDNumberType]:

        d = c.shape[1] - 1
        c = np.array(c, dtype=np.float64)
        l = np.array(l, dtype=np.float64)
        u = np.array(u, dtype=np.float64)
        cc = np.empty((0, 2 * d + 1), dtype=np.float64)

        if self._add_S and not self._add_M:
            c_s = self.compute_S(l, u)
            cc = np.vstack((cc, c_s))

        if self._add_M:
            c_m = self.compute_M(c, v, l, u)
            cc = np.vstack((cc, c_m))

        return cc, dtype_cdd

    def compute_S(
        self,
        l: ndarray,
        u: ndarray,
    ) -> ndarray:

        d = l.shape[0]
        cc = np.empty((0, 1 + d), dtype=np.float64)

        f, df = self._f, self._df
        xl, xu = l, u
        yl, yu, kl, ku = f(xl), f(xu), df(xl), df(xu)
        klu = (yu - yl) / (xu - xl)

        for i in range(d):
            args = (i, d, xl[i], xu[i], yl[i], yu[i], kl[i], ku[i], klu[i], cc)
            _, _, _, _, cc = self._construct_dlp(*args, self._add_S)

        return cc

    def compute_M(
        self,
        c: ndarray,
        v: ndarray,
        l: ndarray | None = None,
        u: ndarray | None = None,
    ) -> ndarray:
        assert l is not None and u is not None, "The lower and upper bounds should be provided for the S-shape activation function."

        d = c.shape[1] - 1
        # S
        cc_s = np.empty((0, 1 + d), dtype=np.float64)
        # M providing lower/upper output bounds
        cc_l, cc_u = c, c.copy()
        vv = v.copy()
        v_l, v_u = v, v.copy()

        f, df = self._f, self._df
        xl, xu = l, u
        yl, yu, kl, ku = f(xl), f(xu), df(xl), df(xu)
        klu = (yu - yl) / (xu - xl)

        for i in range(d):
            args = (i, d, xl[i], xu[i], yl[i], yu[i], kl[i], ku[i], klu[i], cc_s)
            dlp_lines_l, dlp_lines_u, dlp_point_l, dlp_point_u, cc_s = self._construct_dlp(*args, self._add_S)

            if self._add_M:
                cc_l, v_l = self._compute_M_with_one_y(i, cc_l, v_l, dlp_lines_l, dlp_point_l, is_convex=False)
                cc_u, v_u = self._compute_M_with_one_y(i, cc_u, v_u, dlp_lines_u, dlp_point_u, is_convex=True)
                # print(f'l: v: {v_l.shape}, c: {cc_l.shape}; u: v: {v_u.shape}, c: {cc_u.shape}')

                vvv = np.vstack((v_l, v_u))
                ccc = np.vstack((cc_l, cc_u))
                cc_l, cc_u = ccc, ccc
                v_l, v_u = vvv, vvv

        cc = np.empty((0, 2 * d + 1), dtype=np.float64)

        if self._add_S:
            cc = np.vstack((cc, cc_s))
            # print(f'\n{YELLOW} check S: {RESET}', end=' ')
            # soundness_check(cc_s, vv, self._f)

        if self._add_M:
            cc = np.vstack((cc, cc_l, cc_u))
            # print(f'{YELLOW} check M: {RESET}')
            # print(f'{cc}')
            global Mlower_wrong, Mupper_wrong
            print(f'\n{YELLOW} check M lower: {RESET}', end=' ')
            sound = soundness_check(cc_l, vv, self._f)
            if not sound:
                Mlower_wrong += 1
            print(f'\n{YELLOW} check M upper: {RESET}', end=' ')
            sound = soundness_check(cc_u, vv, self._f)
            if not sound:
                Mupper_wrong += 1
            # soundness_check(cc, vv, self._f)
            # def soundness_check(X, V, F=None):
                # if F is None:
                #     F = self._f
                # return np.all(F(X) <= V)

        return cc

    def _compute_M_with_one_y(
        self,
        idx: int,
        c: ndarray,
        v: ndarray,
        dlp_lines: ndarray | ndarray,
        dlp_point: float | None,
        is_convex: bool,
    ) -> tuple[ndarray, ndarray]:
        # if is_convex:
        #     v[:,0] -= 1e-6  # To avoid numerical issue
        # else:
        #     v[:,0] += 1e-6  # To avoid numerical issue
        return compute_M_with_one_y_dlp(
            idx, c, v, dlp_lines, dlp_point, is_convex=is_convex
        )

    def _construct_dlp(
        self, idx: int, dim: int,
        xli: float, xui: float,
        yli: float, yui: float,
        kli: float, kui: float, klui: float,
        c: ndarray,
        add_s: bool,
    ) -> tuple[
        ndarray | None, ndarray | None, float | None, float | None, ndarray | None
    ]:
        """
        Calculate the auxiliary lines, auxiliary point, and S.
        There are three cases:
        1. One linear function as the lower bound and one DLP function as the upper bound.
        2. One DLP function as the lower bound and one linear function as the upper bound.
        3. Two DLP functions as the lower and upper bounds.

        :param idx: The index of dimension to extend in output space.
        :param dim: The dimension of input space.
        :param xli: The lower bound of the input variable.
        :param xui: The upper bound of the input variable.
        :param yli: The lower bound of the output variable.
        :param yui: The upper bound of the output variable.
        :param kli: The slope of the lower bound of the input variable.
        :param kui: The slope of the upper bound of the input variable.
        :param klui: The slope of the linear piece connecting the lower and upper bounds.
        :param c: S of the output polytope.
        :param add_s: Whether to add S.

        :return: The auxiliary lines, auxiliary point, and S.
        """

        if np.allclose(xli, xui):
            # Handle the degenerate case
            c1 = np.zeros((1, idx + dim + 2), dtype=np.float64)
            c2 = np.zeros((1, idx + dim + 2), dtype=np.float64)
            c1[:, 0] = [-kli * xli + yli]
            c1[:, idx + 1] = [kli]
            c1[:, -1] = -1.0
            c2[:, 0] = [-kui * xli + yli]
            c2[:, idx + 1] = [kui]
            c2[:, -1] = -1.0
            c = np.hstack((c, np.zeros((c.shape[0], 1))))
            return c1, c2, None, None, c
        elif kui > klui:
            resolve_case = self._dlp_case1
        elif kli > klui:
            resolve_case = self._dlp_case2
        else:
            resolve_case = self._dlp_case3

        c = np.hstack((c, np.zeros((c.shape[0], 1))))
        args = (idx, dim, xli, xui, yli, yui, kli, kui, klui, c)
        return resolve_case(*args, add_s)


    def _dlp_case1(
        self, idx: int, dim: int,
        xli: float, xui: float,
        yli: float, yui: float,
        kli: float, kui: float, klui: float,
        c: ndarray,
        add_s: bool,
    ) -> tuple[ndarray, ndarray, None, float, ndarray | None]:
        """
        the slope of the upper linear piece is larger than the slope of the linear piece connecting the lower and upper bounds.
        """
        f = self._f
        blu2, klu2, su = self._get_parallel_tangent_line(klui, get_big=False)
        kp1, kli = (yli - f(su)) / (xli - su), kli
        bp1, bli = yli - kp1 * xli, yli - kli * xli

        if xui > 0:
            x = np.asarray([xli, su, xui], dtype=np.float64)
            b, k, _ = self._get_second_tangent_line(x, get_big=True)
            blui, bp2, bui = b
            klui, kp2, kui = k
        else:
            kp2, klui, kui = (yui - f(su)) / (xui - su), klui, kui
            bp2, blui, bui = yui - kp2 * xui, yli - klui * xli, yui - kui * xui

        aux_lines_l = np.zeros((1, idx + dim + 2), dtype=np.float64)
        aux_lines_l[:, 0] = [bli]
        aux_lines_l[:, idx + 1] = [kli]
        aux_lines_l[:, -1] = -1.0
        aux_point_l = None

        if abs((kp1 - kp2) / (1 - kp1 * kp2)) < _MIN_DLP_ANGLE:
            aux_lines_u = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [yli - klui * xli]
            aux_lines_u[:, idx + 1] = [klui]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = None
        else:
            aux_lines_u = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [bp1, bp2]
            aux_lines_u[:, idx + 1] = [kp1, kp2]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = su

        if add_s:
            temp = np.zeros((4, idx + dim + 2), dtype=np.float64)
            temp[:, 0] = [blui, -bli, -bui, -blu2]
            temp[:, idx + 1] = [klui, -kli, -kui, -klu2]
            temp[:, -1] = [-1.0, 1.0, 1.0, 1.0]
            c = np.vstack((c, temp))

        return aux_lines_l, aux_lines_u, aux_point_l, aux_point_u, c

    def _dlp_case2(
        self, idx: int, dim: int,
        xli: float, xui: float,
        yli: float, yui: float,
        kli: float, kui: float, klui: float,
        c: ndarray,
        add_s: bool,
    ) -> tuple[ndarray, ndarray, float, None, ndarray | None]:
        """
        the slope of the lower linear piece is larger than the slope of the linear piece connecting the lower and upper bounds.
        """
        f = self._f
        blu2, klu2, sl = self._get_parallel_tangent_line(klui, get_big=True)
        kp1, kui = (yui - f(sl)) / (xui - sl), kui
        bp1, bui = yui - kp1 * xui, yui - kui * xui

        if xli < 0:
            x = np.asarray([xui, sl, xli], dtype=np.float64)
            b, k, _ = self._get_second_tangent_line(x, get_big=False)
            blui, bp2, bli = b
            klui, kp2, kli = k
        else:
            kp2, klui, kli = (yli - f(sl)) / (xli - sl), klui, kli
            bp2, blui, bli = yli - kp2 * xli, yui - klui * xui, yli - kli * xli

        if abs((kp1 - kp2) / (1 - kp1 * kp2)) < _MIN_DLP_ANGLE:
            aux_lines_l = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [yui - klui * xui]
            aux_lines_l[:, idx + 1] = [klui]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = None
        else:
            aux_lines_l = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [bp1, bp2]
            aux_lines_l[:, idx + 1] = [kp1, kp2]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = sl

        aux_lines_u = np.zeros((1, idx + dim + 2), dtype=np.float64)
        aux_lines_u[:, 0] = [bui]
        aux_lines_u[:, idx + 1] = [kui]
        aux_lines_u[:, -1] = -1.0
        aux_point_u = None

        if add_s:
            temp = np.zeros((4, idx + dim + 2), dtype=np.float64)
            temp[:, 0] = [blui, -bli, -bui, -blu2]
            temp[:, idx + 1] = [klui, -kli, -kui, -klu2]
            temp[:, -1] = [1.0, -1.0, -1.0, -1.0]
            c = np.vstack((c, temp))

        return aux_lines_l, aux_lines_u, aux_point_l, aux_point_u, c

    def _dlp_case3(
        self, idx: int, dim: int,
        xli: float, xui: float,
        yli: float, yui: float,
        kli: float, kui: float, klui: float,
        c: ndarray,
        add_s: bool,
    ) -> tuple[ndarray, ndarray, float, float, ndarray | None]:
        """
        (1) the slope of the upper linear piece is smaller than the slope of the linear piece connecting the lower and upper bounds, and 
        (2) the slope of the lower linear piece is smaller than the slope of the linear piece connecting the lower and upper bounds.
        """
        f = self._f
        blul, klul, su = self._get_parallel_tangent_line(klui, get_big=False)
        bluu, kluu, sl = self._get_parallel_tangent_line(klui, get_big=True)

        x_temp = np.asarray([xli, su], dtype=np.float64)
        b_temp, k_temp, _ = self._get_second_tangent_line(x_temp, get_big=True)
        btu, bp2u = b_temp
        ktu, kp2u = k_temp
        x_temp = np.asarray([xui, sl], dtype=np.float64)
        b_temp, k_temp, _ = self._get_second_tangent_line(x_temp, get_big=False)
        btl, bp2l = b_temp
        ktl, kp2l = k_temp

        kp1u, kp1l = (yli - f(su)) / (xli - su), (yui - f(sl)) / (xui - sl)
        bp1u, bp1l = yli - kp1u * xli, yui - kp1l * xui

        if abs((kp1u - kp2u) / (1 - kp1u * kp2u)) < _MIN_DLP_ANGLE:
            aux_lines_u = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [yli - klui * xli]
            aux_lines_u[:, idx + 1] = [klui]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = None
        else:
            aux_lines_u = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_u[:, 0] = [bp1u, bp2u]
            aux_lines_u[:, idx + 1] = [kp1u, kp2u]
            aux_lines_u[:, -1] = -1.0
            aux_point_u = su

        if abs((kp1l - kp2l) / (1 - kp1l * kp2l)) < _MIN_DLP_ANGLE:
            aux_lines_l = np.zeros((1, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [yui - klui * xui]
            aux_lines_l[:, idx + 1] = [klui]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = None
        else:
            aux_lines_l = np.zeros((2, idx + dim + 2), dtype=np.float64)
            aux_lines_l[:, 0] = [bp1l, bp2l]
            aux_lines_l[:, idx + 1] = [kp1l, kp2l]
            aux_lines_l[:, -1] = -1.0
            aux_point_l = sl

        if add_s:
            bli, bui = yli - kli * xli, yui - kui * xui
            temp = np.zeros((6, idx + dim + 2), dtype=np.float64)
            temp[:, 0] = [bui, btu, bluu, bli, btl, blul]
            temp[:, idx + 1] = [kui, ktu, kluu, kli, ktl, klul]
            temp[:, -1] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
            c = np.vstack((c, temp))

        return aux_lines_l, aux_lines_u, aux_point_l, aux_point_u, c

    def _get_second_tangent_line(self, x1: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        """
        Get the second tangent line given a point x1, which is not the tangent line
        taking x1 as tangent point.

        :param x1: The point where the tangent line is not taken.
        :param get_big: Whether to get the tangent line with a larger slope.

        :return: The bias, slope, and the tangent point of the tangent line.
        """

        pass

    def _get_parallel_tangent_line(self, k: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        """
        Get the parallel tangent line given the slope.

        :param k: The slope of the tangent line.
        :param get_big: Whether to get the tangent line with a larger slope.

        :return: The bias, slope, and the tangent point of the tangent line.
        """
        pass


class SigmoidHull(SShapeHull):
    def _get_second_tangent_line(self, x1: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        return get_second_tangent_line(x1, get_big, "sigmoid")

    def _get_parallel_tangent_line(self, k: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        return get_parallel_tangent_line(k, get_big, "sigmoid")

    def _f(self, x: ndarray | float) -> ndarray | float:
        return sigmoid(x)

    def _df(self, x: ndarray | float) -> ndarray | float:
        return dsigmoid(x)


class TanhHull(SShapeHull):
    def _get_second_tangent_line(self, x1: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        return get_second_tangent_line(x1, get_big, "tanh")

    def _get_parallel_tangent_line(self, k: float | ndarray, get_big: bool | ndarray
    ) -> tuple[float | ndarray, float | ndarray, float | ndarray]:
        return get_parallel_tangent_line(k, get_big, "tanh")

    def _f(self, x: ndarray | float) -> ndarray | float:
        return tanh(x)

    def _df(self, x: ndarray | float) -> ndarray | float:
        return dtanh(x)




def __str_list(floatlist: List[float], precision=3, sep=", ") -> str:
    strlist = [
        f"{f:n}" if float(f).is_integer() else f"{f:0{precision}f}" for f in floatlist
    ]
    return "[" + sep.join(strlist) + "]"


def str_list(floatlist: List[float] | List[List[float]], precision=3, sep=", ") -> str:
    if isinstance(floatlist[0], list):
        strlist = [str_list(fl, precision, sep) for fl in floatlist]
        return "[" + (sep + " ").join(strlist) + "]"
    return __str_list(floatlist, precision, sep)



# def sample_more_points(V, l):
#     # V: mxn, n points, each with m dimension
#     # output: mxl, l points, each with m dimension
#     # ratios R: nxl, each column should sum to 1, and all the values should be postive
#     m, n = V.shape
#     R = np.random.rand(n, l)
#     # if the element in R is negative, then make it positive
#     R = np.abs(R) + 1e-8
#     R = R / np.sum(R, axis=0)
#     return np.dot(V, R)


tol = 1e-3
SampleMore = 0
def soundness_check(X, V, F=None):
    # # print('** check whether H separate V **')
    
    # if F is not None and X.shape[1] != V.shape[0]:
    #     # if SampleMore > 0:
    #     #     Vmore = sample_more_points(V, X.shape[1] * SampleMore)
    #     #     V = np.hstack((V, Vmore))
    #     V = np.vstack((V, F(V[1:])))
    V = V.transpose()
    if F is not None and X.shape[1] != V.shape[0]:
        V = np.vstack((V, F(V[1:])))
    if X.shape[1] != V.shape[0]:
        V = V[: X.shape[1]]
    Vmore = np.zeros((V.shape[0], 0))
    # print('-'*60)
    # print('X:', X.shape, '\n', X, sep='', flush=True)
    # print('V:', V.shape, '\n', V, sep='', flush=True)
    XV = np.dot(X, V)
    threshold = -3e2 * tol
    # threshold = -1E-8
    # threshold = -1E-4
    threshold = -tol
    # print('Using threshold:', threshold)
    XV_pos = np.sum(XV >= threshold)
    XV_neg = np.sum(XV < threshold)
    Loc_neg = np.where(XV < threshold)
    # print('threshold:', threshold)
    # print('[H*V]:\n', normalize_numpy_array(H_V))
    # print('->', ('(+' + str(XV_pos) + ',-' + str(XV_neg) + ')').ljust(10), end='', sep='', flush=True)
    Vstr = "+" + str(Vmore.shape[1]) if SampleMore > 0 else ""
    print(f"{BLUE}*Sound*{RESET} {GRAY}[{Vstr}]{RESET} {str(X.shape)}*{str(V.shape)} => (+{XV_pos},-{XV_neg})".ljust(60), end="", flush=True,)
    if XV_neg > 0:
        print(f'{RED_BK}[ERROR]{RESET} {str_list(list(XV[XV < threshold])[:20], 8)}', end="")
        # if XV_neg > 20:
        #     print("... and", XV_neg - 20, "more ...", end="")
        # if True:
        #     print("\n-- More Info --")
        #     print("Locations:", Loc_neg)
        #     for i, (row, col) in enumerate(zip(Loc_neg[0], Loc_neg[1])):
        #         if i > 5:
        #             break
        #         print(f'{CYAN}|----{i} loc:{(row, col)} values:{XV[row, col]}  <<== X[{row}] * V[{col}]{V[:, col].reshape((-1))} {RESET}')
    else:
        print(GREEN_BK, "[PASS]", RESET, sep="", end="")
    return XV_neg == 0


