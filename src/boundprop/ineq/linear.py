"""
This module contains the implementing the approach of bound propagation with linear
inequalities.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "BasicIneqNode",
    "InputIneqNode",
    "LinearIneqNode",
    "GemmIneqNode",
    "Conv2DIneqNode",
    "NonLinearIneqNode",
    "ReLUIneqNode",
    "SigmoidIneqNode",
    "TanhIneqNode",
    "MaxPool2DIneqNode",
    "ResidualAddIneqNode",
]

import math
from abc import abstractmethod, ABC

import torch
from torch import Tensor

from .backsub import *
from .relaxation import *
from ..base import *
from ..containers import BPSharedData
from ...utils import *


class BasicIneqNode(BasicNode, ABC):
    """
    The basic node for bound propagation with linear inequalities.

    :param name: The name of the layer.
    :param input_names: The names of the input layers. Residual blocks have multiple
        input layers.
    :param input_size: The size of the input.
    :param shared_data: The shared data among all nodes.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
    ):
        BasicNode.__init__(self, name, input_names, input_size, shared_data)
        # This is used for storing the initial constraint bound for the layer, which is
        # used to calculate the bounds of neurons in the layer.
        # This is to reduce the computation when calculate the bounds of neurons of
        # the layer.
        self._cached_init_constr: LinearConstr | None = None

    @abstractmethod
    def forward(
        self,
        input_bound: ScalarBound,
        only_lower_bound: bool = False,
    ) -> tuple[ScalarBound | None, Tensor | None]:
        """
        Calculate the scalar bounds of the neurons in the layer by the linear relaxation
        of the layer or construct the linear relaxation of the layer.

        If the next layer is a non-linear layer, the scalar bounds of the neurons in the
        layer are calculated.

        If the current layer is a non-linear layer, the linear relaxation of the layer
        is constructed.

        :return: The scalar bounds of the neurons in the layer.
        """
        pass

    def init_constr_bound(self, only_lower_bound: bool = False) -> LinearConstrBound:
        """
        Generate the linear inequalities for back-substitution.

        The linear inequalities are in form of a matrix and a bias.
        By default, it returns an identity matrix and a zero bias, representing the equality constraints 
        (coefficients and constants) of the current layer.
        This represents the variables is represented by themselves.

        .. tip::
            Here a cache is used to store the initial linear relaxation of the layer to
            reduce the computation.

        :param only_lower_bound: When enabled, return only the lower bound.

        :return: The initial linear relaxation of the layer.
        """
        if self._cached_init_constr is not None:
            if only_lower_bound:
                return LinearConstrBound(L=self._cached_init_constr)
            return LinearConstrBound(L=self._cached_init_constr, U=self._cached_init_constr)

        constr = LinearConstr(
            A=torch.eye(
                math.prod(self.output_size),
                dtype=self.shared_data.dtype,
                device=self.shared_data.device,
            )
        )

        self._cached_init_constr = constr.detach().clone()

        if only_lower_bound:
            return LinearConstrBound(L=constr)
        return LinearConstrBound(L=constr, U=constr)

    def back_sub_to_input(self, constr_bound: LinearConstrBound) -> LinearConstrBound:
        """
        Back-substitute the linear relaxation of the layer to the input layer and
        obtain a linear relaxation represented by input variables.

        :param constr_bound: The linear relaxation to back-substitute.

        :return: The linear relaxation represented by input variables.
        """
        return back_sub_to_input(self, constr_bound)

    @abstractmethod
    def back_sub_once(self, constr_bound: LinearConstrBound) -> LinearConstrBound:
        """
        Back-substitute the linear relaxations to the previous layers.

        :param constr_bound: The linear relaxation.

        :return: The linear constraints only involving the variables of the
            preceding layer.
        """
        pass

    @staticmethod
    def cal_bounds(
        constr_bound: LinearConstrBound, scalar_bound: ScalarBound
    ) -> tuple[ScalarBound, Tensor | None]:
        """
        Calculate the scalar bounds of the given linear inequalities by the given
        scalar bounds. Also, return the point that makes the lower bound minimal.

        :param constr_bound: The linear inequalities represented by specified variables.
        :param scalar_bound: The scalar bounds of the specified variables.

        :return: The scalar bounds.
        """
        bound = ScalarBound(
            l=cal_scalar_bound(
                constr_bound.L.A, constr_bound.L.b, scalar_bound.l, scalar_bound.u
            )
        )

        if constr_bound.U is None:
            return bound

        bound.u = cal_scalar_bound(
            constr_bound.U.A, constr_bound.U.b, scalar_bound.u, scalar_bound.l
        )

        return bound

    @staticmethod
    def store_bounds(
        all_bounds: dict[str, ScalarBound], name: str, bound: ScalarBound
    ) -> ScalarBound:
        """
        Store or update the new scalar bound of a layer into the cached data.

        :param all_bounds: A dictionary to store the scalar bounds of all layers.
        :param name: The name of the layer.
        :param bound: The new bounds of the layer.
        """
        # The optimized linear relaxation needs gradients.
        bound = bound.detach().clone()

        if name in all_bounds:
            old_bound = all_bounds[name]
            bound = old_bound.intersect(bound)

        all_bounds[name] = bound

        return bound

    @staticmethod
    def store_relaxations(
        all_relaxations: dict[str, LinearConstrBound], name: str, relaxation: LinearConstrBound
    ):
        """
        Store or update the new linear relaxation of the neurons in the layer into the
        cached data.

        :param all_relaxations: A dictionary to store the linear relaxations of all
            layers.
        :param name: The name of the layer.
        :param relaxation: The new linear relaxation of the layer.
        """
        all_relaxations[name] = relaxation

    @property
    def all_bounds(self) -> dict[str, ScalarBound]:
        """
        The scalar bounds of all layers. This a property for the shared data.

        Refer to :class:`BPSharedData` for more information.

        :return: The scalar bounds of all layers.
        """
        return self.shared_data.all_bounds

    @property
    def all_relaxations(self) -> dict[str, LinearConstrBound]:
        """
        The linear relaxations of the neurons in all layers. This a property for the
        shared data.

        Refer to :class:`BPSharedData` for more information.

        :return: The linear relaxations of the neurons in all layers.
        """
        return self.shared_data.all_relaxations


class InputIneqNode(BasicIneqNode, InputNode):
    """
    The input node for bound propagation with linear relaxation.

    :param name: The name of the layer.
    :param input_names: The names of the input layers.
    :param input_size: The size of the input.
    :param shared_data: The shared data among all nodes.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
    ):
        BasicIneqNode.__init__(self, name, input_names, input_size, shared_data)
        InputNode.__init__(self, name, input_names, input_size, shared_data)

    def forward(
        self,
        input_bound: ScalarBound,
        only_lower_bound: bool = False,
    ) -> tuple[ScalarBound | None, Tensor | None]:
        """
        .. attention::
            This method is a placeholder. The input node does not need a forward pass.
        """
        raise RuntimeError(
            f"{self.__class__.__name__} does not support this method. "
            f"Because the input do not need a forward pass."
        )

    def back_sub_to_input(self, constr_bound: LinearConstrBound) -> LinearConstrBound:
        """
        .. attention::
            This method is a placeholder. The input node does not need back
            substitution.
        """
        raise RuntimeError(
            f"{self.__class__.__name__} does not support this method. "
            f"Because the input do not need back substitution."
        )

    def back_sub_once(self, constr_bound: LinearConstrBound) -> LinearConstrBound:
        """
        Back-substitute the linear relaxations to the previous layers.

        .. tip::
            This method is just to reshape the linear relaxation to the input size.

        :param constr_bound: The linear relaxation.
        :return: The linear constraints only involving the variables of the
            preceding layer.
        """
        d = constr_bound.L.A.size(0)
        constr_bound.L.A = constr_bound.L.A.reshape(d, *self.input_size)
        if constr_bound.U is not None:
            constr_bound.U.A = constr_bound.U.A.reshape(d, *self.input_size)
        return constr_bound

    def clear(self):
        pass


class LinearIneqNode(BasicIneqNode, LinearNode, ABC):
    """
    The linear node for bound propagation with linear relaxation.

    :param name: The name of the layer.
    :param input_names: The names of the input layers.
    :param input_size: The size of the input.
    :param shared_data: The shared data among all nodes.
    :param weight: The weight matrix of the layer.
    :param bias: The bias of the layer.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
        weight: Tensor,
        bias: Tensor | None = None,
    ):
        args = (name, input_names, input_size, shared_data)
        BasicIneqNode.__init__(self, *args)
        LinearNode.__init__(self, *args, weight, bias)  # noqa

    def forward(
        self,
        input_bound: ScalarBound,
        only_lower_bound: bool = False,
    ) -> tuple[ScalarBound | None, Tensor | None]:
        """
        Calculate the scalar bounds of the neurons in the layer by the linear relaxation of the layer.
        The scalar bounds are calculated by propagating the linear relaxation from the layer to the input layer.

        :param input_bound: The scalar bound of the input.
        :param only_lower_bound: Return only the lower bound.

        :return: The scalar bounds of the neurons in the layer.
        """
        init_bound = self.init_constr_bound(only_lower_bound)
        # print(f"{DARK_GRAY_BK}    Initial constraint bound: {init_bound}{RESET}")
        approx_wrt_input = self.back_sub_to_input(init_bound)
        # print(f"{DARK_GRAY_BK}    Approximated w.r.t input: {approx_wrt_input}{RESET}")
        bound = self.cal_bounds(
            approx_wrt_input,
            input_bound,
        )
        # print(f"{DARK_GRAY_BK}    Calculated bound: {bound}{RESET}")

        bound = self.store_bounds(self.all_bounds, self.name, bound)
        return bound

    def clear(self):
        self._cached_init_constr = None


class GemmIneqNode(GemmNode, LinearIneqNode):
    """
    The back propagation node for the linear layer with the general matrix
    multiplication (GEMM) operation.

    :param name: The name of the layer.
    :param input_names: The names of the input layers.
    :param input_size: The size of the input.
    :param shared_data: The shared data among all nodes.
    :param weight: The weight matrix of the layer.
    :param bias: The bias of the layer.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int],
        shared_data: BPSharedData,
        weight: Tensor,
        bias: Tensor | None = None,
    ):
        args = (name, input_names, input_size, shared_data)
        LinearIneqNode.__init__(self, *args, weight, bias)  # noqa
        GemmNode.__init__(self, *args, weight, bias)  # noqa

    def back_sub_once(self, constr_bound: LinearConstrBound) -> LinearConstrBound:
        """
        Back-substitute the linear relaxations to the previous layers.

        .. tip::
            The back-substitution of the GEMM operation is a matrix multiplication
            and an addition operation.

        :param constr_bound: The linear relaxation.
        :return: The linear constraints only involving the variables of the
            preceding layer.
        """
        args = (self.weight, self.bias)

        # print("On the LOWER bound back-substitution:")
        LA = constr_bound.L.A.reshape(-1, *self.output_size)
        result_constr_bound = LinearConstrBound(
            L=LinearConstr(
                *gemm_back_sub(LA, constr_bound.L.b, *args)
            )
        )
        if constr_bound.U is not None:
            # print("On the UPPER bound back-substitution:")
            UA = constr_bound.U.A.reshape(-1, *self.output_size)
            result_constr_bound.U = LinearConstr(
                *gemm_back_sub(UA, constr_bound.U.b, *args)
            )

        return result_constr_bound

    def clear(self):
        LinearIneqNode.clear(self)
        GemmNode.clear(self)


class Conv2DIneqNode(LinearIneqNode, Conv2DNode):
    """
    The back propagation node for the convolutional layer.

    :param name: The name of the layer.
    :param input_names: The names of the input layers.
    :param input_size: The size of the input.
    :param shared_data: The shared data among all nodes.
    :param weight: The weight matrix of the layer.
    :param bias: The bias of the layer.
    :param stride: The stride of the convolution.
    :param padding: The padding of the convolution.
    :param dilation: The dilation of the convolution.
    :param groups: The groups of the convolution.
    :param ceil_mode: The ceil mode of the convolution.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int, int, int],
        shared_data: BPSharedData,
        weight: Tensor,
        bias: Tensor | None = None,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        ceil_mode: bool = False,
    ):
        args = (name, input_names, input_size, shared_data, weight, bias)
        kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "ceil_mode": ceil_mode,
        }
        LinearIneqNode.__init__(self, *args)
        Conv2DNode.__init__(self, *args, **kwargs)

    def back_sub_once(
        self,
        constr_bound: LinearConstrBound,
    ) -> LinearConstrBound:
        """
        Back-substitute the linear relaxations to the previous layers.

        .. tip::
            The back-substitution of the convolutional operation is a transposed
            convolution operation.

        :param constr_bound: The linear relaxation.
        :return: The linear constraints only involving the variables of the
            preceding layer.
        """
        args = (
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

        result_constr_bound = LinearConstrBound(
            L=LinearConstr(
                *conv2d_back_sub(
                    constr_bound.L.A.reshape(-1, *self.output_size),
                    constr_bound.L.b,
                    *args,
                )
            )
        )
        if constr_bound.U is None:
            return result_constr_bound

        result_constr_bound.U = LinearConstr(
            *conv2d_back_sub(
                constr_bound.U.A.reshape(-1, *self.output_size), constr_bound.U.b, *args
            )
        )
        return result_constr_bound

    def clear(self):
        LinearIneqNode.clear(self)
        Conv2DNode.clear(self)


class NonLinearIneqNode(BasicIneqNode, NonLinearNode, ABC):
    """
    The non-linear node for bound propagation.

    :param name: The name of the layer.
    :param input_names: The names of the input layers.
    :param input_size: The size of the input.
    :param shared_data: The shared data among all nodes.
    :param act_relax_args: The arguments for activation relaxation.
    """
    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
        act_relax_args: ActRelaxArgs,
    ):
        args = (name, input_names, input_size, shared_data)
        BasicIneqNode.__init__(self, *args)
        NonLinearNode.__init__(self, *args, act_relax_args)  # noqa

    def forward(
        self,
        input_bound: ScalarBound,
        only_lower_bound: bool = False,
    ) -> tuple[ScalarBound | None, Tensor | None]:
        """
        The non-linear layer needs the scalar bounds of the preceding layer to calculate
        the linear relaxation first.

        If necessary, calculate the scalar bounds of the neurons in the layer by the
        linear relaxation of the layer. The scalar bounds are calculated by
        propagating the linear relaxation from the layer to the input layer.

        .. tip::
            The preceding layer of MaxPool layer is commonly a non-linear layer, where
            the scalar bounds of the non-linear layer are necessary to calculate the
            scalar bounds of the MaxPool layer.
            So the scalar bounds of the non-linear layer are calculated by
            back-substitution.

        :return: The scalar bounds of the neurons in the layer.
        """
        relaxation = self.cal_relaxation(input_bound, self.shared_data)
        self.store_relaxations(self.all_relaxations, self.name, relaxation)

        if (
            isinstance(self.next_nodes[0], NonLinearIneqNode)
            or self.act_relax_args.update_scalar_bounds_per_layer
        ):
            # If the next layer is MaxPool layer, we need to calculate the scalar bounds
            # of the neurons in the current layer.
            bound = self.cal_bounds(
                self.back_sub_to_input(self.init_constr_bound(only_lower_bound)),
                input_bound,
            )

            # ----- Tighten the bounds with activation function -----
            bound = self.update_bounds_by_act_func(bound)
            bound = self.store_bounds(self.all_bounds, self.name, bound)

            return bound
        return None

    def update_bounds_by_act_func(self, old_bound: ScalarBound) -> ScalarBound:
        """
        Update the scalar bounds of the neurons in the layer by the activation function.

        :param old_bound: The scalar bounds of the neurons in the layer.
        :return: The updated scalar bounds of the neurons in the layer.
        """
        pre_bound = self.all_bounds[self.input_names[0]]

        new_l = self.f(pre_bound.l)
        new_u = self.f(pre_bound.u)
        old_bound.l = torch.max(old_bound.l, new_l)
        old_bound.u = torch.min(old_bound.u, new_u)

        return old_bound

    def back_sub_once(self, constr_bound: LinearConstrBound) -> LinearConstrBound:
        relaxation = self.all_relaxations[self.name]

        result_constr_bound = LinearConstrBound(
            L=LinearConstr(
                *nonlinear_back_sub(
                    constr_bound.L.A.reshape(-1, math.prod(self.output_size)),
                    constr_bound.L.b,
                    relaxation.L.A,
                    relaxation.U.A,
                    relaxation.L.b,
                    relaxation.U.b,
                )
            )
        )

        if constr_bound.U is None:
            return result_constr_bound

        result_constr_bound.U = LinearConstr(
            *nonlinear_back_sub(
                constr_bound.U.A.reshape(-1, math.prod(self.output_size)),
                constr_bound.U.b,
                relaxation.U.A,
                relaxation.L.A,
                relaxation.U.b,
                relaxation.L.b,
            )
        )

        return result_constr_bound

    def cal_relaxation(
        self,
        input_bound: ScalarBound,
        shared_data: BPSharedData,
    ) -> LinearConstrBound:
        """
        Calculate the relaxation based on single neuron constraints. The relaxation
        consists of two inequalities. One is for the lower bound and the other is for
        the upper bound.

        :param input_bound: The scalar bounds of the input.
        :param shared_data: The cached data of bound propagation.

        .. attention::
            The sparse format has not been implemented.

        :return: The linear relaxation of the lower and upper bounds of the
            current non-linear layer.
        """

        print(f"[DEBUG] Calculate single-neuron relaxation.")
        pre_bound = shared_data.all_bounds[self.input_names[0]]
        # print(f"{RED_BK}[DEBUG] [before relaxation] =>=> pre_bound: {pre_bound}{RESET}")

        k_l, k_u, b_l, b_u = self._cal_relaxation(
            pre_bound.l.flatten(),
            pre_bound.u.flatten(),
            self.act_relax_args.mode,
        )

        # print(f"{RED_BK}[DEBUG] [after relaxation] =>=> k_l: \n{k_l}, \nk_u: \n{k_u}, \nb_l: \n{b_l}, \nb_u: \n{b_u}{RESET}")
        return LinearConstrBound(L=LinearConstr(A=k_l, b=b_l), U=LinearConstr(A=k_u, b=b_u))  # noqa

    @staticmethod
    @abstractmethod
    def _cal_relaxation(
        l: Tensor,
        u: Tensor,
        mode: RelaxMode,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pass


class ReLUIneqNode(NonLinearIneqNode, ReLUNode):
    """
    The back propagation node for the ReLU layer with linear relaxation.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
        act_relax_args: ActRelaxArgs,
    ):
        args = (name, input_names, input_size, shared_data, act_relax_args)
        NonLinearIneqNode.__init__(self, *args)
        ReLUNode.__init__(self, *args)

    @staticmethod
    def _cal_relaxation(
        l: Tensor,
        u: Tensor,
        mode: RelaxMode,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return cal_relaxation_relu(l, u, mode)

    def back_sub_once(self, constr_bound: LinearConstrBound) -> LinearConstrBound:
        print("** ReLUIneqNode back_sub_once")
        relaxation = self.all_relaxations[self.name]
        result_constr_bound = LinearConstrBound(
            L=LinearConstr(
                *relu_back_sub(
                    constr_bound.L.A.reshape(-1, math.prod(self.output_size)),
                    constr_bound.L.b,
                    relaxation.L.A,
                    relaxation.U.A,
                    relaxation.U.b,
                )
            )
        )

        if constr_bound.U is None:
            return result_constr_bound
        # NOTE: Here, it is different from other functions. Pay attension to the order of the arguments.
        result_constr_bound.U = LinearConstr(
            *relu_back_sub(
                constr_bound.U.A.reshape(-1, math.prod(self.output_size)),
                constr_bound.U.b,
                relaxation.L.A,
                relaxation.U.A,
                relaxation.U.b,
                is_lower=False,
            )
        )

        return result_constr_bound


class SigmoidIneqNode(NonLinearIneqNode, SigmoidNode):
    """
    The back propagation node for the sigmoid layer with linear relaxation.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
        act_relax_args: ActRelaxArgs,
    ):
        args = (name, input_names, input_size, shared_data, act_relax_args)
        NonLinearIneqNode.__init__(self, *args)
        SigmoidNode.__init__(self, *args)

    @staticmethod
    def _cal_relaxation(
        l: Tensor,
        u: Tensor,
        mode: RelaxMode,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return cal_relaxation_sigmoid(l, u, mode)


class TanhIneqNode(NonLinearIneqNode, TanhNode):
    """
    The back propagation node for the tanh layer with linear relaxation.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
        act_relax_args: ActRelaxArgs,
    ):
        args = (name, input_names, input_size, shared_data, act_relax_args)
        NonLinearIneqNode.__init__(self, *args)
        TanhNode.__init__(self, *args)

    @staticmethod
    def _cal_relaxation(
        l: Tensor,
        u: Tensor,
        mode: RelaxMode,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return cal_relaxation_tanh(l, u, mode)


class MaxPool2DIneqNode(NonLinearIneqNode, MaxPool2DNode):
    """
    The back propagation node for the max pooling layer with linear relaxation.

    :param name: The name of the layer.
    :param input_names: The names of the input layers.
    :param input_size: The size of the input.
    :param shared_data: The shared data among all nodes.
    :param kernel_size: The size of the kernel.
    :param act_relax_args: The arguments for activation relaxation.
    :param stride: The stride of the convolution.
    :param padding: The padding of the convolution.
    :param dilation: The dilation of the convolution.
    :param ceil_mode: The ceil mode of the convolution.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int, int, int],
        shared_data: BPSharedData,
        act_relax_args: ActRelaxArgs,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        ceil_mode: bool = False,
    ):
        args = (name, input_names, input_size, shared_data, act_relax_args)
        kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "ceil_mode": ceil_mode,
        }
        NonLinearIneqNode.__init__(self, *args)
        MaxPool2DNode.__init__(self, *args, **kwargs)

    def cal_relaxation(
        self, input_bound: ScalarBound, shared_data: BPSharedData
    ) -> LinearConstrBound:

        print(f"[DEBUG] Calculate single-neuron relaxation.")

        pre_bound = self.shared_data.all_bounds[self.input_names[0]]

        l, u = self.get_unfolded_pre_bound(pre_bound)
        l_max, l_argmax = self._get_l_argmax(pre_bound)
        mask = self.get_nontrivial_neuron_mask(pre_bound)
        k_l, k_u, b_l, b_u = self._cal_relaxation(
            l, u, self.act_relax_args.mode, l_max, l_argmax, mask
        )

        return LinearConstrBound(L=LinearConstr(A=k_l, b=b_l), U=LinearConstr(A=k_u, b=b_u))  # noqa

    @staticmethod
    def _cal_relaxation(
        l: Tensor,
        u: Tensor,
        mode: RelaxMode,
        l_max: Tensor = None,
        l_argmax: Tensor = None,
        mask: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return cal_relaxation_maxpool2d(l, u, mode, l_max, l_argmax, mask)

    def back_sub_once(
        self,
        constr_bound: LinearConstrBound,
    ) -> LinearConstrBound:
        relaxation = self.all_relaxations[self.name]
        args = (
            self.input_size,
            self.output_size,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )

        result_constr_bound = LinearConstrBound(
            L=LinearConstr(
                *maxpool2d_back_sub(
                    constr_bound.L.A.reshape((-1, self._co * self._nk)),
                    constr_bound.L.b,
                    relaxation.L.A,
                    relaxation.U.A,
                    relaxation.L.b,
                    relaxation.U.b,
                    *args,
                )
            )
        )

        if constr_bound.U is None:
            return result_constr_bound

        result_constr_bound.U = LinearConstr(
            *maxpool2d_back_sub(
                constr_bound.U.A.reshape((-1, self._co * self._nk)),
                constr_bound.U.b,
                relaxation.U.A,
                relaxation.L.A,
                relaxation.U.b,
                relaxation.L.b,
                *args,
            )
        )

        return result_constr_bound

    def update_bounds_by_act_func(self, old_bound: ScalarBound) -> ScalarBound:
        """
        Update the scalar bounds of the neurons in the layer by the activation function.

        :param old_bound: The scalar bounds of the neurons in the layer.
        :return: The updated scalar bounds of the neurons in the layer.
        """
        pre_bound = self.all_bounds[self.input_names[0]]
        pre_l = pre_bound.l.reshape(self.input_size)
        pre_u = pre_bound.u.reshape(self.input_size)
        l = self.f(pre_l).flatten()
        u = self.f(pre_u).flatten()
        old_bound.l = torch.max(old_bound.l, l)
        old_bound.u = torch.min(old_bound.u, u)

        return old_bound

    def clear(self):
        NonLinearIneqNode.clear(self)
        MaxPool2DNode.clear(self)


class ResidualAddIneqNode(LinearIneqNode, ResidualAddNode):
    """
    The back propagation node for the residual addition layer.

    :param name: The name of the layer.
    :param input_names: The names of the input layers.
    :param input_size: The size of the input.
    :param shared_data: The shared data among all nodes.
    """

    def __init__(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
        shared_data: BPSharedData,
    ):
        args = (name, input_names, input_size, shared_data)
        weight = torch.empty(0)
        LinearIneqNode.__init__(self, *args, weight)  # noqa
        ResidualAddNode.__init__(self, *args)

    def back_sub_once(
        self,
        constr_bound: LinearConstrBound,
    ) -> LinearConstrBound:
        return constr_bound
