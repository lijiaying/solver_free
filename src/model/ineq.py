"""
This module provides the class of implementing the bound propagation with
symbolic inequalities.
"""

__docformat__ = "restructuredtext"
__all__ = ["IneqBoundModel"]

import logging
import time
from typing import TypeVar, Iterator

import torch
from torch import Tensor

from src.boundprop import *
from src.model.base import BasicBoundModel
from src.utils import *

T = TypeVar("T", bound=BasicIneqNode)


class IneqBoundModel(BasicBoundModel[T]):
    """
    This is an implementation of bound propagation with symbolic inequalities, where one
    lower bound and one upper bound are calculated for each neuron in the network.

    :param net_fpath: The path of the network file.
    :param perturb_args: The perturbation arguments.
    :param act_relax_args: The activation relaxation arguments.
    :param ada_act_relax_args: The adaptive constraint arguments.
    :param dtype: The data type used in torch.
    :param device: The device used in torch.
    """

    def __init__(
        self,
        net_fpath: str,
        perturb_args: PerturbArgs,
        act_relax_args: ActRelaxArgs,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super(IneqBoundModel, self).__init__(net_fpath, perturb_args, dtype, device, *args, **kwargs)
        self.act_relax_args: ActRelaxArgs = act_relax_args
        # The following will be shared with the submodules when building them.
        self.bp_shared_data = BPSharedData(dtype, device)


    def forward(
        self,
        target_label: int,
        input_sample: Tensor,
        input_bound: ScalarBound,
        *args,
        **kwargs,
    ) -> tuple[ScalarBound, Tensor | None]:
        """
        This is the main method to calculate the bound of the output of the network.

        - The hidden layer is splitted as 1 linear op and 1 non-linear op (i.e. activation).
        - In this method, the bounds of neurons in each layer are calculated until
          the last layer. The scalar bound of the output is returned.
        - For each layer, back-substitution is used to calculate the bounds of the neurons in the layer.

        Here it utilized the mechanism of pytorch module for the forward method.

        :param target_label: The target label for the output.
        :param input_sample: The input sample point.
        :param input_bound: The scalar bound (lower and upper bounds) of the input.

        :return: The scalar bound of the output and the minimum input point if enabled.
        """

        time_start = time.perf_counter()
        print(f">>>>>>>>>>>>>>> Start bound propagation.")
        bound = self.bound_propagate(input_bound)
        print(f"<<<<<<<<<<<<<<< Finish bound propagation in {time.perf_counter() - time_start:.4f}s")
        return bound


    def bound_propagate(
        self,
        input_bound: ScalarBound,
    ) -> tuple[ScalarBound, Tensor | None]:
        """
        This is the specific method to implement bound propagation in the :func:`forward`.
        :param input_bound: The scalar bound of the input.
        :return: The scalar bound of the output.
        """

        bound = None
        self.all_bounds[self.input_name] = input_bound

        modules_iter: Iterator[BasicIneqNode] = iter(self.submodules.values())
        module = next(modules_iter)  # Input layer
        print(f"{BLUE}>>> Process {module}{RESET}")

        print(f"Lower bound: {input_bound.l.flatten()[:5]}")
        print(f"Upper bound: {input_bound.u.flatten()[:5]}")

        module = next(modules_iter)
        while module is not None:
            start = time.perf_counter()
            print(f"{BLUE}>>> Process {module}{RESET}")

            module.clear()
            bound = module.forward(input_bound, only_lower_bound=module.next_nodes is None)

            print(f"    INPUT  bound: {input_bound}")
            print(f"    OUTPUT bound: lower: {bound.l.flatten()[:5]}, upper: {bound.u.flatten()[:5] if bound.u is not None else None}")
            print(f"{GREEN}<<< Finish processing {module} in {time.perf_counter() - start:.4f}s{RESET}")
            module = next(modules_iter, None)
        return bound


    def _handle_input(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
    ) -> InputNode:
        args = (name, input_names, input_size, self.bp_shared_data)
        return InputIneqNode(*args)


    def _handle_gemm(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int],
        weight: Tensor,
        bias: Tensor | None = None,
    ) -> GemmNode:
        args = (name, input_names, input_size, self.bp_shared_data, weight, bias)
        return GemmIneqNode(*args)


    def _handle_conv2d(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int, int, int],
        weight: Tensor,
        bias: Tensor = None,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        ceil_mode: bool = False,
    ) -> Conv2DNode:
        args = (
            name,
            input_names,
            input_size,
            self.bp_shared_data,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            ceil_mode,
        )
        return Conv2DIneqNode(*args)


    def _handle_relu(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
    ) -> ReLUNode:
        args = (name, input_names, input_size, self.bp_shared_data, self.act_relax_args)
        return ReLUIneqNode(*args)


    def _handle_sigmoid(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
    ) -> SigmoidNode:
        args = (name, input_names, input_size, self.bp_shared_data, self.act_relax_args)
        return SigmoidIneqNode(*args)


    def _handle_tanh(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
    ) -> TanhNode:
        args = (name, input_names, input_size, self.bp_shared_data, self.act_relax_args)
        return TanhIneqNode(*args)


    def _handle_maxpool2d(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int, int, int],
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        ceil_mode: bool = False,
    ) -> MaxPool2DNode:
        args = (
            name,
            input_names,
            input_size,
            self.bp_shared_data,
            self.act_relax_args,
            kernel_size,
        )

        kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "ceil_mode": ceil_mode,
        }
        return MaxPool2DIneqNode(*args, **kwargs)


    def _handle_residual_add(
        self,
        name: str,
        input_names: list[str],
        input_size: tuple[int] | tuple[int, int, int],
    ) -> ResidualAddNode:
        args = (name, input_names, input_size, self.bp_shared_data)
        return ResidualAddIneqNode(*args)


    def clear(self):
        """
        Clear the cached data in the current sample to verify the next sample.
        """
        self.bp_shared_data.clear()


    @property
    def all_bounds(self) -> dict[str, ScalarBound]:
        """
        All scalar bounds of the neurons in the network.

        Refer to :class:`BPSharedData` for more details.
        """
        return self.bp_shared_data.all_bounds


    @property
    def all_relaxations(self) -> dict[str, LinearConstrBound]:
        """
        All linear constraints of the neurons in the network.

        Refer to :class:`BPSharedData` for more details.
        """
        return self.bp_shared_data.all_relaxations
