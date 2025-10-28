"""
This module contains the arguments for the experiment.
"""

__docformat__ = "restructuredtext"
__all__ = ["Arguments"]

import logging
import os
from dataclasses import dataclass

import torch

from src.utils import *

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Arguments:
    net_fpath: str
    dataset: str
    epsilon: float
    bound_propagation_method: BoundPropagate = BoundPropagate.INEQUALITY
    act_relax_mode: RelaxMode = RelaxMode.STM_SN
    opt_method: OptimizationMethod | None = None
    num_labels: int | None = None

    check_ignored_samples: bool = True
    num_samples: int = 1
    first_sample_index: int = 0
    """The index of the first sample to verify."""

    normalize: bool = True
    """Whether to normalize the input data."""

    means: torch.Tensor = None
    """The means of normalization."""

    stds: torch.Tensor = None
    """The standard deviations of normalization."""

    input_limited_range: tuple = (0.0, 1.0)
    """
    The limited range of the input data.
     
    For example, an image with a pixel value range of [0, 1] should be limited to [0, 1]
    after perturbed because the perturbation should not exceed the pixel value range.
    """

    net_dir_path: str | None = None
    """The directory path of the network file."""

    net_fname: str | None = None
    """The name of the network file, excluding the file extension and directory path."""

    log_file: str | None = None
    """The name of the log file."""

    log_level: int = logging.INFO
    """The logging level."""

    random_seed: int = 0
    """The random seed for all random number generators (python, numpy, torch)."""

    dtype: str = "float64"
    """The data type that numpy and torch use."""

    device: str = "cpu"
    """The device that torch uses."""

    perturb_args: PerturbArgs | None = None
    """The perturbation arguments."""

    act_relax_args: ActRelaxArgs | None = None
    """The activation relaxation arguments."""

    lp_args: LPArgs | None = None
    """The LP arguments."""

    kact_lp_args: KActLPArgs | None = None
    """The kact LP arguments."""

    def __post_init__(self):

        if not os.path.exists(self.net_fpath):
            raise ValueError(f"Network file {self.net_fpath} does not exist.")

        if not self.net_fpath.endswith(".onnx"):
            raise ValueError(f"Network file {self.net_fpath} is not an ONNX file.")

        # if self.dataset not in {"mnist", "cifar10", "custom"}:
        #     raise ValueError(f"Dataset {self.dataset} is not supported.")

        if self.epsilon < 0:
            raise ValueError(
                f"Perturbation radius {self.epsilon} should be " f"non-negative."
            )

        if self.num_labels is not None:
            if self.num_labels < 2:
                raise ValueError(
                    f"Number of labels {self.num_labels} should be at least 2."
                )

        if self.num_samples < 1:
            raise ValueError(
                f"Number of samples {self.num_samples} should be at least 1."
            )

        if self.first_sample_index < 0:
            raise ValueError(
                f"First sample index {self.first_sample_index} should be non-negative."
            )

        if self.input_limited_range[0] > self.input_limited_range[1]:
            raise ValueError(
                f"Input limited range {self.input_limited_range} is invalid."
            )

        # if self.dataset in {"mnist", "cifar10"}:
        #     self.num_labels = 10
        net_lower = self.net_fpath.lower()
        if self.dataset == "deeppoly":
            self.num_labels = 2
        elif self.dataset == "mnist":
            self.num_labels = 10
        elif self.dataset == "cifar10":
            self.num_labels = 10
        else:
            assert (
                False
            ), f"net_fpath {self.net_fpath} not recognized for dataset setting."

        if self.net_dir_path is None:
            self.net_dir_path = os.path.dirname(self.net_fpath)

        if self.net_fname is None:
            self.net_fname = os.path.basename(self.net_fpath).split(".")[0]

        # If the log file is not None, check the dir path.
        if self.log_file is None:
            self.log_file = os.path.join(
                repo_dir,
                "logs",
                f"{self.net_fname}_{self.epsilon}_{self.bound_propagation_method}_{self.act_relax_mode}.log",
            )
        print("log file:", self.log_file)

        log_dir_path = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)

        self._set_means_stds()
        self._set_args()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _set_args(self):
        # -------- Set perturbation arguments --------

        dtype = torch.float64 if self.dtype == "float64" else torch.float32
        device = torch.device(self.device)
        print("use device:", device)

        self.perturb_args = PerturbArgs(
            epsilon=self.epsilon,
            norm=float("inf"),
            means=self.means.to(dtype=dtype, device=device),
            stds=self.stds.to(dtype=dtype, device=device),
            lower_limit=self.input_limited_range[0],
            upper_limit=self.input_limited_range[1],
        )

        print(f"[DEBUG] Set perturbation arguments: {self.perturb_args}.")
        # -------- Set activation relaxation arguments --------
        self.act_relax_args = ActRelaxArgs(mode=self.act_relax_mode)
        self.act_relax_args.update_scalar_bounds_per_layer = (
            self.act_relax_mode != CROWN
        )
        print(f"[DEBUG] Set activation relaxation arguments: {self.act_relax_args}.")

        # -------- Set LP arguments --------
        if self.opt_method is not None:
            self.lp_args = LPArgs()
            print(f"[DEBUG] Set LP arguments: {self.lp_args}")

            if self.opt_method == OptimizationMethod.MNLP:
                self.kact_lp_args = KActLPArgs()
                print(f"[DEBUG] Set kact LP arguments: {self.kact_lp_args}")

    def __str__(self) -> str:
        return (
            f"Arguments(\n"
            + ",\n".join(f"\t{k:<25}: {v}" for k, v in self.__dict__.items())
            + "\n"
            f")"
        )

    def _set_means_stds(self):
        """
        Set the means and standard deviations for normalization.
        """
        if self.means is None or self.stds is None:
            if self.dataset == "mnist":
                self.means = torch.tensor([0.0])
                self.stds = torch.tensor([1.0])
            elif self.dataset == "cifar10":
                self.means = torch.tensor([[[0.4914]], [[0.4822]], [[0.4465]]])
                self.stds = torch.tensor([[[0.2023]], [[0.1994]], [[0.201]]])
            elif self.dataset == "deeppoly":
                self.means = torch.tensor([0.0])
                self.stds = torch.tensor([1.0])
                self.normalize = False
                return
            else:
                raise ValueError(f"Dataset {self.dataset} is not supported.")

        self.normalize = True
