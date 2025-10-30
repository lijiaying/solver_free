"""
Factory class for creating models.
This class is responsible for set up everything needed for the verification and choose
the specified approach to verify the given model.
"""

__docformat__ = "restructuredtext"
__all__ = ["ModelFactory"]

import os
import random
import time

import numpy as np
import onnxruntime as ort
import torch

from .arguments import Arguments
from .model import *
from .utils import load_dataset, VerificationStatus


class ModelFactory:
    """
    Factory class for creating models.

    :param arguments: The arguments of the experiment.
    """

    def __init__(self, arguments: Arguments):
        self.arguments = arguments
        self.model = None
        self.data_loader = None

        self._ignored_samples = None
        print(f"[INFO] Arguments:\n{arguments}")

        _init_calculation_settings(
            arguments.random_seed, arguments.device, arguments.dtype
        )

        self.dtype = torch.float64 if arguments.dtype == "float64" else torch.float32
        self.device = torch.device(arguments.device)

    def prepare(self):
        """
        Prepare the dataset. Here we do two things:

        1. Load the dataset and recorded the ignored samples, which are samples that
           are not classified correctly by the given model and will be skipped in the
           verification process. In fact, we run the samples through the original model
           and record the samples that are classified correctly.
        2. Load the dataset for verification. The dataset sample will not be normalized
           because the normalization is done when verification.

        .. note::

           - By default, we load the test set of the given dataset. We will download the
             dataset from the internet and save it in the directory `../.temp/datasets`.
           - The ignored samples are samples that are not classified correctly by the
             given model and will be recorded in the directory
             `../.temp/ignored_samples`.
        """

        args = self.arguments

        print(f"[DEBUG] Prepare dataset.")

        print(f"[DEBUG] Find the ignored samples for the model {args.net_fpath}.")
        self._ignored_samples = _load_ignored_samples(
            "../.temp/ignored_samples/" + args.net_fname + ".txt",
            args.net_fpath,
            load_dataset(
                args.dataset,
                dir_path="../.temp/datasets",
                normalize=args.normalize,
                means=args.means,
                stds=args.stds,
            ),
            args.num_samples,
            args.first_sample_index,
            args.check_ignored_samples,
        )
        print(f"[INFO] Ignored samples: {self._ignored_samples}")

        print(f"[DEBUG] Load dataset {args.dataset}.")
        self.data_loader = load_dataset(
            args.dataset, dir_path="../.temp/datasets", normalize=False
        )

    def build(self):
        """
        Build the model based on the arguments.
        """
        perturb_args = self.arguments.perturb_args
        act_relax_args = self.arguments.act_relax_args
        lp_args = self.arguments.lp_args
        kact_lp_args = self.arguments.kact_lp_args

        dtype = self.dtype
        device = self.device

        if lp_args is None:
            self.model = IneqBoundModel(
                self.arguments.net_fpath,
                perturb_args,
                act_relax_args,
                dtype=dtype,
                device=device,
            )
        else:
            if kact_lp_args is None:
                self.model = LPBoundModel(
                    self.arguments.net_fpath,
                    perturb_args,
                    act_relax_args,
                    lp_args,
                    dtype=dtype,
                    device=device,
                )
            else:
                self.model = KActLPBoundModel(
                    self.arguments.net_fpath,
                    perturb_args,
                    act_relax_args,
                    lp_args,
                    kact_lp_args,
                    dtype=dtype,
                    device=device,
                )

        self.model.build()
        self.model.eval()

    def verify(self):
        """
        Verify the model with the given dataset and arguments.
        """

        args = self.arguments
        dtype = self.dtype
        device = self.device

        samples_stats = {
            "verified": set(),
            "unknown": set(),
        }
        samples_v = []
        time_sample_list = []
        num_samples = args.num_samples

        for i, (sample, target_label) in enumerate(self.data_loader):
            if _skip_sample(i, self._ignored_samples, args.first_sample_index):
                print("skip sample ", i, flush=True)
                continue
            if num_samples <= 0:
                break

            print(f"*"*100)
            print(f"*"*100)
            print(f"*"*100)
            print(f"*"*100)
            print(f"Sample {i}".center(100, "="))
            time_sample = time.perf_counter()

            target_label = int(target_label.item())
            print(f"[INFO] Target label is {target_label}.")
            self.model.update_output_constrs(
                self.model.get_output_weight_bias(target_label, args.num_labels)[0]
            )

            sample = sample.squeeze_(0).to(dtype=dtype, device=device)

            input_bound = self.model.get_input_bound(sample)

            # ======================= Incomplete Verification =======================
            bound = self.model(target_label, sample, input_bound)
            l = bound.l
            print(f"[INFO] Verified lower bound: {l}")

            if torch.all(l >= 0):
                global_status = VerificationStatus.SAT
                samples_v.append(i)
                samples_stats["verified"].add(i)
            else:
                global_status = VerificationStatus.UNKNOWN
                samples_stats["unknown"].add(i)
            print(f"[INFO] Verification results: {global_status}")

            if global_status != VerificationStatus.SAT:
                if (
                    torch.min(l) > -10.0
                    and hasattr(self.model, "lp_args")
                    and self.model.lp_args is not None
                ):
                    print(f"[INFO] Start verification by LP.")
                    self.model.build_lp()

                    # Find the negative items in l and rearrange their indices
                    labels = torch.argsort(l)

                    adv_labels = labels[l[labels] < 0].tolist()
                    print(f"[INFO] Adversarial labels: {adv_labels}")

                    results = self.model.verify_lp(target_label, adv_labels)
                    print(f"[DEBUG] Verification results: {results}")

                    if all(results):
                        global_status = VerificationStatus.SAT
                        samples_v.append(i)
                        samples_stats["verified"].add(i)
                    else:
                        global_status = VerificationStatus.UNKNOWN
                        samples_stats["unknown"].add(i)
                    print(f"[INFO] Verification results: {global_status}.")

            self.model.clear()

            time_sample = time.perf_counter() - time_sample
            print(f"[INFO] Finish sample {i}. Cost time: {time_sample:.4f}s")
            time_sample_list.append(time_sample)
            print(f"[INFO] Current {len(samples_v)}/{len(time_sample_list)} verified.")
            print(f"[DEBUG] Verified: {samples_v}")
            num_samples -= 1

        print(f"Stats: {samples_stats}")
        time_total = sum(time_sample_list)
        print(f"Finish verification. Cost Time:{time_total:.4f}s")
        print(f"Total {len(samples_v)}/{len(time_sample_list)} verified samples.")


def _init_calculation_settings(random_seed: int, device: str, dtype: str):

    print(f"[DEBUG] Set print options.")
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=200, profile="full")
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    print(f"[DEBUG] Set random seed as {random_seed}.")
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    print(f"[DEBUG] Set dtype as {dtype}.")
    if dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise ValueError(f"Invalid dtype: {dtype}.")

    print(f"[DEBUG] Set device as {device}")
    if device != "cpu":
        torch.cuda.manual_seed(random_seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        print(f"[DEBUG] Set torch deterministic algorithms as True.")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = dtype == "float32"
        torch.backends.cudnn.allow_tf32 = dtype == "float32"


def _record_ignored_samples(fpath: str, ignored_samples: set):

    print(f"[DEBUG] Write ignored samples in {fpath}.")

    if not fpath.endswith(".txt"):
        raise ValueError(f"File path must end with .txt, not {fpath}.")

    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    with open(fpath, "w") as f:
        f.write(str(ignored_samples))


def _get_ignored_samples(
    net_fpath: str,
    data_loader: torch.utils.data.DataLoader,  # type: ignore
    num_samples: int,
    start_index: int,
) -> set:

    print(f"[DEBUG] Get ignored samples by original model.")

    sess = ort.InferenceSession(net_fpath)
    # input_name = sess.get_inputs()[0]._idx
    # output_name = sess.get_outputs()[0]._idx
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    ignored_samples = set()
    for i, (sample, label) in enumerate(data_loader):
        if i < start_index:
            continue
        if num_samples <= 0:
            break

        print(f"-> checking sample", i, flush=True)
        # print("    sample:", sample)
        # print("    label:", label)

        output = sess.run(
            [output_name],
            {input_name: sample.numpy().astype(np.float32)},  # noqa
        )[0]
        # Attention, do not change the original sample
        if np.argmax(output) != int(label.item()):
            ignored_samples.add(i)
            continue

        num_samples -= 1

    print(f"[DEBUG] Get ignored samples: {ignored_samples}.")

    return ignored_samples


def _load_ignored_samples(
    fpath: str,
    net_fpath: str,
    data_loader: torch.utils.data.DataLoader,  # type: ignore
    num_samples: int,
    start_index: int,
    check_ignored_samples: bool = True,
) -> set:

    if not fpath.endswith(".txt"):
        raise ValueError(f"File path must end with .txt, not {fpath}.")

    print(f"[DEBUG] Load ignored samples from {fpath}.")

    if check_ignored_samples:
        ignored_samples = _get_ignored_samples(
            net_fpath, data_loader, num_samples, start_index
        )
        _record_ignored_samples(fpath, ignored_samples)

    else:
        if not os.path.exists(fpath):
            print(
                f"Ignored samples file {fpath} does not exist. "
                f"There is no ignored samples."
            )
            return set()
        print(f"[DEBUG] Read existing ignored samples file {fpath}.")
        with open(fpath, "r") as f:
            ignored_samples = eval(f.readline())

        print(f"[DEBUG] Get ignored samples: {ignored_samples}.")

    return ignored_samples


def _skip_sample(i: int, ignored_samples: set, start_index: int):

    skip = (i in ignored_samples) or (i < start_index)
    if skip:
        print(f"[DEBUG] Sample {i} is skipped.")
    return skip
