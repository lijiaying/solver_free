"""
This script is used to run the verification experiment for a neural network model.
"""
import argparse
import sys

sys.path.append("../")

from datetime import datetime

from src import *


def main(args):
    if args.bp == "deeppoly":
        bp_method = RelaxMode.DEEPPOLY
    elif args.bp == "crown":
        bp_method = RelaxMode.CROWN
    elif args.bp == "stm":
        bp_method = RelaxMode.STM_SN
    else:
        raise ValueError(f"Invalid BP method: {args.bp}")

    if args.opt == "lp":
        opt_method = OptimizationMethod.LP
    elif args.opt == "mnlp":
        opt_method = OptimizationMethod.MNLP
    else:
        opt_method = None

    log_file = None
    if args.log_name is not None:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        log_file = args.log_name
        log_file = "./logs/" + log_file + f"_{current_time}.log"

    if args.dataset == 'deeppoly':
        input_limited_range = (-1.0, 1.0)
    else:
        input_limited_range = (0.0, 1.0)
    # Set arguments for the model.
    args = Arguments(
        net_fpath=args.net_fpath,
        dataset=args.dataset,
        epsilon=args.epsilon,
        act_relax_mode=bp_method,
        log_file=log_file,
        opt_method=opt_method,
        first_sample_index=0,
        num_samples=args.num_samples,
        input_limited_range=input_limited_range,
        # device="cuda:0",  # noqa
        dtype="float64",
    )

    # Create the model and verify it.
    model = ModelFactory(args)
    model.prepare()
    model.build()
    model.verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiment.")
    parser.add_argument("--net_fpath", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--bp", type=str, default=None)
    parser.add_argument("--opt", type=str, default=None)
    parser.add_argument("--log_name", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()
    if args.dataset is None:
        net_lower = args.net_fpath.lower()
        if 'deeppoly' in net_lower:
            args.dataset = 'deeppoly'
        elif 'mnist' in net_lower:
            args.dataset = "mnist"
        elif 'cifar10' in net_lower:
            args.dataset = "cifar10"

    if args.num_samples is not None:
        assert args.num_samples > 0 and args.num_samples <= 100, "num_samples should be positive."
        print(f"Set num_samples to {args.num_samples}")
    else:
        args.num_samples = 1
        
    main(args)


# python3 exp.py --net_fpath ../nets/deeppoly.onnx --epsilon 1 --bp stm --opt mnlp