"""
This script is used to test the verification of a neural network model by one small
sample.
"""
import sys

sys.path.append("../")

import logging
from datetime import datetime

from src import *


def main():
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    # Set arguments for the model.
    args = Arguments(
        net_fpath="../nets/mnist_leaky_relu_3_500.onnx",
        dataset="mnist",
        epsilon=0.03,
        log_level=logging.INFO,
        act_relax_mode=RelaxMode.CROWN,
        opt_method=OptimizationMethod.MNLP,
        first_sample_index=0,
        num_samples=1,
        device="cuda:0",
        dtype="float64",
    )

    # Create the model and verify it.
    model = ModelFactory(args)
    model.prepare()
    model.build()
    model.verify()


if __name__ == "__main__":
    main()
