"""This is for calculating the function hull with given constraints and bounds."""
from setproctitle import setproctitle
setproctitle("python check_sma_value_equivalence.py")
import os
import sys
import time
import warnings

cpu_affinity = os.sched_getaffinity(0)

from pathlib import Path
# Path object for the current file
path = Path(__file__).resolve()
repo_root = path.parent.parent

sys.path.insert(0, str(repo_root))
import numpy as np
from src.utils import *
from src.funchull.acthull import *

def read_constraint_file(constraint_fpath: str):
    assert os.path.exists(constraint_fpath), f"File {constraint_fpath} does not exist."
    assert constraint_fpath.endswith('.npy'), f"File {constraint_fpath} is not a .npy file."
    constraint = np.load(constraint_fpath)
    constraint = np.asarray(constraint)
    return constraint


if __name__ == "__main__":
    constraint_fpath = sys.argv[1]
    assert os.path.exists(constraint_fpath), f"File {constraint_fpath} does not exist."
    time_total = time.perf_counter()

    print("[INFO] Start calculating the function hulls...")

    methods = [
        # "single_sigmoid",
        "our_sigmoid",
        # "our_sigmoid-a",
        # "our_sigmoid-b",
        # "single_tanh",
        # "our_tanh",
    ]

    get_wrongs(reset=True)
    for method in methods:
        # print('>' * 50, method, '<' * 50)
        print(f"{GREEN}{BOLD}[{method.upper()}]{RESET} >>> {constraint_fpath} >>> ...")
        constraint = read_constraint_file(constraint_fpath)

        if method == "our_sigmoid":
            fun_hull = SigmoidHull(S=True)
            output = fun_hull.cal_hull(constrs=constraint)
        elif method == "our_tanh":
            fun_hull = TanhHull(S=True)
            output = fun_hull.cal_hull(constrs=constraint)
        # elif method == "our_maxpool_dlp":
        #     fun_hull = MaxPoolHullDLP(S=True)
        #     output = fun_hull.cal_hull(constrs=constraint)
        else:
            raise NotImplementedError(f"Method {method} is not implemented.")

    print(f"[INFO] Done in {time.perf_counter() - time_total:.2f} seconds")
