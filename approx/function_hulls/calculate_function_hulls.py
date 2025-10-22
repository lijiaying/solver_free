"""This is for calculating the function hull with given constraints and bounds."""
from setproctitle import setproctitle
setproctitle("python check_sma_value_equivalence.py")
import os
import sys
import time
import warnings

cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, "../..")
sys.path.insert(0, "../../../ELINA/python_interface/")

import numpy as np
from utils import *

try:
    from fconv import ftanh_orthant, fsigm_orthant, fkpool  # noqa
except ImportError as e:
    warnings.warn(
        f"[WARNING] ELINA is not installed, so we cannot use some methods in fconv: {e}"
    )


from src.funchull.acthull import *
from src.funchull.ablation_study import *


def read_constraints_and_bounds(constraints_file_path: str, bounds_file_path: str):
    with open(constraints_file_path, "r") as f:
        constraints = f.readlines()
    with open(bounds_file_path, "r") as f:
        bounds = f.readlines()
    constraints = [eval(constraint) for constraint in constraints]
    bounds = [eval(bound) for bound in bounds]
    return constraints, bounds


if __name__ == "__main__":
    time_total = time.perf_counter()

    print("[INFO] Start calculating the function hulls...")

    # Get the current directory and go to the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    constraints_dir = os.path.join(current_dir, "../polytope_samples")
    bounds_dir = os.path.join(current_dir, "../polytope_bounds")
    constraints_files = os.listdir(constraints_dir)
    constraints_files = [file for file in constraints_files if file.endswith(".txt") and (not file.endswith("_oct.txt"))]
    # Sort by dimension
    constraints_files.sort(key=lambda x: int(x.split(".")[-2].split("_")[1][:-1]))
    methods = [
        "single_sigmoid",
        "single_tanh",
        # "single_maxpool",
        # "single_leakyrelu",
        # "prima_sigmoid",
        # "prima_tanh",
        # "prima_maxpool",
        "our_sigmoid",
        "our_tanh",
        # "our_maxpool_dlp",
        # "our_leakyrelu",
        "our_sigmoid-a",
        "our_sigmoid-b",
        "our_tanh-a",
        "our_tanh-b",
        # "our_maxpool_dlp-a",
    ]
    methods = [
        "single_sigmoid",
        # "single_maxpool",
        # "single_leakyrelu",
        # "prima_sigmoid",
        # "prima_tanh",
        # "prima_maxpool",
        "our_sigmoid",
        # "our_maxpool_dlp",
        # "our_leakyrelu",
        "our_sigmoid-a",
        "our_sigmoid-b",
        "single_tanh",
        "our_tanh",
        "our_tanh-a",
        "our_tanh-b",
        # "our_maxpool_dlp-a",
    ]
    methods = [
        # "single_sigmoid",
        # "prima_sigmoid",
        "our_sigmoid",
        # "our_sigmoid-a",
        # "our_sigmoid-b",
        # "single_tanh",
        # "our_tanh",
    ]

    nf = len(constraints_files)
    for i, constraints_file in enumerate(constraints_files):
        get_wrongs(reset=True)
        print('=' * 50, f"{i+1}/{nf}: {constraints_file}", '=' * 50)
        for method in methods:
            method_start = time.perf_counter()
            # print('>' * 50, method, '<' * 50)
            dim = int(constraints_file.split(".")[-2].split("_")[-2][:-1])
            _constraints_file = constraints_file.replace(".txt", "_oct.txt") if "prima" in method else constraints_file
            constraints_file_path = os.path.abspath(os.path.join(constraints_dir, _constraints_file))
            bounds_file_path = os.path.abspath(os.path.join(bounds_dir, constraints_file.replace(".txt", "_bounds.txt")))
            # else:
            #     print(f"{RED}{CROSS}{RESET} Skipping {constraints_file} with {method}...")
            #     continue
            if dim >= 4:
                break

            if dim > 4 and (
                "prima" in method
                or "-a" in method
                or "-b" in method
                or "single" in method
            ):
                # PRIMA does not support dimensions greater than 4
                continue

            print(f"{GREEN}{BOLD}[{method.upper()}]{RESET} --> {constraints_file_path} ...")

            constraints_list, bounds_list = read_constraints_and_bounds(constraints_file_path, bounds_file_path)

            bounds_file_path = os.path.basename(bounds_file_path)
            saved_file_path = bounds_file_path.replace("_bounds.txt", f"_{method}.txt").split("\\")[-1]

            file = open(saved_file_path, "w")
            for n, (constraints, bounds) in enumerate(zip(constraints_list, bounds_list)):
                # if n >= 5:
                #     break
                lb, ub = bounds
                lb = np.asarray(lb)
                ub = np.asarray(ub)

                constraints = np.asarray(constraints)
                d = constraints.shape[1] - 1

                time_cal = time.perf_counter()
                output_constraints = None
                if method == "prima_sigmoid":
                    try:
                        output_constraints = fsigm_orthant(constraints)
                    except Exception as e:
                        print(f"[WARNING] Failed to calculate hull for {constraints_file_path} with {method}: {e}")

                elif method == "prima_tanh":
                    try:
                        output_constraints = ftanh_orthant(constraints)
                    except Exception as e:
                        print(f"[WARNING] Failed to calculate hull for {constraints_file_path} with {method}: {e}")
                # elif method == "prima_maxpool":
                #     try:
                #         output_constraints = fkpool(constraints)  # noqa
                #     except Exception as e:
                #         print(f"[WARNING] Failed to calculate hull for {constraints_file_path} with {method}: {e}")
                elif method == "our_sigmoid":
                    fun_hull = SigmoidHull(S=True)
                    output_constraints = fun_hull.cal_hull(constrs=constraints)
                elif method == "our_tanh":
                    fun_hull = TanhHull(S=True)
                    output_constraints = fun_hull.cal_hull(constrs=constraints)
                # elif method == "our_maxpool_dlp":
                #     fun_hull = MaxPoolHullDLP(S=True)
                #     output_constraints = fun_hull.cal_hull(constrs=constraints)
                # elif method == "our_leakyrelu":
                #     fun_hull = LeakyReLUHull(S=True)
                #     output_constraints = fun_hull.cal_hull(constrs=constraints)
                elif method == "our_sigmoid-a":
                    fun_hull = SigmoidHullA(S=True)
                    output_constraints = fun_hull.cal_hull(constrs=constraints)
                elif method == "our_sigmoid-b":
                    fun_hull = SigmoidHullB(S=True)
                    output_constraints = fun_hull.cal_hull(constrs=constraints)
                elif method == "our_tanh-a":
                    fun_hull = TanhHullA(S=True)
                    output_constraints = fun_hull.cal_hull(constrs=constraints)
                elif method == "our_tanh-b":
                    fun_hull = TanhHullB(S=True)
                    output_constraints = fun_hull.cal_hull(constrs=constraints)
                # elif method == "our_maxpool_dlp-a":
                #     fun_hull = MaxPoolDLPHullA(S=True)
                #     output_constraints = fun_hull.cal_hull(constrs=constraints)
                elif method == "single_sigmoid":
                    fun_hull = SigmoidHull(S=True, M=False)
                    output_constraints = fun_hull.cal_hull(lower=lb, upper=ub)
                elif method == "single_tanh":
                    fun_hull = TanhHull(S=True, M=False)
                    output_constraints = fun_hull.cal_hull(lower=lb, upper=ub)
                # elif method == "single_maxpool":
                #     fun_hull = MaxPoolHullDLP(S=True, M=False)
                #     output_constraints = fun_hull.cal_hull(lower=lb, upper=ub)
                # elif method == "single_leakyrelu":
                #     fun_hull = LeakyReLUHull(S=True, M=False)
                #     output_constraints = fun_hull.cal_hull(lower=lb, upper=ub)
                else:
                    raise NotImplementedError(f"Method {method} is not implemented.")
                time_cal = time.perf_counter() - time_cal

                if output_constraints is None:
                    continue

                output_constraints = output_constraints.tolist()

                file.write(
                    f"{time_cal}\t{len(output_constraints)}\t{output_constraints}\t"
                    f"{lb.tolist()}\t{ub.tolist()}\n"
                )

            file.close()
            method_time = time.perf_counter() - method_start
            print(f"[INFO] {method} takes {method_time:.2f} seconds for {n+1} cases.")
            print(f"[INFO] Save to {saved_file_path}")
            print()
        u, l = get_wrongs()
        print(f"UPPER: {u}, LOWER: {l}")

    print(f"[INFO] Finish {i+1}/{len(constraints_files)} files.")
    print(f"[INFO] Done in {time.perf_counter() - time_total:.2f} seconds")
