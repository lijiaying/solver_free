# X

This is an artifact for the tool called **X**, which over-approximates the function hulls of various activation functions (including ReLU, sigmoid, tanh, and maxpool).


The expected outputs of `bash test_X.sh` is to verify a small instance of local robustness verification, and it prints the instance to verify is `UNKNOWN`, which means our sound approach cannot decide this instance is verified or not. Another possible result is `SAT`, which means the instance is verified and the local robustness is satisfied. If you see the output like this, it means the installation is successful and the tool works well.


## Python Environment

```cmd
conda create --name X python=3.12
conda activate X
```

## Python Libraries

**PyTorch**

- pytorch==2.3.1
- torchvision==0.18.1
- torchaudio==2.3.1
- pytorch-cuda=12.1

**Gurobi**

Then, you need the library for the programming solver Gurobi and install it by `pip` (not supported by `conda`).

```cmd
pip install gurobipy==11.0.3
```

**SciPy, ONNX, ONNXRuntime, Numba**

You also need the following libraries, where `scipy` is for extracting sparse constraint matrices from `gurobipy`, `onnx` is for reading ONNX format models, `onnxruntime` is for checking the correctness of ONNX models, and `numba` is for compiling some functions (about tangent lines) to speed up the code.

```cmd
pip install scipy==1.15.3 onnx==1.18.0 onnxruntime==1.22.0 numba==0.61.2
```

**CDDLib**

The following library _needs a specific version_ because there are some changes in the latest version of `pycddlib` that will cause the code to fail. We use version `2.1.7` in our experiments. `pycddlib` is for calculating the vertices of a convex polytope in this work. It is compatible with `numpy` operations.

```cmd
pip install pycddlib==2.1.7
```


## Quick Kick-the-Tires

There is an example script `exp_test.py` to run a small instance of local robustness verification. You can run this Python script by the following command `bash test.sh` in the `evaluation_verification` folder to check if the tool works well.

```cmd
cd X/evaluation_verification
bash test.sh
```

## Evaluation: verify 

The `evaluation_verification` folder contains the evaluation code for local robustness verification. To run the following commands, make sure you are in the corresponding subfolder:

```cmd
cd X/evaluation_verification
```

You need to first download the benchmark ONNX models from [Google Drive link](https://drive.google.com/drive/folders/1C4kYaKb_Pd3xCo6aCy6W80tw43CM8Nn8?usp=sharing). All the evaluation scripts call `exp.py` for running experiments. You can see the following bash files in the `evaluation_verification` folder:

- `mnist_sshape.sh`: for S-shape functions (sigmoid, tanh) on the MNIST dataset (Table 4 in the paper)
- `cifar10_sshape.sh`: for S-shape functions (sigmoid, tanh) on the CIFAR-10 dataset (Table 4 in the paper)
- `maxpool.sh`: for maxpool on the MNIST and CIFAR-10 datasets (Table 4 in the paper)
- `resnet.sh`: for the ResNet benchmark on the CIFAR-10 dataset (Table 6 in the paper)

Run the above bash files, and you can collect the log files in the `evaluation_verification/logs` folder.

> **ATTENTION**: Running all the code takes a long time (total 2~3 days, details shown in the paper), so we suggest you run them one by one. You can also comment out some lines in the bash files to reduce the number of experiments.

```cmd
bash mnist_sshape.sh
bash mnist_relulike.sh
bash maxpool.sh
bash cifar10_sshape.sh
bash cifar10_relulike.sh
bash resnet.sh
```

# Reuse the Source Code

## Main Code Structure

The folder structure of this repository is as follows. We only list the main folders and files here. The source code of X is in the `src/` folder, which contains the code for bound propagation, function hull approximation, linear programming, model building, and utility functions. The evaluation code is in the `evaluation_volume` and `evaluation_verification` folders.

```
X/                      # Main folder of X
├── .temp/                   # Auto-created temporary files (e.g., downloaded datasets)
├── archived_logs/           # Archived logs of our paper (download from Google Drive)
├── evaluation_volume/       # Code for volume evaluation of convex hull approximation
├── evaluation_verification/ # Code for local robustness verification
├── nets/                    # Benchmark ONNX models (download from Google Drive)
├── src/                     # Source code of X
│   ├── boundprop/           # Code for bound propagation
│   │   ├── ineq/            # Linear inequalities bounds for propagation
│   │   │   ├── backsub/     # Symbolic back-substitution for inequalities
│   │   │   └── relaxation/  # Linear relaxation for activation functions
│   │   └── base.py          # Base classes for propagation
│   ├── funchull/            # Code for function hull approximation
│   ├── linprog/             # Code for linear programming
│   ├── model/               # Code for verification models
│   └── utils/               # Miscellaneous utility functions
└── README.md
```


- If you want to use new dataset or new models, you can take the example `evaluation_verification/exp.py` as a reference. You possibly need to modify the model arguments and the preprocessing for the dataset.
- If you want to extend new activation functions, you need to design how to calculate their relaxation in the `src/boundprop/ineq/relaxation/` folder and define their layer classes in the `src/boundprop/ineq/` folder. The back-substitution methods commonly have no need to be modified.
- If you want to extend new linear programming methods, you can refer to the `src/linprog/` folder and implement your own linear programming methods.
- If you want to extend new function hull approximation methods, you can refer to the `src/funchull/` folder and implement your own methods.
- The folder `src/model/` contains the verification models, which define the behaviors of the whole verification process. If you need to introduce new model topological structures, you may need to modify the model classes in this folder.
- The folder `src/utils/` contains some utility functions, which are used in the other modules. You can add your own utility functions here.


