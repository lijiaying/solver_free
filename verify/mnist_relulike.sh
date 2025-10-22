#!/bin/bash

python3 exp.py --net_fpath ../nets/mnist_leaky_relu_3_500.onnx --epsilon 0.03 --bp rover --log_name mnist_leaky_relu_3_500_ROVER

python3 exp.py --net_fpath ../nets/mnist_leaky_relu_3_500.onnx --epsilon 0.03 --bp rover --opt lp --log_name mnist_leaky_relu_3_500_ROVER_LP

python3 exp.py --net_fpath ../nets/mnist_leaky_relu_3_500.onnx --epsilon 0.03 --bp rover --opt mnlp --log_name mnist_leaky_relu_3_500_ROVER_MNLP