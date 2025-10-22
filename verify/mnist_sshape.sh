#!/bin/bash

python3 exp.py --net_fpath ../nets/mnist_sigmoid_3_500.onnx --epsilon 0.02 --bp crown --opt mnlp --log_name mnist_sigmoid_3_500_CROWN_MNLP

python3 exp.py --net_fpath ../nets/mnist_sigmoid_3_500.onnx --epsilon 0.02 --bp deeppoly --opt mnlp --log_name mnist_sigmoid_3_500_DEEPPOLY_MNLP

python3 exp.py --net_fpath ../nets/mnist_tanh_3_500.onnx --epsilon 0.01 --bp crown --opt mnlp --log_name mnist_tanh_3_500_CROWN_MNLP

python3 exp.py --net_fpath ../nets/mnist_tanh_3_500.onnx --epsilon 0.01 --bp deeppoly --opt mnlp --log_name mnist_tanh_3_500_DEEPPOLY_MNLP