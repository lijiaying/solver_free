#!/bin/bash

python3 exp.py --net_fpath ../nets/maxpool/mnist_conv_maxpool.onnx --epsilon 0.01 --bp crown --opt mnlp --log_name mnist_conv_maxpool_CROWN_MNLP
python3 exp.py --net_fpath ../nets/maxpool/mnist_conv_maxpool.onnx --epsilon 0.01 --bp deeppoly --opt mnlp --log_name mnist_conv_maxpool_DEEPPOLY_MNLP

