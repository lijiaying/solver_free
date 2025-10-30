#!/bin/bash

python3 exp.py --net_fpath ../nets/maxpool/mnist_conv_maxpool.onnx --epsilon 0.01 --bp crown --opt mnlp
python3 exp.py --net_fpath ../nets/maxpool/mnist_conv_maxpool.onnx --epsilon 0.01 --bp deeppoly --opt mnlp

