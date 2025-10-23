# pip install torch onnx onnxruntime
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)
        self.act = nn.ReLU()
        # 固定一组权重，便于与论文示例一致（可替换）
        with torch.no_grad():
            self.fc1.weight[:] = torch.tensor([[1, 1], [1,  -1]])
            self.fc1.bias[:]   = torch.tensor([0, 0])
            self.fc2.weight[:] = torch.tensor([[1, 1], [1,  -1]])
            self.fc2.bias[:]   = torch.tensor([0, 0])
            self.fc3.weight[:] = torch.tensor([[1, 1], [0,  1]])
            self.fc3.bias[:]   = torch.tensor([1, 0])

    def forward(self, x):
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))

net = TinyNet().eval()
dummy = torch.randn(1, 2)           # 占位输入
torch.onnx.export(
    net, dummy, 'deeppoly.onnx',
    input_names=['x'], output_names=['y'],
    opset_version=13, do_constant_folding=True,
    dynamic_axes={'x': {0: 'N'}, 'y': {0: 'N'}}
)


import onnx

model = onnx.load("deeppoly.onnx")
graph = model.graph

# 重新连接 identity 的输入和输出
to_remove = []
for node in graph.node:
    if node.op_type == "Identity":
        identity_in = node.input[0]
        identity_out = node.output[0]
        # 更新引用这个输出的其他节点
        for n in graph.node:
            for i, inp in enumerate(n.input):
                if inp == identity_out:
                    n.input[i] = identity_in
        # 如果它是模型的输出，也要改成 identity_in
        for out in graph.output:
            if out.name == identity_out:
                out.name = identity_in
        to_remove.append(node)

for node in to_remove:
    graph.node.remove(node)

onnx.save(model, "deeppoly.onnx")
print("Removed", len(to_remove), "Identity nodes")


# onnxruntime 验证
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession('deeppoly.onnx', providers=['CPUExecutionProvider'])
# x = np.array([[0.2, 0.1], [-1.0, 2.0]], np.float32)
x = np.array([[0, 0]], np.float32)
y = sess.run(None, {'x': x})
label = np.argmax(y[0], axis=1)
np.save("train_data.npy", x)
np.save("train_labels.npy", label)
np.save("test_data.npy", x)
np.save("test_labels.npy", label)
for i1 in range(100):
    for i2 in range(100):
        x[0,0] = i1 / 100 * 2.0 - 1.0
        x[0,1] = i2 / 100 * 2.0 - 1.0
        y = sess.run(None, {'x': x})
        label = np.argmax(y[0], axis=1)
        print('.', end='')
        assert label[0] == 0, f"Wrong label for input {x[0]}: got {label[0]}"
        # print(f"x: {x[0]} => y: {y[0]} => label: {label}")
    print()
print("All inputs are classified correctly.")
