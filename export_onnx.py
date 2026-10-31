"""Convert the check_orientation PyTorch weights to ONNX. Only used during the Docker build.

Usage: python export_onnx.py <checkpoint.pth> <output.onnx>
"""
import sys

import timm
import torch

checkpoint_path, output_path = sys.argv[1:3]

# ResNeXt-50 with 4 classes: the image was turned 0, 1, 2 or 3 times by 90°
model = timm.create_model("resnext50_32x4d", pretrained=False, num_classes=4)
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]
# The checkpoint was saved from a training wrapper that prefixes every key with "model."
model.load_state_dict({key.removeprefix("model."): value for key, value in state_dict.items()})
model.eval()

torch.onnx.export(
    model,
    torch.zeros(1, 3, 224, 224),
    output_path,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    dynamo=False,
)
