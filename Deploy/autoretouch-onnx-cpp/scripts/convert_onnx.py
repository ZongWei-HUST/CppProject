import os
import cv2
import numpy as np
import onnx
import torch
import onnxruntime
import torch.onnx
from torch import nn
from autoretouch import preprocess_img, postprocess_img, load_net


def torch2onnx(model, out_path):
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            out_path,
            opset_version=11,
            input_names=['input'],
            output_names=['output'])


def check_onnx(model_path):
    onnx_model = onnx.load(model_path)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print("Model incorrect")
    else:
        print("Model correct")

def postprocess_img(generate_numpy, h, w):
    generate_numpy = np.squeeze(generate_numpy, 0)
    generate_numpy = (np.transpose(generate_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    generate_numpy = generate_numpy.astype(np.uint8)
    generate_numpy = cv2.cvtColor(generate_numpy, cv2.COLOR_BGR2RGB)
    generate_numpy = cv2.resize(generate_numpy, (h, w))
    return generate_numpy


if __name__ == "__main__":
    out_path = "../models/autoretouch.onnx"
    model = load_net()
    torch2onnx(model, out_path)
    check_onnx(out_path)

    src_path = "../assets/HDF-cqq.jpg"
    trg_path = "../assets/HDF-cqq-result-onnx.jpg"

    input_img = cv2.imread(src_path)
    input_img, h , w = preprocess_img(input_img)
    input_img = input_img.numpy()

    ort_session = onnxruntime.InferenceSession(out_path)
    ort_inputs = {'input': input_img} 
    ort_output = ort_session.run(['output'], ort_inputs)[0]

    result = postprocess_img(ort_output, h, w)
    cv2.imwrite(trg_path, result)
