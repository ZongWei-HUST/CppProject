import os
import cv2
import numpy as np
import onnx
import torch
import onnxruntime
import torch.onnx
from torch import nn
from srcnn import SuperResolutionNet, init_torch_model
from srcnn import preprocess_img, postprocess_img


def torch2onnx(model, out_path):
    x = torch.randn(1, 3, 256, 256)
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


def infer(model, input_):
    return model(torch.from_numpy(input_)).detach().numpy()

def postprocess_img(torch_output):
    # NCHW -> HWC
    torch_output = np.squeeze(torch_output, 0)
    torch_output = np.clip(torch_output, 0, 255)
    output_img = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)
    return output_img


if __name__ == "__main__":
    out_path = "../models/srcnn.onnx"

    model = init_torch_model()
    torch2onnx(model, out_path)
    check_onnx(out_path)

    src_path = "../assets/face.png"
    trg_path = "../assets/face_result_onnx.png"

    input_img = cv2.imread(src_path).astype(np.float32)
    input_img = preprocess_img(input_img)

    ort_session = onnxruntime.InferenceSession(out_path)
    ort_inputs = {'input': input_img} 
    ort_output = ort_session.run(['output'], ort_inputs)[0]

    ort_output = postprocess_img(ort_output)
    cv2.imwrite(trg_path, ort_output)
