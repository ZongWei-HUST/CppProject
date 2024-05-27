import os
import cv2
import numpy as np
import requests
import torch
import torch.onnx
from torch import nn


# Download srcnn.pth
urls = ['https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth']
names = ['../models/srcnn.pth']
for url, name in zip(urls, names):
    if not os.path.exists(name):
        open(name, 'wb').write(requests.get(url).content)


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)
 
        self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4)
        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0)
        self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2)
 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def init_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=3)
    state_dict = torch.load('../models/srcnn.pth')['state_dict']
    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


def preprocess_img(input_img):
    # HWC -> NCHW
    input_img = np.transpose(input_img, [2, 0, 1])
    input_img = np.expand_dims(input_img, 0)
    return input_img

def infer(model, input_):
    return model(torch.from_numpy(input_)).detach().numpy()

def postprocess_img(torch_output):
    # NCHW -> HWC
    torch_output = np.squeeze(torch_output, 0)
    torch_output = np.clip(torch_output, 0, 255)
    output_img = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)
    return output_img


if __name__ == "__main__":
    model = init_torch_model()
    src_path = "../assets/face.png"
    trg_path = "../assets/face_result.png"

    input_img = cv2.imread(src_path).astype(np.float32)
    input_img = preprocess_img(input_img)
    result = infer(model, input_img)
    result = postprocess_img(result)
    cv2.imwrite(trg_path, result)