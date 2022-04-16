import cv2
import torch
import numpy as np
import torch.nn as nn


def Canny(image_batch):
    # 转化为numpy
    device = image_batch.device
    np_batch = image_batch.detach().cpu().numpy()
    # 调整维度顺序
    np_batch = np.transpose(np_batch, (0, 2, 3, 1))
    # 进行canny边缘提取
    for idx, np_img in enumerate(np_batch):
        # 转为cv2支持的图像
        img = np.uint8(np_img * 255)
        # 提取边沿
        edge_image = cv2.Canny(img, 100, 200) / 255.0
        edge_image = torch.from_numpy(edge_image).unsqueeze(0)
        if idx == 0:
            canny_batch = edge_image
        else:
            canny_batch = torch.cat((canny_batch, edge_image), dim=0)
    canny_batch = canny_batch.unsqueeze(1)
    return canny_batch.to(device)


def image_gradient(input, gray=True):
    input_device = input.device
    base_kernel = torch.tensor([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]], dtype=torch.float32).to(input_device)
    # base_kernel = torch.tensor([[0,  1,  0],
    #                             [1, -4,  1],
    #                             [0,  1,  0]], dtype=torch.float32).to(device)
    if gray:
        conv_op = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1).to(input_device)

        kernel = base_kernel.reshape((1, 1, 3, 3))
        conv_op.weight.data = kernel
        return conv_op(input)
    else:
        conv_op = nn.Conv2d(3, 1, kernel_size=3, bias=False, padding=1).to(input_device)

        kernel = torch.zeros((1, 3, 3, 3), dtype=torch.float32).to(input_device)
        for i in range(3):
            kernel[:, i] = base_kernel
        conv_op.weight.data = kernel
        return conv_op(input)

def gradient_Gray(input):
    input_device = input.device
    conv_op = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1).to(input_device)
    kernel = torch.tensor([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=torch.float32).to(input_device)
    # kernel = torch.tensor([[0,  1, 0],
    #                        [1, -4, 1],
    #                        [0,  1, 0]], dtype=torch.float32).to(input_device)
    kernel = kernel.reshape((1, 1, 3, 3))
    conv_op.weight.data = kernel

    return conv_op(input)
