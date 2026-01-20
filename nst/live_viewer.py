import cv2
import numpy as np
import torch
from PIL import Image

def tensor_to_cv2(tensor):
    img = tensor.clone().detach().cpu().squeeze(0)
    img = img.permute(1,2,0)
    img = img.numpy()
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def tensor_to_pil(tensor):
    img = tensor.detach().cpu()
    img = img * std + mean         
    img = torch.clamp(img, 0, 1)
    img = img.squeeze(0).permute(1,2,0)
    img = (img.numpy() * 255).astype("uint8")
    return Image.fromarray(img)
