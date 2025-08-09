import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")   
import cv2
import pandas as pd
from typing import List, Any
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import is_image_file, default_loader
import tqdm
import functools
from PIL import Image
from augly.image import functional as aug_functional
from torchvision.transforms import functional as F
from collections import defaultdict
from src.utils.param_utils import seed_all
from torch.utils.data import Dataset


default_transform = transforms.Compose([transforms.ToTensor()])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                                std=[0.229, 0.224, 0.225])

import time
from torchvision import transforms
from torchvision.transforms import GaussianBlur
import random
from src.utils import data_utils
import torchvision

ATTACKS = ["clean", "gaussian_noise",
            "brightness", "gaussian_blur", 
            "resize", "salt_and_pepper_noise",
            "crop", "contrast", "saturation", "hue", "gamma", "sharpness","frame_average",
            "frame_drop", "frame_swap", "frame_insert", "frame_insert_gaussian_noise"]

def adjust_brightness(frames, factor: float):
    """
    Adjust the brightness of the frames.
    """
   
    return torch.stack([F.adjust_brightness(frame, factor) for frame in frames])

def adjust_contrast(frames, factor):
    """
    Adjust contrast of an image
    """
    return torch.stack([F.adjust_contrast(frame, factor) for frame in frames])

def adjust_saturation(frames, factor):
    """ 
    Adjust saturation of an image
    """
    return torch.stack([F.adjust_saturation(frame, factor) for frame in frames])

def adjust_hue(frames, factor):
    """ 
    Adjust hue of an image
    """

    return torch.stack([F.adjust_hue(frame, factor) for frame in frames])

def adjust_gamma(frames, gamma, gain=1):
    """ 
    Adjust gamma of an image
    """
    return torch.stack([F.adjust_gamma(frame, gamma, gain) for frame in frames])

def adjust_sharpness(frames, factor):
    """ Adjust sharpness of an image
    Args:
        x: PIL image
        sharpness_factor: sharpness factor
    """
    return torch.stack([F.adjust_sharpness(frame, factor) for frame in frames])

def gaussian_noise(frames, sigma: float):
    """
    Add Gaussian noise to the frames.
    """
    noise = torch.randn_like(frames) * sigma
    return torch.clamp(frames + noise, min=0.0, max=1.0)


def gaussian_blur(frames, kernel_size:int):

    blur = GaussianBlur(kernel_size=kernel_size, sigma = 2.0)
    adjusted_frames = torch.stack([blur(f) for f in frames])  

    return adjusted_frames  # (T, C, H, W)

def salt_and_pepper_noise(frames, noise_prob: float):
    """
    Add salt and pepper noise to the frames.
    """
    noise = torch.rand_like(frames)
    frames[noise < noise_prob / 2] = 1.0
    frames[noise > 1 - noise_prob / 2] = 0.0
    return frames

def resize_frames(frames: torch.Tensor, factor: float):
    """
    Resize video frames of shape (T, C, H, W) by a scale factor.
    """
    scale = np.sqrt(factor)
    new_edges_size = [int(s*scale) for s in frames.shape[-2:]][::-1]
    resized = torch.stack([F.resize(frame, new_edges_size) for frame in frames])
    return resized

def crop_frames(frames: torch.Tensor, factor: float):  
    """
    Crop the frames.
    """
    scale = np.sqrt(factor)
    new_edges_size = [int(s*scale) for s in frames.shape[-2:]][::-1]
    cropped = torch.stack([F.center_crop(x, new_edges_size) for x in frames])
    return cropped

def frame_drop(frames: torch.Tensor):

    idx = random.choice(range(len(frames)))
    frames = torch.cat([frames[:idx], frames[idx+1:]], dim=0)
    return frames

def frame_swap(frames: torch.Tensor):

    idx1, idx2 = torch.randperm(len(frames))[:2]
    frames[idx1], frames[idx2] = frames[idx2].clone(), frames[idx1].clone()
    return frames

def frame_insert_exist(frames: torch.Tensor):

    "insert idx1 frame to idx2 position"

    idx1, idx2 = torch.randperm(len(frames))[:2]
    frames = torch.cat([frames[:idx2], frames[idx1].unsqueeze(0), frames[idx2:]], dim=0)
    return frames

def frame_insert_gaussian_noise(frames: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    idx = random.randint(0, frames.shape[0]) 
    noise = torch.randn_like(frames[0]) * sigma
    noise = noise.to(dtype=frames.dtype, device=frames.device)
    return torch.cat([frames[:idx], noise.unsqueeze(0), frames[idx:]], dim=0)


def frame_average(frames: torch.Tensor, n: int = 1) -> torch.Tensor:
    
    T = frames.shape[0]
    assert n >= 1 and T >= n

    start = random.randint(0, T - n)
    averaged_frame = frames[start: start + n].mean(dim=0, keepdim=True)
    return torch.cat([frames[:start], averaged_frame, frames[start + n:]], dim=0)

def transform_video(video):
    transform = data_utils.default_transform()
    video_tensor = torch.stack([transform(frame) for frame in video])
    return video_tensor

class VideoDataset(Dataset):
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                                std=[0.229, 0.224, 0.225])
    def __init__(self, video_paths: str, attack: str = 'clean', factor: float = None):
        self.video_paths = self._read_video_paths(video_paths)
        self.cache = [np.load(p, mmap_mode=None) for p in self.video_paths]  
        self.attack = attack
        if factor is not None:
            self.factor = factor
    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, idx):
        frames = self.cache[idx]
        frames = torch.stack([self.to_tensor(frame) for frame in frames])

        if self.attack == 'gaussian_noise':
            frames = gaussian_noise(frames, self.factor)
        elif self.attack == 'brightness':
            frames = adjust_brightness(frames, self.factor)
        elif self.attack == 'gaussian_blur':
            frames = gaussian_blur(frames, self.factor)
        elif self.attack == 'resize':
            frames = resize_frames(frames, self.factor)
        elif self.attack == 'crop':
            frames = crop_frames(frames, self.factor)
        elif self.attack == 'contrast':
            frames = adjust_contrast(frames, self.factor)
        elif self.attack == 'saturation':
            frames = adjust_saturation(frames, self.factor)
        elif self.attack == 'hue':
            frames = adjust_hue(frames, self.factor)
        elif self.attack == 'gamma':
            frames = adjust_gamma(frames, self.factor)
        elif self.attack == 'sharpness':
            frames = adjust_sharpness(frames, self.factor)
        elif self.attack == 'salt_and_pepper_noise':
            frames = salt_and_pepper_noise(frames, self.factor)
        elif self.attack == 'frame_drop':
            frames = frame_drop(frames)
        elif self.attack == 'frame_swap':
            frames = frame_swap(frames)
        elif self.attack == 'frame_insert':
            frames = frame_insert_exist(frames)
        elif self.attack == 'frame_insert_gaussian_noise':
            frames = frame_insert_gaussian_noise(frames)
        elif self.attack == 'frame_average':
            frames = frame_average(frames, n = self.factor)
        
        return torch.stack([self.normalize(frame) for frame in frames])
    def _read_video_paths(self, path: str):

        paths = []
        for current_path, _, files in os.walk(path):
            for filename in files:
                if filename.endswith('.npy'): 
                    full_path = os.path.join(current_path, filename)
                    paths.append(full_path)
        return sorted(paths)
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--attack_type', type=str, default='clean')
    parser.add_argument('--factor', type=float, default=2.0)
    parser.add_argument('--frame_array_path', type=str, default='.')
    parser.add_argument('--msg_decoder_path', type=str, default='./ckpts/msg_decoder/dec_48b_whit.torchscript.pt')
    parser.add_argument('--key', type=str, default='100011100001001101101100100011111101111110000000')
    params = parser.parse_args()

    os.makedirs(params.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    msg_decoder = torch.jit.load(params.msg_decoder_path).to(device)
    msg_decoder.eval()
    key = data_utils.list_to_torch(data_utils.str_to_list(params.key)).to(device)

    assert params.attack_type in ATTACKS, f"Attack type is not supported"

    accuracys = []
    tprs = {}
    fprs = ['1e-13', '1e-12', '1e-11', '1e-10', '1e-9', '1e-8', 
            '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1', '1']
    match_thresholds = [47, 46, 45, 44, 43, 42, 41, 40, 38, 37, 35, 32, 28, 0]
    for fpr in fprs:
        tprs[fpr] = []

    video_dataset = VideoDataset(params.frame_array_path, params.attack_type, params.factor)

    dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=16)
    for _, frames in enumerate(dataloader):
        frames = frames.to(device).squeeze(0).to(torch.float32)
        keys = key.repeat(frames.shape[0], 1)
        with torch.no_grad():
            decoded = msg_decoder(frames)
            binarized = (decoded > 0).int()
            votes = binarized.sum(dim=0)  
            majority = (votes > (decoded.size(0) // 2)).int().reshape(key.shape)  
        diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b
        bit_accs = (torch.sum(diff, dim=-1) / diff.shape[-1]).mean().item() # b k -> b
        diff_video = (~torch.logical_xor(majority, key>0))
        bit_video_acc = torch.sum(diff_video).item() / diff_video.numel()
        match_bit = torch.sum(diff_video).item()
        for j, fpr in enumerate(fprs):
            threshold = match_thresholds[j]
            is_success = match_bit >= threshold
            tprs[fpr].append(int(is_success))

    accuracys[params.attack_type].append(bit_video_acc)
    print(f'{params.attack_type} accuracy: {bit_video_acc}')

    with open(os.path.join(params.output_dir, 'output_tpr.txt'), 'a') as f:
        f.write(f'{params.attack_type} {params.factor}\n')     
        for fpr in fprs:
            f.write(f'{params.attack_type} fpr = {fpr}: {np.mean(tprs[fpr])}\n')

if __name__ == "__main__":

    main()
    