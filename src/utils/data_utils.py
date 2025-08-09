import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch

import cv2
import pandas as pd
from typing import List, Any
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import is_image_file, default_loader

import functools
from PIL import Image
from augly.image import functional as aug_functional
from torchvision.transforms import functional
from collections import defaultdict
from src.utils.param_utils import seed_all

normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225]) # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                           std=[1/0.229, 1/0.224, 1/0.225]) # Unnormalize (x * std) + mean

normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                           std=[0.5, 0.5, 0.5]) # Normalize (x - 0.5) / 0.5
    
unnormalize_vqgan = transforms.Normalize(mean=[-1, -1, -1], 
                                             std=[1/0.5, 1/0.5, 1/0.5]) # Unnormalize (x * 0.5) + 0.5

def list_to_str(key: List[int]):
    """
    convert list watermark key to str
    """
    return ''.join(str(bit) for bit in key)

def str_to_list(key: str):
    """
    convert list watermark key to str
    """
    return [int(item) for item in key if item in '0' or item in '1']

def list_to_numpy(key: List[int]):
    """
    convert list watermark key to numpy NDAaray
    """
    return np.array(key)

def list_to_torch(key: List[int]):
    """
    convert list watermark key to torch Tensor
    """
    return torch.tensor(key, dtype = torch.float32)

def torch_to_str(key: torch.Tensor):
    """
    convert torch Tensor watermark key to str
    """
    list_key = (key > 0).int().tolist()
    #if len(list_key) > 1:
    return [list_to_str(item) for item in list_key]

def default_transform():

    return transforms.Compose([
                transforms.ToTensor(),
                normalize_img])

def img_transform(img_size: int):

    return  transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize_img])

def vqgan_transform(img_size: int):

    return  transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize_vqgan
                ])

def vqgan_to_imnet():
    
    return transforms.Compose([unnormalize_vqgan, normalize_img])

def psnr(x, y, img_space='vqgan'):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == 'vqgan':
        delta = torch.clamp(unnormalize_vqgan(x), 0, 1) - torch.clamp(unnormalize_vqgan(y), 0, 1)
    elif img_space == 'img':
        delta = torch.clamp(unnormalize_img(x), 0, 1) - torch.clamp(unnormalize_img(y), 0, 1)
    else:
        delta = x - y
    delta = 255 * delta
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]) # BxCxHxW
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2, dim=(1,2,3)))  # B
    return psnr

def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.center_crop(x, new_edges_size)

def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)

def rotate(x, angle):
    """ Rotate image by angle
    Args:
        x: image (PIl or tensor)
        angle: angle in degrees
    """
    return functional.rotate(x, angle)


def adjust_brightness(x, brightness_factor):
    """ Adjust brightness of an image
    Args:
        x: PIL image
        brightness_factor: brightness factor
    """
    return normalize_img(functional.adjust_brightness(unnormalize_img(x), brightness_factor))

def adjust_contrast(x, contrast_factor):
    """ Adjust contrast of an image
    Args:
        x: PIL image
        contrast_factor: contrast factor
    """
    return normalize_img(functional.adjust_contrast(unnormalize_img(x), contrast_factor))

def adjust_saturation(x, saturation_factor):
    """ Adjust saturation of an image
    Args:
        x: PIL image
        saturation_factor: saturation factor
    """
    return normalize_img(functional.adjust_saturation(unnormalize_img(x), saturation_factor))

def adjust_hue(x, hue_factor):
    """ Adjust hue of an image
    Args:
        x: PIL image
        hue_factor: hue factor
    """
    return normalize_img(functional.adjust_hue(unnormalize_img(x), hue_factor))

def adjust_gamma(x, gamma, gain=1):
    """ Adjust gamma of an image
    Args:
        x: PIL image
        gamma: gamma factor
        gain: gain factor
    """
    return normalize_img(functional.adjust_gamma(unnormalize_img(x), gamma, gain))

def adjust_sharpness(x, sharpness_factor):
    """ Adjust sharpness of an image
    Args:
        x: PIL image
        sharpness_factor: sharpness factor
    """
    return normalize_img(functional.adjust_sharpness(unnormalize_img(x), sharpness_factor))



def collate_fn(batch):
    """ Collate function for data loader. Allows to have img of different size"""
    return batch

@functools.lru_cache()
def read_metadata(path) -> pd.DataFrame:
    return pd.read_csv(path)

class SubOpenVid(Dataset):

    """OpenVid Dataset"""
    
    def __init__(self,
                 metadata_path: str,
                 data_dir: str,
                 num_frames: int,
                 frame_interval: int,
                 transform: transforms = None) -> None:
        
        self.metadata = read_metadata(metadata_path)
        if transform:
            self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.videos = self._get_video_paths(data_dir)


    @functools.lru_cache()
    def _get_video_paths(self, path:str):
        """
        get the image paths, filter the videos that cannot be sampled in condition of "num_frames" and "interval"
        """
        self.metadata.set_index('video', inplace=True)
        paths = []
        for current_path, _, files in os.walk(path):

            for filename in files:
                if filename.endswith(('.mp4', '.avi', '.mov')): 
                    full_path = os.path.join(current_path, filename)
                    video_name = os.path.basename(filename)
                    frame_count = self.metadata.at[video_name, 'frame']
                    if frame_count >= (self.num_frames - 1) * self.frame_interval + 1:  
                        paths.append(full_path)
        self.metadata.reset_index(inplace=True)
        return sorted(paths)
    
    def read_video_frames(self, video_path: str) -> torch.Tensor:
   
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  

        sampled_indices = np.arange(0, total_frames, self.frame_interval)[:self.num_frames]  
        frames = []

        for frame_id in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id) 
            ret, frame = cap.read()
            if not ret:
                break  

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)  
        
            frames.append(frame)

        cap.release()

        return torch.stack(frames)
    
    def __getitem__(self, index: int) -> Any:
        assert 0 <= index < len(self), "invalid index"
        frames = self.read_video_frames(self.videos[index])

        return frames

    def __len__(self) -> int:

        return len(self.videos)


def get_video_dataloader(
                        metadata_path: str,
                        data_dir: str, 
                        num_frames: int,
                        frame_interval: int,
                        transform: transforms, 
                        num_videos: int = None, 
                        batch_size: int = 1,
                        num_workers: int = 16,
                        shuffle: bool = False, 
                        collate_fn: Any = collate_fn):
    
    """ Get dataloader"""
    dataset = SubOpenVid(metadata_path, data_dir, num_frames, frame_interval, transform)
    if num_videos is not None:
        assert num_videos < len(dataset)
        dataset = Subset(dataset, np.random.choice(len(dataset), num_videos, replace=False))

    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, 
                      num_workers = num_workers, collate_fn = collate_fn)


if __name__ == "__main__":

    print('testing')

    metadata_path = '/hpc2hdd/home/yhuang489/OpenVid/data/train/OpenVid-1M.csv'
    data_dir = '/hpc2hdd/home/yhuang489/OpenVid/eval'
    num_frames = 30
    frame_interval = 1
    transform = vqgan_transform(img_size = 512)
    num_videos = 10

    print("build loader")
    loader = get_video_dataloader(metadata_path,
                                  data_dir,
                                  num_frames,
                                  frame_interval,
                                  transform,
                                  num_videos,
                                  collate_fn=None)
    print("done")
    
    for x in loader:
        print(x.shape)
        break