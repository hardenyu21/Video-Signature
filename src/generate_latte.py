import warnings
warnings.filterwarnings("ignore")

import torch
from diffusers import LattePipeline
import os
import sys
sys.path.append(os.getcwd())
from utils.param_utils import seed_all, get_params
from utils.data_utils import read_metadata
from torchvision.datasets.folder import is_image_file
import random
import json
import torch
import numpy as np
from src.utils import data_utils
from diffusers import LattePipeline
from diffusers.models import AutoencoderKLTemporalDecoder, AutoencoderKL
from torchvision.utils import save_image
import torch
import copy
import torch
from torchvision import transforms
import time
from diffusers import StableVideoDiffusionPipeline, TextToVideoSDPipeline
from diffusers.utils import load_image, export_to_video
import torchvision
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                                std=[0.229, 0.224, 0.225])


def get_images(path:str, num_eval: int):
    """
    get the image paths
    """
    paths = []
    for current_path, _, files in os.walk(path):
        for filename in files:
            paths.append(os.path.join(current_path, filename))
    
    paths = sorted([fn for fn in paths if is_image_file(fn)])
    assert num_eval <= len(paths) 
    images = random.sample(paths, num_eval)

    return images

def transform_video(video):
    transform = data_utils.default_transform()
    video_tensor = torch.stack([transform(frame) for frame in video])
    return video_tensor

def main(params):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #prompts = get_prompt(file_path = params.file_path, num_eval = params.num_eval)
    with open(params.prompt_file, 'r') as f:
        prompts = [line.strip() for line in f]
    ##load message decoder
    try:
        msg_decoder = torch.jit.load(params.msg_decoder_path)
    except:
        raise KeyError(f"No checkpoint found in {params.msg_decoder_path}")
    for msg_decoder_params in msg_decoder.parameters():
        msg_decoder_params.requires_grad = False
    msg_decoder.eval()
    
    key = data_utils.list_to_torch(data_utils.str_to_list(params.key))

    pipe = LattePipeline.from_pretrained(params.model_id, torch_dtype = torch.float16).to(device)
    watermarked_vae_ms = AutoencoderKL.from_pretrained('damo-vilab/text-to-video-ms-1.7b', subfolder = 'vae').to(device)
    watermarked_vae_svd = AutoencoderKLTemporalDecoder.from_pretrained('stabilityai/stable-video-diffusion-img2vid-xt', subfolder = 'vae').to(device)
    watermarked_vae_ms.load_state_dict(torch.load(params.ckpt_path_ms, map_location = 'cpu'))
    watermarked_vae_svd.load_state_dict(torch.load(params.ckpt_path_svd, map_location = 'cpu'))
    watermarked_vae_ms.to(pipe.vae.dtype)
    watermarked_vae_svd.to(pipe.vae.dtype)
    msg_decoder.to(pipe.vae.dtype)


    if not os.path.exists(params.nw_saved_dir):
        os.makedirs(params.nw_saved_dir)
    
    if not os.path.exists(params.w_saved_dir_ms):
        os.makedirs(params.w_saved_dir_ms)
    if not os.path.exists(params.w_saved_dir_svd):
        os.makedirs(params.w_saved_dir_svd)

    bit_accuracy = []
    video_bit_accuracy = []
    key_frame = key.repeat(params.num_frames, 1).to(device)
    key = key.unsqueeze(0).to(device)
    
    for i, prompt in enumerate(prompts):
        for j, seed in enumerate(params.seed):
            generator = torch.manual_seed(seed)
            nw_frames = pipe(prompt = prompt, 
                        height = params.height,
                        width = params.width,
                        video_length = params.num_frames,
                        num_inference_steps = params.num_inference_steps,
                        generator = generator
                        ).frames[0]
            nw_frames = np.stack([np.array(img.convert("RGB")) for img in nw_frames]) / 255.0
            export_to_video(nw_frames, os.path.join(params.nw_saved_dir, f"{i * len(params.seed) + j}.mp4"))
            np.save(os.path.join('./frame_array/original/latte', f"{i * len(params.seed) + j}.npy"), nw_frames)
    
    pipe.vae = watermarked_vae_ms
    for i, prompt in enumerate(prompts):
        
        for j, seed in enumerate(params.seed):
            generator = torch.manual_seed(seed)
            w_frames = pipe(prompt = prompt, 
                        height = params.height,
                        width = params.width,
                        video_length = params.num_frames,
                        num_inference_steps = params.num_inference_steps,
                        generator = generator
                        ).frames[0]
            w_frames = np.stack([np.array(img.convert("RGB")) for img in w_frames]) / 255.0
            export_to_video(w_frames, os.path.join(params.w_saved_dir_ms, f"{i * len(params.seed) + j}.mp4"))
            np.save(os.path.join('./frame_array/vidsig/latte_ms', f"{i * len(params.seed) + j}.npy"), w_frames)
            w_frames = transform_video(w_frames).to(pipe.vae.dtype).to(device)  
 
            decoded = msg_decoder(w_frames)
            binarized = (decoded > 0).int()
            votes = binarized.sum(dim=0)  
            majority = (votes > (decoded.size(0) // 2)).int().reshape(key.shape)  
            diff = (~torch.logical_xor(decoded>0, key_frame>0)) # b k -> b k
            bit_accs = (torch.sum(diff, dim=-1) / diff.shape[-1]).mean().item() # b k -> b
            bit_accuracy.append(bit_accs)
            diff_video = (~torch.logical_xor(majority, key>0))
            bit_video_acc = torch.sum(diff_video).item() / diff_video.numel()
            video_bit_accuracy.append(bit_video_acc)
            print(f'bit acc: {bit_accs}')
            print(f'video bit acc: {bit_video_acc}')
    print(np.mean(bit_accuracy))
    print(np.mean(video_bit_accuracy))
    with open('output/rvm/latte/log.txt', 'a') as f:
        f.write(f'bit acc: {np.mean(bit_accuracy)}\n')
        f.write(f'video bit acc: {np.mean(video_bit_accuracy)}\n')
    f.close()

    pipe.vae = watermarked_vae_svd
    bit_accuracy = []
    video_bit_accuracy = []
    for i, prompt in enumerate(prompts):
        for j, seed in enumerate(params.seed):
            generator = torch.manual_seed(seed)
            w_frames = pipe(prompt = prompt, 
                        height = params.height,
                        width = params.width,
                        video_length = params.num_frames,
                        num_inference_steps = params.num_inference_steps,
                        generator = generator
                        ).frames[0]
            w_frames = np.stack([np.array(img.convert("RGB")) for img in w_frames]) / 255.0
            export_to_video(w_frames, os.path.join(params.w_saved_dir_svd, f"{i * len(params.seed) + j}.mp4"))
            np.save(os.path.join('./frame_array/vidsig/latte_svd', f"{i * len(params.seed) + j}.npy"), w_frames)
            w_frames = transform_video(w_frames).to(pipe.vae.dtype).to(device)  

            decoded = msg_decoder(w_frames)
            binarized = (decoded > 0).int()
            votes = binarized.sum(dim=0)  
            majority = (votes > (decoded.size(0) // 2)).int().reshape(key.shape)  
            diff = (~torch.logical_xor(decoded>0, key_frame>0)) # b k -> b k
            bit_accs = (torch.sum(diff, dim=-1) / diff.shape[-1]).mean().item() # b k -> b
            bit_accuracy.append(bit_accs)
            diff_video = (~torch.logical_xor(majority, key>0))  
            bit_video_acc = torch.sum(diff_video).item() / diff_video.numel()
            video_bit_accuracy.append(bit_video_acc)
            print(f'bit acc: {bit_accs}')
            print(f'video bit acc: {bit_video_acc}')
    print(np.mean(bit_accuracy))
    print(np.mean(video_bit_accuracy))
    with open('output/rvm/latte/log.txt', 'a') as f:
        f.write(f'bit acc: {np.mean(bit_accuracy)}\n')
        f.write(f'video bit acc: {np.mean(video_bit_accuracy)}\n')
    f.close()

if __name__ == "__main__":
    
    
    yaml_path = './yamls/latte.yml'
    params = get_params(yaml_path)
    main(params)