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
from diffusers import StableVideoDiffusionPipeline, I2VGenXLPipeline, TextToVideoSDPipeline
from diffusers.utils import load_image, export_to_video
import torchvision
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                                std=[0.229, 0.224, 0.225])

def read_json(file_path: str):

    "read captions of MSCOCO dataset"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data

def get_prompt(file_path: str, num_eval: int):
    """
    read prompt from caption
    """
    data = read_json(file_path)
    annotations = data['annotations']
    prompts = []
    image_id = {}
    for annotation in annotations:
        if annotation['image_id'] not in image_id:
            image_id[annotation['image_id']] = 1
            prompts.append(annotation['caption'])
    
    assert num_eval <= len(prompts) 
    prompts = random.sample(prompts, num_eval)
    with open('prompt.txt', 'w') as f:
        for prompt in prompts:
            f.write(prompt)
    f.close()
    return prompts


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

    pipe = TextToVideoSDPipeline.from_pretrained(params.model_id, torch_dtype = torch.float16).to(device)
    watermarked_vae = copy.deepcopy(pipe.vae)
    watermarked_vae.load_state_dict(torch.load(params.ckpt_path, map_location = 'cpu'))
    watermarked_vae.to(torch.float16)

    if not os.path.exists(params.nw_saved_dir):
        os.makedirs(params.nw_saved_dir)
    
    if not os.path.exists(params.w_saved_dir):
        os.makedirs(params.w_saved_dir)

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
                        num_frames = params.num_frames,
                        num_inference_steps = params.num_inference_steps,
                        generator = generator
                        ).frames[0]
            export_to_video(nw_frames, os.path.join(params.nw_saved_dir, 'videos', f"{i * len(params.seed) + j}.mp4"))
            np.save(os.path.join(params.nw_saved_dir, 'frames', f"{i * len(params.seed) + j}.npy"), nw_frames)

    pipe.vae = watermarked_vae
    detect_time = []
    for i, prompt in enumerate(prompts):
        
        for j, seed in enumerate(params.seed):
            generator = torch.manual_seed(seed)
            w_frames = pipe(prompt = prompt, 
                        height = params.height,
                        width = params.width,
                        num_frames = params.num_frames,
                        num_inference_steps = params.num_inference_steps,
                        generator = generator
                        ).frames[0]
            export_to_video(w_frames, os.path.join(params.w_saved_dir, 'videos', f"{i * len(params.seed) + j}.mp4"))
            np.save(os.path.join(params.w_saved_dir, 'frames', f"{i * len(params.seed) + j}.npy"), w_frames)
            
            start_time = time.time()
            decoded = msg_decoder(w_frames)
            binarized = (decoded > 0).int()
            votes = binarized.sum(dim=0)  
            majority = (votes > (decoded.size(0) // 2)).int().reshape(key.shape)  
            end_time = time.time()
            detect_time.append(end_time - start_time)
            diff = (~torch.logical_xor(decoded>0, key_frame>0)) # b k -> b k
            bit_accs = (torch.sum(diff, dim=-1) / diff.shape[-1]).mean().item() # b k -> b
            bit_accuracy.append(bit_accs)
            diff_video = (~torch.logical_xor(majority, key>0))
            bit_video_acc = torch.sum(diff_video).item() / diff_video.numel()
            video_bit_accuracy.append(bit_video_acc)

    with open(os.path.join(params.output_dir, 'log.txt'), 'a') as f:
        f.write(f'bit acc: {np.mean(bit_accuracy)}\n')
        f.write(f'video bit acc: {np.mean(video_bit_accuracy)}\n')
        f.write(f'average detect time: {np.mean(detect_time)}\n')
    
if __name__ == "__main__":
    
    import warnings
    warnings.filterwarnings("ignore")
    
    yaml_path = 'yamls/generate.yml'
    params = get_params(yaml_path)
    main(params)