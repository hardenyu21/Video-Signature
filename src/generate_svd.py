import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.path.append(os.getcwd())
from utils.param_utils import seed_all, get_params
from utils.data_utils import read_metadata
from torchvision.datasets.folder import is_image_file
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

from diffusers import StableVideoDiffusionPipeline, I2VGenXLPipeline, TextToVideoSDPipeline
from diffusers.utils import load_image, export_to_video
import time

def get_images_paths(path:str):
    """
    get the image paths
    """
    paths = []
    for current_path, _, files in os.walk(path):
        for filename in files:
            paths.append(os.path.join(current_path, filename))
    
    paths = sorted([fn for fn in paths if is_image_file(fn)])

    return paths

def transform_video(video):
    transform = data_utils.default_transform()
    video_tensor = torch.stack([transform(frame) for frame in video])
    return video_tensor

def main(params):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #prompts = get_prompt(file_path = params.file_path, num_eval = params.num_eval)
    image_paths = get_images_paths(params.image_path)
    ##load message decoder
    try:
        msg_decoder = torch.jit.load(params.msg_decoder_path)
    except:
        raise KeyError(f"No checkpoint found in {params.msg_decoder_path}")
    for msg_decoder_params in msg_decoder.parameters():
        msg_decoder_params.requires_grad = False
    msg_decoder.eval()
    
    key = data_utils.list_to_torch(data_utils.str_to_list(params.key))

    pipe = StableVideoDiffusionPipeline.from_pretrained(params.model_id, torch_dtype = torch.float16).to(device)
    watermarked_vae = copy.deepcopy(pipe.vae)
    watermarked_vae.load_state_dict(torch.load(params.ckpt_path, map_location = 'cpu'))
    watermarked_vae.to(device)
    watermarked_vae.to(torch.float16)

    if not os.path.exists(params.nw_saved_dir):
        os.makedirs(params.nw_saved_dir)
    
    if not os.path.exists(params.w_saved_dir):
        os.makedirs(params.w_saved_dir)

    bit_accuracy = []
    video_bit_accuracy = []
    key_frame = key.repeat(params.num_frames, 1).to(device)
    key = key.unsqueeze(0).to(device)
    decode_time = []
    

    for i, image_path in enumerate(image_paths):
        image = load_image(image_path)
        for j, seed in enumerate(params.seed):
            generator = torch.manual_seed(seed)
            nw_frames = pipe(image = image, 
                        height = params.height,
                        width = params.width,
                        num_frames = params.num_frames,
                        num_inference_steps = params.num_inference_steps,
                        generator = generator
                        ).frames[0]
            export_to_video(nw_frames, os.path.join(params.nw_saved_dir, 'videos', f"{i * len(params.seed) + j}.mp4"))
            np.save(os.path.join(params.nw_saved_dir, 'frames', f"{i * len(params.seed) + j}.npy"), nw_frames)
    
    
    pipe.vae = watermarked_vae
    for i, image_path in enumerate(image_paths):
        image = load_image(image_path)
        for j, seed in enumerate(params.seed):
            generator = torch.manual_seed(seed)
            w_frames = pipe(image = image, 
                        height = params.height,
                        width = params.width,
                        num_frames = params.num_frames,
                        num_inference_steps = params.num_inference_steps,
                        generator = generator
                        ).frames[0]
            np.save(os.path.join(params.w_saved_dir, 'frames', f"{i * len(params.seed) + j}.npy"), w_frames)
            
            export_to_video(w_frames, os.path.join(params.w_saved_dir, 'videos', f"{i * len(params.seed) + j}.mp4"))
            w_frames = transform_video(w_frames).to(pipe.vae.dtype).to(device)  
            
            start_time = time.time()
            decoded = msg_decoder(w_frames)
            binarized = (decoded > 0).int()
            votes = binarized.sum(dim=0)  
            majority = (votes > (decoded.size(0) // 2)).int().reshape(key.shape)  
            decode_time.append(time.time() - start_time)
            diff = (~torch.logical_xor(decoded>0, key_frame>0)) # b k -> b k
            bit_accs = (torch.sum(diff, dim=-1) / diff.shape[-1]).mean().item() # b k -> b
            bit_accuracy.append(bit_accs)
            diff_video = (~torch.logical_xor(majority, key>0))
            bit_video_acc = torch.sum(diff_video).item() / diff_video.numel()
            video_bit_accuracy.append(bit_video_acc)

    with open(os.path.join(params.output_dir, 'log.txt'), 'a') as f:
        f.write(f'bit acc: {np.mean(bit_accuracy)}\n')
        f.write(f'video bit acc: {np.mean(video_bit_accuracy)}\n')
        f.write(f'average decode time: {np.mean(decode_time)}\n')
    f.close()
    
if __name__ == "__main__":
    
    import warnings
    warnings.filterwarnings("ignore")
    yaml_path = 'yamls/generate.yml'
    params = get_params(yaml_path)
    main(params)