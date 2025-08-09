import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from diffusers import AutoencoderKLTemporalDecoder, AutoencoderKL
from torchvision.io import read_video
import os
import numpy as np
import cv2
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #vae = AutoencoderKLTemporalDecoder.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", subfolder = "vae").to(device)
    vae = AutoencoderKL.from_pretrained("damo-vilab/text-to-video-ms-1.7b", subfolder = "vae").to(device)
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()
    sensitivity_list = {}
    transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(0.5, 0.5)])
    num_iterations = 10
    for k in range(num_iterations):
        print(f"Running Pertubation-Aware Layer Filtering for iteration {k + 1}/{num_iterations}")
        for i, video in enumerate(os.listdir('PAS_data')):
            print(f"Running Pertubation-Aware Layer Filtering for data {i + 1}/{len(os.listdir('PAS_data'))}")
            video_path = os.path.join('PAS_data', video)
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = transform(frame)
                frames.append(frame)
            cap.release()
            video_tensor = torch.stack(frames).to(device)  # (T, C, H, W)
            num_frames = video_tensor.shape[0]

            latent = vae.encode(video_tensor).latent_dist.mode()
            output = vae.decode(latent).sample
            model = copy.deepcopy(vae).eval()

            for name, param in tqdm(model.decoder.named_parameters(), total=len(list(model.decoder.named_parameters()))):
                if name not in sensitivity_list:
                    sensitivity_list[name] = []
                original_param = param.clone()
                param.data += torch.randn_like(param) * 1e-2
                output_perturbed = model.decode(latent, num_frames).sample
                sensitivity = ((output_perturbed - output) ** 2).mean().item()
                sensitivity_list[name].append(sensitivity)
                param.data = original_param 
    for name, sensitivities in sensitivity_list.items():
        sensitivities = np.array(sensitivities)
        sensitivities = sensitivities.mean(axis=0)
        sensitivity_list[name] = sensitivities
    with open(f'sensitivity_list_2d.txt', 'w') as f:
        for name, sensitivities in sensitivity_list.items():
            f.write(f"{name}: {sensitivities}\n")
    f.close()
