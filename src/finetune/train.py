import os
import sys

import wandb.integration
sys.path.append(os.getcwd())
import random
import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
import torch.nn.functional as F
from copy import deepcopy
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import AutoencoderKLTemporalDecoder, AutoencoderKL
from omegaconf import OmegaConf
from src.utils.param_utils import  get_params, parse_optim_params, adjust_learning_rate, seed_all
from src.utils import data_utils
from src.loss.loss_provider import LossProvider
from src.utils.log_utils import MetricLogger, OutputWriter
import torch
import torch.nn as nn
#from peft import LoraConfig, get_peft_model


class Trainer():

    """

    This class is responsible for handling the entire fine-tuning workflow, including:
    - Loading the pre-trained VAE model.
    - Generating and managing watermark keys.
    - Initializing and freezing parts of the model as required.
    - Preparing datasets and dataloaders for training and validation.
    - Configuring optimizers and learning rate schedulers.
    - Defining and applying custom loss functions for training.
    - Logging training and validation statistics, including saving fine-tuned models.

    Methods:
        - `_generate_key`: Generates random watermark keys based on the specified number of keys and bit length.
        - `_seed_all`: Sets the random seed for reproducibility.
        - `_build_finetuned_vae`: Creates fine-tuned VAE instances, ensuring only the decoder is trainable.
        - `_freeze`: Freezes the pre-trained VAE to prevent updates during training.
        - `_load_msg_decoder`: Loads the message decoder for watermark extraction.
        - `_get_dataloader`: Prepares training and validation dataloaders from the specified dataset.
        - `_build_optimizer`: Builds optimizers for training, configured for fine-tuning specific VAE decoders.
        - `_loss_fn`: Configures the loss functions for image and watermark reconstruction.
        - `_train_per_key`: Handles the training process for a single watermark key.
        - `_evaluate`: Evaluates the fine-tuned VAE decoder using various robustness tests.
        - `_save`: Saves the fine-tuned model, training statistics, and watermark keys.
        - `train`: Orchestrates the training process for all watermark keys.

    Attributes:
        - `vae` (AutoencoderKL): The pre-trained VAE model used as a base for fine-tuning.
        - `finetuned_vaes` (list[AutoencoderKL]): A list of fine-tuned VAE models, one for each watermark key.
        - `params` (OmegaConf): Configuration parameters for training.
        - `output_dir` (str): Directory for saving logs and outputs.
        - `model_dir` (str): Directory for saving fine-tuned model checkpoints.
        - `watermark_key` (list[list[int]]): A list of binary watermark keys.
        - `msg_decoder` (nn.Module): The message decoder model for extracting watermarks.
        - `train_loader`, `val_loader` (DataLoader): Data loaders for training and validation datasets.
        - `optimizers` (list[torch.optim.Optimizer]): A list of optimizers for each fine-tuned VAE decoder.
        - `loss_w`, `loss_i`: Custom loss functions for watermark and image reconstruction, respectively.
        - `device` (str): The device (CPU or GPU) used for computation.
    """


    def __init__(self, params: OmegaConf) -> None:

        self.params = params

        self.output_dir = os.path.join(os.getcwd(), params.output_dir)
        self.model_dir = os.path.join(os.getcwd(), params.model_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.log_file = os.path.join(self.output_dir, params.log_file)
        self.writer = OutputWriter(self.log_file)
        #set seed
        seed_all(params.seed)
        #generate key
        self.watermark_key = self._generate_key(params.num_bits)
        #Initialize vae decoder and msg_decode
        
        if params.model_id == 'damo-vilab/text-to-video-ms-1.7b':
            self.vae = AutoencoderKL.from_pretrained(params.model_id, subfolder = "vae") 
        elif params.model_id == 'stabilityai/stable-video-diffusion-img2vid-xt':
            self.vae = AutoencoderKLTemporalDecoder.from_pretrained(params.model_id, subfolder = "vae") 
        else:
            raise ValueError(f"Model {params.model_id} not supported")
        #except:
        #self.vae = AutoencoderKLTemporalDecoder.from_pretrained(params.model_id, subfolder = "vae") 
        self.finetuned_vae = self._build_finetuned_vae(params)   ##requires_grad = True
        self._freeze()
        self.msg_decoder = self._load_msg_decoder(params)
        self._to(self.device)
        
        #self.temporal_extractor = self._load_temporal_extractor()
        
        #dataset
        self.train_loader, self.val_loader = self._get_dataloader(params)
        
        ##optimizer and learning scheduler
        self.optimizer = self._build_optimizer(params)
                                   
        ##loss function
        self.loss_w, self.loss_i, self.loss_t = self._loss_fn(params)

        wandb.init(project = 'video watermarking', name = "rvm_modelscope")
    
    def _generate_key(self, num_bits: int) -> list[list]:

        """
        generate watermark key
        """        
        key = random.choices([0, 1], k = num_bits)
        print(f'generated watermark key: {data_utils.list_to_str(key)}')
        return key
    
    def _seed_all(self, seed: int) -> None:
        "Set the random seed for reproducibility"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    def _build_finetuned_vae(self, params):

        vae = deepcopy(self.vae)
        for vae_params in vae.parameters():
            vae_params.requires_grad = False

        for decoder_params in vae.decoder.parameters():
            decoder_params.requires_grad = True

        for name, module in vae.decoder.named_modules():
            if name in params.target_layers:
                for param in module.parameters():
                    param.requires_grad = False
        
        print(f'Total parameters for VAE Decoder: {sum(p.numel() for p in vae.decoder.parameters())}')
        print(f'Trainable parameters: {sum(p.numel() for p in vae.decoder.parameters() if p.requires_grad)}')
        return vae  
    
    def _freeze(self):

        for model_params in self.vae.parameters():
            model_params.requires_grad = False
        self.vae.eval()

    def _load_msg_decoder(self, params) -> nn.Module:

        """load msg decoder from checkpoint"""
        ckpt_path = params.msg_decoder_path
        try:
            msg_decoder = torch.jit.load(ckpt_path)
        except:
            raise KeyError(f"No checkpoint found in {ckpt_path}")
        for msg_decoder_params in msg_decoder.parameters():
            msg_decoder_params.requires_grad = False
        msg_decoder.eval()
        return msg_decoder


    def _get_dataloader(self, params) -> Tuple[DataLoader, DataLoader]:

        transform = data_utils.vqgan_transform(params.img_size)
        train_loader = data_utils.get_video_dataloader(params.metadata_path, params.train_dir, params.num_frames,
                                                       params.frame_interval, transform = transform, 
                                                       num_videos = params.num_train, batch_size = params.batch_size,
                                                       shuffle = True, collate_fn = None)
        val_loader = data_utils.get_video_dataloader(params.metadata_path, params.train_dir, params.num_frames,
                                                       params.frame_interval, transform = transform, 
                                                       num_videos = params.num_val, batch_size = params.batch_size,
                                                       shuffle = False, collate_fn = None)
        
        return train_loader, val_loader

    def _build_optimizer(self, params) -> torch.optim.Optimizer:
        
        optimizer = params.optimizer
        optim_params = parse_optim_params(params)

        torch_optimizers = sorted(name for name in torch.optim.__dict__
            if name[0].isupper() and not name.startswith("__")
            and callable(torch.optim.__dict__[name]))
        if hasattr(torch.optim, optimizer):
            return getattr(torch.optim, optimizer)(self.finetuned_vae.parameters(), **optim_params)
        raise ValueError(f'Unknown optimizer "{optimizer}", choose among {str(torch_optimizers)}')
    
    def _loss_fn(self, params):
        
        """
        get loss function, the loss function and weights copy from https://github.com/SteffenCzolbe/PerceptualSimilarity
        """

        print(f'>>> Creating losses')
        print(f'Losses: {params.loss_w} and {params.loss_i} and {params.loss_t}')
        if params.loss_w == 'mse':        
            loss_w = lambda decoded, keys, temp = 10.0: torch.mean((decoded * temp - (2 * keys- 1))**2) # b k - b k
        elif params.loss_w == 'bce':
            loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded * temp, keys, reduction = 'mean')
        else:
            raise NotImplementedError
    
        if params.loss_i == 'mse':
            loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)
        elif params.loss_i == 'watson-dft':
            provider = LossProvider()
            loss_percep = provider.get_loss_function('Watson-DFT', colorspace = 'RGB', pretrained = True, reduction = 'sum')
            loss_percep = loss_percep.to(self.device)
            loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
        elif params.loss_i == 'watson-vgg':
            provider = LossProvider()
            loss_percep = provider.get_loss_function('Watson-VGG', colorspace = 'RGB', pretrained = True, reduction = 'sum')
            loss_percep = loss_percep.to(self.device)
            loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
        elif params.loss_i == 'ssim':
            provider = LossProvider()
            loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
            loss_percep = loss_percep.to(self.device)
            loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
        else:
            raise NotImplementedError
        
        if params.loss_t == 'mse':

            loss_t = lambda t_w, t_nw: torch.mean((t_w - t_nw)**2)
        elif params.loss_t == 'mae':
            loss_t = lambda t_w, t_nw: torch.mean(torch.abs(t_w - t_nw))
        
        elif params.loss_t == 'watson-dft':
            provider = LossProvider()
            loss_percep = provider.get_loss_function('Watson-DFT', colorspace = 'RGB', pretrained = True, reduction = 'sum')
            loss_percep = loss_percep.to(self.device)
            loss_t = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
        elif params.loss_t == 'watson-vgg':
            provider = LossProvider()
            loss_percep = provider.get_loss_function('Watson-VGG', colorspace = 'RGB', pretrained = True, reduction = 'sum')
            loss_percep = loss_percep.to(self.device)
            loss_t = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
        elif params.loss_t == 'ssim':
            provider = LossProvider()
            loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
            loss_percep = loss_percep.to(self.device)
            loss_t = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
        else:
            raise NotImplementedError
        
        return loss_w, loss_i, loss_t
    
    def _to(self, device: str):

        self.vae.to(device)
        self.finetuned_vae.to(device)
        self.msg_decoder.to(device)
        
    def _train(
                self,
                params,
                vqgan_to_imnet: transforms.Compose) -> dict:

        """
        fine_tune vae decoder for one watermark key
        """
        key = data_utils.list_to_torch(self.watermark_key)
        #header = 'Train'
        metric_logger = MetricLogger(delimiter = '\n', window_size = params.log_freq)
        self.finetuned_vae.train()
        total_steps = len(self.train_loader)
        for step, frames in tqdm.tqdm(enumerate(self.train_loader), total = len(self.train_loader), desc = 'finetuning'):
            frames = frames.to(self.device)
            adjust_learning_rate(self.optimizer, step, total_steps, int(total_steps * 0.2), params.lr)
            bs, num_frames, c, h, w = frames.shape          

            imgs = frames.view(bs * num_frames, c, h, w)

            latents = self.vae.encode(imgs).latent_dist.mode() # b c h w -> b z h/f w/f
            if self.vae.__class__ == AutoencoderKLTemporalDecoder:  
                imgs_nw = self.vae.decode(latents, num_frames).sample
                imgs_w = self.finetuned_vae.decode(latents, num_frames).sample
            else:
                imgs_nw = self.vae.decode(latents).sample
                imgs_w = self.finetuned_vae.decode(latents).sample # b z h/f w/f -> b c h w

            diff_w = (imgs_w.view(bs, num_frames, c, h, w)[:, 1:] - imgs_w.view(bs, num_frames, c, h, w)[:, :-1]).view(bs * (num_frames - 1), c, h, w)
            diff_nw = (imgs_nw.view(bs, num_frames, c, h, w)[:, 1:] - imgs_nw.view(bs, num_frames, c, h, w)[:, :-1]).view(bs * (num_frames - 1), c, h, w)

            keys1 = key.repeat(bs * num_frames, 1).to(self.device) 
            decoded1 = self.msg_decoder(vqgan_to_imnet(imgs_w)) 
            decoded2 = decoded1.view(bs, num_frames, -1).mean(dim = 1, keepdim = True).view(bs, -1)
            keys2 = key.repeat(bs, 1).to(self.device)
            # compute loss
            lossw = params.alpha1 * self.loss_w(decoded1, keys1) + params.alpha2 * self.loss_w(decoded2, keys2)
            lossi = self.loss_i(imgs_w, imgs_nw)
            losst = self.loss_t(diff_w, diff_nw)
            loss = params.lambda_w * lossw + params.lambda_i * lossi + params.lambda_t * losst
            # optim step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # log stats
            diff = (~torch.logical_xor(decoded2>0, keys2>0)) # b k -> b k
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            #word_accs = (bit_accs == 1) # b
            train_log_stats = {
                "loss": loss.item(),
                "loss_w": lossw.item(),
                "loss_i": lossi.item(),
                "loss_t": losst.item(),
                "bit_acc_avg": torch.mean(bit_accs).item(),
                }
            for name, meter in train_log_stats.items():
                metric_logger.update(**{name: meter})
            wandb.log(train_log_stats)
        
        print(f"Final train stats: the value in () indicates the averaged stats")
        print(metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}        

    @torch.no_grad()
    def _evaluate(
                self,
                vqgan_to_imnet: transforms
                ) -> dict:
        metric_logger = MetricLogger(delimiter = "\n", fmt = "{global_avg:4f}")
        key = data_utils.list_to_torch(self.watermark_key)
        self.finetuned_vae.eval()
        for frames in tqdm.tqdm(self.val_loader):

            frames = frames.to(self.device)
            bs, num_frames, c, h, w = frames.shape          
            keys = key.repeat(bs * num_frames, 1).to(self.device)  #(b, k)

            imgs = frames.view(bs * num_frames, c, h, w)

            latents = self.vae.encode(imgs).latent_dist.mode()
            if self.vae.__class__ == AutoencoderKLTemporalDecoder:  
                imgs_nw = self.vae.decode(latents, num_frames).sample
                imgs_w = self.finetuned_vae.decode(latents, num_frames).sample
            else:
                imgs_nw = self.vae.decode(latents).sample
                imgs_w = self.finetuned_vae.decode(latents).sample # b z h/f w/f -> b c h w
            
            val_log_stats = {
                #"iteration": step + 1,
                "psnr": data_utils.psnr(imgs_w, imgs_nw).mean().item(),
                "psnr_ori": data_utils.psnr(imgs_w, imgs).mean().item(),
            }
            attacks = {
                'none': lambda x: x,
                'crop_01': lambda x: data_utils.center_crop(x, 0.1),
                'crop_05': lambda x: data_utils.center_crop(x, 0.5),
                'rotate_30': lambda x: data_utils.rotate(x, 30), 
                'resize_07': lambda x: data_utils.resize(x, 0.7),
                'brightness_1p5': lambda x: data_utils.adjust_brightness(x, 1.5),
                'brightness_2': lambda x: data_utils.adjust_brightness(x, 2),
            }
            for name, attack in attacks.items():
                imgs_aug = attack(vqgan_to_imnet(imgs_w))
                decoded = self.msg_decoder(imgs_aug) # b c h w -> b k
                diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
                bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
                #word_accs = (bit_accs == 1) # b
                val_log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
                #val_log_stats[f'word_acc_{name}'] = torch.mean(word_accs.type(torch.float)).item()
            wandb.log(val_log_stats)
            for name, meter in val_log_stats.items():
                metric_logger.update(**{name: meter})
            
        print("Averaged eval stats")
        print(metric_logger)
            
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            
        
    def _save(self, **kargs) -> None:

        model_path = os.path.join(self.model_dir, f"checkpoint.pth")
        metadata = {'key': data_utils.list_to_str(self.watermark_key),
                    'saved_path': model_path,
                    'train_stats': {**{f'train_{k}': v for k, v in kargs['train_stats'].items()}},
                    'val_stats': {**{f'val_{k}': v for k, v in kargs['val_stats'].items()}},
                    }
        log_metadata = {f'Finetuned vae': metadata}
        torch.save(self.finetuned_vae.state_dict(), model_path)
        self.writer.write_dict(log_metadata)
    
    def fit(self) -> None:
        
        train_stats =  self._train(self.params, data_utils.vqgan_to_imnet())
        val_stats = self._evaluate(data_utils.vqgan_to_imnet())
        self._save(train_stats = train_stats, val_stats = val_stats)

    @property
    def device(self):

        return "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    params_path = "src/utils/yamls/finetune/rvm.yaml"

    trainer = Trainer(params = get_params(params_path))
    trainer.fit()