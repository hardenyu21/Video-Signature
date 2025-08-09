import torch
import torch.nn as nn
import os
from collections import OrderedDict

from src.loss.color_wrapper import ColorWrapper, GreyscaleWrapper
from src.loss.shift_wrapper import ShiftWrapper
from src.loss.watson import WatsonDistance
from src.loss.watson_fft import WatsonDistanceFft
from src.loss.watson_vgg import WatsonDistanceVgg
from src.loss.deep_loss import PNetLin
from src.loss.ssim import SSIM


class LossProvider():
    def __init__(self):
        self.loss_functions = ['L1', 'L2', 'SSIM', 'Watson-dct', 'Watson-fft', 'Watson-vgg', 'Deeploss-vgg', 'Deeploss-squeeze', 'Adaptive']
        self.color_models = ['LA', 'RGB']
        work_dir = os.getcwd()
        weights_path = os.path.join(work_dir, 'ckpts/loss')
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        self.weights_path = weights_path

    def load_state_dict(self, filename):
        path = os.path.join(self.weights_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist, you may download them from https://github.com/SteffenCzolbe/PerceptualSimilarity first.")
        return torch.load(path, map_location='cpu')
    
    def get_loss_function(self, model, colorspace='RGB', reduction='sum', deterministic=False, pretrained=True, image_size=None):
        """
        returns a trained loss class.
        model: one of the values returned by self.loss_functions
        colorspace: 'LA' or 'RGB'
        deterministic: bool, if false (default) uses shifting of image blocks for watson-fft
        image_size: tuple, size of input images. Only required for adaptive loss. Eg: [3, 64, 64]
        """
        is_greyscale = colorspace in ['grey', 'Grey', 'LA', 'greyscale', 'grey-scale']
        
        if model.lower() in ['l2']:
            loss = nn.MSELoss(reduction=reduction)
        elif model.lower() in ['l1']:
            loss = nn.L1Loss(reduction=reduction)
        elif model.lower() in ['ssim']:
            loss = SSIM(size_average=(reduction in ['sum', 'mean']))
        elif model.lower() in ['watson', 'watson-dct']:
            if is_greyscale:
                if deterministic:
                    loss = WatsonDistance(reduction=reduction)
                    if pretrained: 
                        loss.load_state_dict(self.load_state_dict('gray_watson_dct_trial0.pth'))
                else:
                    loss = ShiftWrapper(WatsonDistance, (), {'reduction': reduction})
                    if pretrained: 
                        loss.loss.load_state_dict(self.load_state_dict('gray_watson_dct_trial0.pth'))
            else:
                if deterministic:
                    loss = ColorWrapper(WatsonDistance, (), {'reduction': reduction})
                    if pretrained: 
                        loss.load_state_dict(self.load_state_dict('rgb_watson_dct_trial0.pth'))
                else:
                    loss = ShiftWrapper(ColorWrapper, (WatsonDistance, (), {'reduction': reduction}), {})
                    if pretrained: 
                        loss.loss.load_state_dict(self.load_state_dict('rgb_watson_dct_trial0.pth'))
        elif model.lower() in ['watson-fft', 'watson-dft']:
            if is_greyscale:
                if deterministic:
                    loss = WatsonDistanceFft(reduction=reduction)
                    if pretrained: 
                        loss.load_state_dict(self.load_state_dict('gray_watson_fft_trial0.pth'))
                else:
                    loss = ShiftWrapper(WatsonDistanceFft, (), {'reduction': reduction})
                    if pretrained: 
                        loss.loss.load_state_dict(self.load_state_dict('gray_watson_fft_trial0.pth'))
            else:
                if deterministic:
                    loss = ColorWrapper(WatsonDistanceFft, (), {'reduction': reduction})
                    if pretrained: 
                        loss.load_state_dict(self.load_state_dict('rgb_watson_fft_trial0.pth'))
                else:
                    loss = ShiftWrapper(ColorWrapper, (WatsonDistanceFft, (), {'reduction': reduction}), {})
                    if pretrained: 
                        loss.loss.load_state_dict(self.load_state_dict('rgb_watson_fft_trial0.pth'))
        elif model.lower() in ['watson-vgg', 'watson-deep']:
            if is_greyscale:
                loss = GreyscaleWrapper(WatsonDistanceVgg, (), {'reduction': reduction})
                if pretrained: 
                    loss.loss.load_state_dict(self.load_state_dict('gray_watson_vgg_trial0.pth'))
            else:
                loss = WatsonDistanceVgg(reduction=reduction)
                if pretrained: 
                    loss.load_state_dict(self.load_state_dict('rgb_watson_vgg_trial0.pth'))
        elif model.lower() in ['deeploss-vgg']:
            if is_greyscale:
                loss = GreyscaleWrapper(PNetLin, (), {'pnet_type': 'vgg', 'reduction': reduction, 'use_dropout': False})
                if pretrained: 
                    loss.loss.load_state_dict(self.load_state_dict('gray_pnet_lin_vgg_trial0.pth'))
            else:
                loss = PNetLin(pnet_type='vgg', reduction=reduction, use_dropout=False)
                if pretrained: 
                    loss.load_state_dict(self.load_state_dict('rgb_pnet_lin_vgg_trial0.pth'))
        elif model.lower() in ['deeploss-squeeze']:
            if is_greyscale:
                loss = GreyscaleWrapper(PNetLin, (), {'pnet_type': 'squeeze', 'reduction': reduction, 'use_dropout': False})
                if pretrained: 
                    loss.loss.load_state_dict(self.load_state_dict('gray_pnet_lin_squeeze_trial0.pth'))
            else:
                loss = PNetLin(pnet_type='squeeze', reduction=reduction, use_dropout=False)
                if pretrained: 
                    loss.load_state_dict(self.load_state_dict('rgb_pnet_lin_squeeze_trial0.pth'))
        else:
            raise Exception('Metric "{}" not implemented'.format(model))

        # freeze all training of the loss functions
        if pretrained: 
            for param in loss.parameters():
                param.requires_grad = False
        
        return loss
