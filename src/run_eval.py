import os
import sys
sys.path.append(os.getcwd())

import torch
import lpips
import numpy as np
import cv2
import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from src.utils.log_utils import MetricLogger, OutputWriter
import argparse

class VideoQualityEvaluator():
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.lpips_metric = lpips.LPIPS(net='vgg').to(device)
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = torch.tensor(frame, device=self.device).permute(2,0,1).float()/255.0
            frames.append(tensor_frame)
        cap.release()
        video_tensor = torch.stack(frames)  # (T, C, H, W)
        return video_tensor

    def compute_psnr(self, video1, video2):
        return self.psnr_metric(video1, video2).item()

    def compute_ssim(self, video1, video2):
        return self.ssim_metric(video1, video2).item()

    def compute_lpips(self, video1, video2):
        video1_norm = 2 * video1 - 1
        video2_norm = 2 * video2 - 1
        lpips_values = [self.lpips_metric(v1.unsqueeze(0), v2.unsqueeze(0)).item() for v1, v2 in zip(video1_norm, video2_norm)]
        return np.mean(lpips_values)

    def compute_tlp(self, video1, video2):
        tlp_values = []
        video1_norm = 2 * video1 - 1
        video2_norm = 2 * video2 - 1
        for i in range(video1.shape[0]-1):
            lp1 = self.lpips_metric(video1_norm[i].unsqueeze(0), video1_norm[i+1].unsqueeze(0)).item()
            lp2 = self.lpips_metric(video2_norm[i].unsqueeze(0), video2_norm[i+1].unsqueeze(0)).item()
            tlp_values.append(abs(lp1 - lp2))
        return np.mean(tlp_values)


    def evaluate(self, video_path1, video_path2):
        video1 = self._load_video(video_path1)
        video2 = self._load_video(video_path2)

        min_frames = min(video1.shape[0], video2.shape[0])
        video1, video2 = video1[:min_frames], video2[:min_frames]

        metrics = {
            "PSNR": self.compute_psnr(video1, video2),
            "SSIM": self.compute_ssim(video1, video2),
            "LPIPS": self.compute_lpips(video1, video2),
            "tLP": self.compute_tlp(video1,video2),
        }

        return metrics

def read_videos(path: str):

    paths = []
    for current_path, _, files in os.walk(path):
        for filename in files:
            if filename.endswith(('.mp4', '.avi', '.mov')): 
                full_path = os.path.join(current_path, filename)
                paths.append(full_path)
    return sorted(paths)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--video_path', type=str, default='')
    parser.add_argument('--watermarked_video_path', type=str, default='')
    args = parser.parse_args()

    evaluator = VideoQualityEvaluator()
    os.makedirs(args.output_dir, exist_ok = True)
    writer = OutputWriter(os.path.join(args.output_dir, 'log.txt'))
    original_videos =  read_videos(args.video_path)
    watermarked_videos = read_videos(args.watermarked_video_path)

    print("Evaluating video_signature:")
    metriclogger = MetricLogger(delimiter = '\t', window_size = 200)
    for video1, video2 in tqdm.tqdm(zip(original_videos, watermarked_videos), total = len(original_videos)):
        try:
            result = evaluator.evaluate(video1, video2)
            for key, item in result.items():
                metriclogger.update(**{key: item})
        except:
            pass

    log_stats = {k: meter.global_avg for k, meter in metriclogger.meters.items()}    
    writer.write_dict(log_stats)
     

if __name__ == "__main__":

    import warnings
    warnings.filterwarnings('ignore')
    main()