# Video-Signature
This repository contains the implementation for the paper "Video Signature: In-generation Watermarking for Latent Video Diffusion Models" (https://arxiv.org/pdf/2506.00652)

### Install Dependencies
Basic Dependencies:
* Python >= 3.10
Then run:
```shell
pip install -r requirements.txt
```
Download the checkpoint we have fine-tuned at [Google Drive](https://drive.google.com/file/d/1XFyzeX6T0iHgcxSN_DxvjLFy1EXZye-Q/view?usp=drive_link), and put them into the root directory.

### Watermark Embedding and Extraction
```shell
python src/generate_ms.py 
```
You can check ./yamls/generate_ms.yml for detailed arguments, there will be three directories to store the output, one is for the original videos, one is for the watermarked videos and one is for the corresponding frame arrays in .npy

### Video Quality Evaluation Via Traditional Metrics
```shell
python src/run_eval.py --output_dir <output dir to log the results> \ 
                       --video_path  <the path to store the original videos> \
                       --watermarked_video_path <the path to store the watermarked videos>
```
---

### Video Quality Evaluation Via Vbench

Using [VBench](https://github.com/Vchitect/VBench) — VBench: Comprehensive Benchmark Suite for Video Generative Models

### Attack
```shell
python src/attack.py --output_dir <output dir to log the results> \ 
                     --attack_type  clean \
                     --factor 2.0 \
                     --frame_array_path <the path to store the frame arrays>
```

### Finetune by yourself
If you want to finetune by yourself, you can download the dataset here [OpenVid](https://github.com/NJU-PCALab/OpenVid-1M), the full dataset is too large, so we recommend you to download a small set of it. For detail of the file organization, you can check './yamls/finetune.yml'

Once you download the dataset, you can finetune your own watermarked model by
'''shell
python src/finetune.py
'''

## Citation
```bibtex
@article{huang2025video,
  title={Video Signature: In-generation Watermarking for Latent Video Diffusion Models},
  author={Huang, Yu and Chen, Junhao and Zheng, Qi and Li, Hanqian and Liu, Shuliang and Hu, Xuming},
  journal={arXiv preprint arXiv:2506.00652},
  year={2025}
}
```