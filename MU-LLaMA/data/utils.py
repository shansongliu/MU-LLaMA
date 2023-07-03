import logging
import math

import torch
import torch.nn as nn
import torchaudio
from PIL import Image
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

from .models.multimodal_preprocessors import SimpleTokenizer

BPE_PATH = "data/bpe/bpe_simple_vocab_16e6.txt.gz"

def load_and_transform_audio_data(
    audio_paths,
    sample_rate=24000
):
    audios = []
    for path in audio_paths:
        waveform, sr = torchaudio.load(path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
        waveform = torch.mean(waveform, 0)
        audios.append(waveform)
    return torch.stack(audios, dim=0)

def load_and_transform_text(text, device):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens