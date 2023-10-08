import argparse
import torch.cuda

import llama
from util.misc import *
from data.utils import load_and_transform_audio_data

from mutagen.mp3 import MP3
from mutagen.wave import WAVE

from pydub import AudioSegment

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="./ckpts/checkpoint.pth", type=str,
    help="Name of or path to the trained checkpoint",
)
parser.add_argument(
    "--llama_type", default="7B", type=str,
    help="Type of llama original weight",
)
parser.add_argument(
    "--llama_dir", default="/path/to/llama", type=str,
    help="Path to LLaMA pretrained checkpoint",
)
parser.add_argument(
    "--mert_path", default="m-a-p/MERT-v1-330M", type=str,
    help="Path to MERT pretrained checkpoint",
)
parser.add_argument(
    "--knn_dir", default="./ckpts", type=str,
    help="Path to directory with KNN Index",
)
parser.add_argument(
    "--audio_path", required=True, type=str,
    help="Path to the input music file",
)
parser.add_argument(
    "--question", default="Describe the Audio", type=str,
    help="Question to ask the model",
)

args = parser.parse_args()
model = llama.load(args.model, args.llama_dir, mert_path=args.mert_path, knn=True, knn_dir=args.knn_dir,
                   llama_type=args.llama_type)
model.eval()


def split_audio(audio, i, format):
    if len(audio) > 60000:
        audio[:60000].export(f"temp_" + str(i) + f".{format}", format=format)
        return [f"temp_" + str(i) + f".{format}"] + split_audio(audio[60000:], i + 1, format)
    if len(audio) > 10000:
        audio.export(f"temp_" + str(i) + f".{format}", format=format)
        return [f"temp_" + str(i) + f".{format}"]
    return []


def multimodal_generate(
        audio_path,
        audio_weight,
        prompt,
        cache_size,
        cache_t,
        cache_weight,
        max_gen_len,
        gen_t, top_p):
    inputs = {}
    if audio_path is None:
        audio = None
    else:
        audio = load_and_transform_audio_data([audio_path])

    inputs['Audio'] = [audio, audio_weight]

    prompts = [llama.format_prompt(prompt)]

    prompts = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    with torch.cuda.amp.autocast():
        results = model.generate(inputs, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
                                 cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)
    text_output = results[0].strip()
    return text_output


def generate_answer(audio_path,
                    audio_weight,
                    prompt,
                    cache_size,
                    cache_t,
                    cache_weight,
                    max_gen_len,
                    gen_t, top_p):
    if audio_path.endswith(".mp3"):
        duration = MP3(audio_path).info.length
        if duration > 60:
            audio = AudioSegment.from_mp3(audio_path)
            audio_splits = split_audio(audio, 1, "mp3")
            answers = []
            for audio_path in audio_splits:
                answers.append(multimodal_generate(audio_path, audio_weight, prompt, cache_size, cache_t, cache_weight,
                                                   max_gen_len, gen_t, top_p))
            answers = " ".join(set(answers))
            return answers
        else:
            return multimodal_generate(audio_path, audio_weight, prompt, cache_size, cache_t, cache_weight, max_gen_len,
                                       gen_t, top_p)
    elif audio_path.endswith(".wav"):
        duration = WAVE(audio_path).info.length
        if duration > 60:
            audio = AudioSegment.from_wav(audio_path)
            audio_splits = split_audio(audio, 1, "wav")
            answers = []
            for audio_path in audio_splits:
                answers.append(multimodal_generate(audio_path, audio_weight, prompt, cache_size, cache_t, cache_weight,
                                                   max_gen_len, gen_t, top_p))
            answers = " ".join(set(answers))
            return answers
        else:
            return multimodal_generate(audio_path, audio_weight, prompt, cache_size, cache_t, cache_weight, max_gen_len,
                                       gen_t, top_p)


output = multimodal_generate(args.audio_path, 1, args.question, 100, 20.0, 0.0, 512, 0.6, 0.8)
print()
print(f"Audio File: {args.audio_path}")
print(f"Q: {args.question}")
print(f"A: {output}")
