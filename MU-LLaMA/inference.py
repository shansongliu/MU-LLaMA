import argparse
import torch.cuda

import llama
from util.misc import *
from data.utils import load_and_transform_audio_data

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
model = llama.load(args.model, args.llama_dir, mert_path=args.mert_path, knn=True, knn_dir=args.knn_dir, llama_type=args.llama_type)
model.eval()

def multimodal_generate(
        audio_path,
        audio_weight,
        prompt,
        cache_size,
        cache_t,
        cache_weight,
        max_gen_len,
        gen_t, top_p
):
    inputs = {}
    audio = load_and_transform_audio_data([audio_path])
    inputs['Audio'] = [audio, audio_weight]
    image_prompt = prompt
    text_output = None
    prompts = [llama.format_prompt(prompt)]
    prompts = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    with torch.cuda.amp.autocast():
        results = model.generate(inputs, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
                                     cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)
    text_output = results[0].strip()
    return text_output

output = multimodal_generate(args.audio_path, 1, args.question, 100, 20.0, 0.0, 512, 0.6, 0.8)
print()
print(f"Audio File: {args.audio_path}")
print(f"Q: {args.question}")
print(f"A: {output}")

