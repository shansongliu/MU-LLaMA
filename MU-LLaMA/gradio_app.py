import argparse

import gradio as gr
import random

import torch.cuda

from diffusers import StableUnCLIPImg2ImgPipeline
from image_generate import image_generate

import llama
from util.misc import *
from data.utils import load_and_transform_audio_data

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="./ckpts/checkpoint.pth", type=str,
    help="Name of or path to MU-LLaMA pretrained checkpoint",
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
        gen_t, top_p, output_type
):
    inputs = {}
    if audio_path is None:
        raise gr.Error('Please select an audio')
    if audio_weight == 0:
        raise gr.Error('Please set the weight')
    audio = load_and_transform_audio_data([audio_path])
    inputs['Audio'] = [audio, audio_weight]

    image_prompt = prompt  # image use original prompt

    text_output = None
    image_output = None
    if output_type == "Text":
        # text output
        prompts = [llama.format_prompt(prompt)]

        prompts = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        with torch.cuda.amp.autocast():
            results = model.generate(inputs, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
                                     cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)
        text_output = results[0].strip()
        print(text_output)

    # else:
    #     # image output
    #     image_output = image_generate(inputs, model, pipe, image_prompt, cache_size, cache_t, cache_weight)

    return text_output


def create_imagebind_llm_demo():
    with gr.Blocks() as imagebind_llm_demo:
        with gr.Column():
            with gr.Row():
                audio_path = gr.Audio(label='Audio Input', type='filepath')
                with gr.Column():
                    output_dropdown = gr.Dropdown(['Text', 'Image'], value='Text', label='Output type')
                    # with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        cache_size = gr.Slider(minimum=1, maximum=100, value=10, interactive=True, label="Cache Size")
                        cache_t = gr.Slider(minimum=0.0, maximum=100, value=20, interactive=True,
                                            label="Cache Temperature")
                        cache_weight = gr.Slider(minimum=0.0, maximum=1, value=0.1, interactive=True,
                                                 label="Cache Weight")
                    with gr.Row() as text_config_row:
                        max_gen_len = gr.Slider(minimum=1, maximum=1024, value=1024, interactive=True, label="Max Length")
                        gen_t = gr.Slider(minimum=0, maximum=1, value=0.25, interactive=True, label="Temperature")
                        top_p = gr.Slider(minimum=0, maximum=1, value=1.0, interactive=True, label="Top p")

            with gr.Column():
                chatbot = gr.Chatbot()
                msg = gr.Textbox(label='Question')
                clear = gr.ClearButton([msg, chatbot])

    # def change_output_type(output_type):
    #     if output_type == 'Text':
    #         result = [gr.update(visible=False),
    #         gr.update(visible=True),
    #         gr.update(label='Question'),
    #         gr.update(visible=True)]
    #     elif output_type == 'Image':
    #         result = [gr.update(visible=True),
    #         gr.update(visible=False),
    #         gr.update(label='Prompt'),
    #         gr.update(visible=False)]
    #
    #     return result

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, audio_file_path, cache_size_value, cache_t_value, cache_weight_value, max_gen_len_value, gen_t_value, top_p_value):
        #print(cache_size.value, cache_t.value, cache_weight.value, max_gen_len.value, gen_t.value, top_p.value)
        bot_message = multimodal_generate(
            audio_file_path,
            1,
            history[-1][0],
            cache_size_value,
            cache_t_value,
            cache_weight_value,
            max_gen_len_value,
            gen_t_value, top_p_value, "Text")
        history[-1][1] = ""
        for word in bot_message.split():
            history[-1][1] = " ".join([history[-1][1], word])
            yield history

    def hear_audio(history):
        return history + [["Listen to this music", None]]

    def start_chat(history):
        bot_message = "I have listened to the music, please ask any question you want"
        history[-1][1] = ""
        for word in bot_message.split():
            history[-1][1] = " ".join([history[-1][1], word])
            yield history

    audio_path.upload(hear_audio, chatbot, chatbot, queue=False).then(
        start_chat, chatbot, chatbot
    )

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, audio_path, cache_size, cache_t, cache_weight, max_gen_len, gen_t, top_p], chatbot
    )

    clear.click(lambda: None, None, chatbot, queue=False)

    # output_dropdown.change(change_output_type, output_dropdown,
    #                        [image_output, text_output, prompt, text_config_row])
    #

    return imagebind_llm_demo


# pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("./ckpts/stable-diffusion-2-1-unclip")
# pipe = pipe.to("cuda")
description = """
# MU-LLaMA🎧
"""

with gr.Blocks(theme=gr.themes.Default(), css="#pointpath {height: 10em} .label {height: 3em}") as demo:
    gr.Markdown(description)
    create_imagebind_llm_demo()

if __name__ == "__main__":
    demo.queue(api_open=True, concurrency_count=1).launch(share=False, inbrowser=True, server_name='0.0.0.0',
                                                          server_port=24000, debug=True)
