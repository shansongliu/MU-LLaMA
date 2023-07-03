import data.utils as data
import llama


llama_dir = "./ckpts/LLaMA"

model = llama.load("./ckpts/audio_finetuned_2/checkpoint.pth", llama_dir, knn=True)
model.eval()

inputs = {}
#image = data.load_and_transform_vision_data(["examples/girl.jpg"], device='cuda')
#inputs['Image'] = [image, 1]
audio = data.load_and_transform_audio_data(['./examples/looperman-l-0139050-0013124-dusthill-hot-two-chored-trumpets.wav'], device='cuda')
inputs['Audio'] = [audio, 1]

results = model.generate(
    inputs,
    [llama.format_prompt("Describe the music")],
    max_gen_len=256
)
result = results[0].strip()
print(result)
