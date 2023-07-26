<div>
  <h1>
    MU-LLaMA: <br>Music Understanding Large Language Model
    <img src="MU-LLaMA.png" height=100px align="right"/>
  </h1>
</div>

This is the official repository for *MU-LLaMA: Large Language Model for Music Question Answering*

## Introduction
The MU-LLaMA model is Music Understanding Language Model designed with the purpose of answering questions based on music. Our model is also designed with the purpose of captioning music files to generate Text-to-Music Generation datasets. We also provide the code for generating our MusicQA dataset from [MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps) and the [MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) datasets.

<p align="center">
  <img src="./assets/MU-LLaMA.png">
</p>

## MU-LLaMA Demo

For the working of our model, Facebook's LLaMA model weights are required, details on obtaining these weights are given on [HuggingFace](https://huggingface.co/docs/transformers/main/model_doc/llama). Our pretrained weights for the MU-LLaMA model can be downloaded here. Once downloaded, store the files in the ckpts folder within the MU-LLaMA directory. 

Once downloaded the directory structure will be as shown below.
```
.
├── ...
├── MU-LLaMA                
│   ├── ckpts
│   │   │── LLaMA
│   │   │   │── 7B
│   │   │   │   │── checklist.chk
│   │   │   │   │── consolidated.00.pth
│   │   │   │   │── params.json
│   │   │   │── llama.sh
│   │   │   │── tokenizer.model
│   │   │   │── tokenizer_checklist.chk
│   │   │── 7B.pth
│   │   ├── checkpoint.pth
└── ...
```

The demo can be run using [***gradio_app.py***](./MU-LLaMA/gradio_app.py).
```
python gradio_app.py --model ./ckpts/checkpoint.pth --llama_dir ./ckpts/LLaMA
```

## Training MU-LLaMA

To train the MU-LLaMA model, follow the steps as below.

### MusicQA Dataset

We use the [MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps) and the [MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) dataset to generate our MusicQA dataset. You can download the generated MusicQA dataset [here](./). To generate the dataset yourself, first download the MusicCaps and MTT datasets. Once downloaded, the directory structure would be as shown.

```
.
├── ...
├── MusicQA                
│   ├── MTT
│   │   ├── audios
│   │   │   │── ...
│   │   ├── annotations_final.csv
│   ├── MusicCaps
│   │   ├── audios
│   │   │   │── ...
│   │   ├── musiccaps-public.csv
│   ├── MTT_process.py
│   ├── musiccaps_process.py
│   ├── generate_dataset.py
└── ...
```

 
> &#128221; **Note**:
> Run the following command to flatten the MTT audio file structure once downloaaded and extracted,
```
find ./MTT/audios -mindepth 2 -type f -exec mv -t ./MTT/audios -i '{}' +
```


By running [***musiccaps_process.py***](./MusicQA/musiccaps_process.py) and [***MTT_process.py***](./MusicQA/MTT_process.py), you can generate the question answer pairs from each of the datasets and by running [***generate_dataset.py***](./MusicQA/generate_dataset.py) the final dataset will be generated.

```
usage: generate_dataset.py [-h] --mtt MTT --musiccaps MUSICCAPS --musicqa MUSICQA

optional arguments:
  -h, --help            show this help message and exit
  --mtt MTT             Directory of the MTT dataset
  --musiccaps MUSICCAPS
                        Directory of the MusicCaps dataset
  --musicqa MUSICQA     Directory of the MusicQA dataset to be generated
```

## MU-LLaMA Training

To train the MU-LLaMA model, use the [***finetune.sh***](./MU-LLaMA/finetune.sh) script.
```
./finetune.sh ./ckpts/LLaMA ./ckpts/7B.pth ./musicqa.yaml ./ckpts/MU-LLaMA
```

This will train the MU-LLaMA model for 20 epochs. The hyperparameters can be modified in the [***finetune.sh***](./MU-LLaMA/finetune.sh) file. Once trained, the model can be tested using the Gradio demo.
