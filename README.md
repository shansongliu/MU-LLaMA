<div>
  <h1>
    MU-LLaMA: <br>Music Understanding Large Language Model
    <img src="./assets/logo.png" height=100px align="right"/>
  </h1>
</div>

[![PWC](https://img.shields.io/badge/%F0%9F%93%8E%20arXiv-Paper-red)](https://arxiv.org/abs/2308.11276)
[![PWC](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MusicQA%20Dataset-green)]([https://arxiv.org/abs/2308.11276](https://huggingface.co/datasets/mu-llama/MusicQA))
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/music-understanding-llama-advancing-text-to/music-question-answering-on-musicqa-dataset)](https://paperswithcode.com/sota/music-question-answering-on-musicqa-dataset?p=music-understanding-llama-advancing-text-to)

This is the official repository for *[Music Understanding LLaMA: Advancing Text-to-Music Generation with Question Answering and Captioning](https://arxiv.org/abs/2308.11276)*

The demo page with more information regarding the MU-LLaMA model is avilable [here](https://crypto-code.github.io/MU-LLaMA-Demo/).

## Introduction
The MU-LLaMA model is Music Understanding Language Model designed with the purpose of answering questions based on music. Our model is also designed with the purpose of captioning music files to generate Text-to-Music Generation datasets. The model uses MERT + LLaMA as the backbone and employs an adapter to encoperate music context information to guide LLaMA's output. MERT was chosen as the music encoder for our model after comparison of different music representation models, which can be viewed [here](https://github.com/crypto-code/Music-Representation-Comparison). We also provide the code for generating our MusicQA dataset from [MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps) and the [MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) datasets.

<p align="center">
  <img src="./assets/MU-LLaMA.png">
</p>

## MU-LLaMA Demo

For the working of our model, Facebook's LLaMA-2 model weights are required, details on obtaining these weights are given on [HuggingFace](https://huggingface.co/docs/transformers/main/model_doc/llama). Our pretrained weights for the MU-LLaMA model, finetuned from **LLaMA 7B-2** can be downloaded [here](https://huggingface.co/mu-llama/MU-LLaMA/tree/main). Once downloaded, store the files in the ckpts folder within the MU-LLaMA directory. 

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

We use Python 3.9.17 for this project and the library requirements are given in [***requirements.txt***](./requirements.txt). The demo can be run using [***gradio_app.py***](./MU-LLaMA/gradio_app.py).
```
python gradio_app.py --model ./ckpts/checkpoint.pth --llama_dir ./ckpts/LLaMA
```

## Training MU-LLaMA

To train the MU-LLaMA model, follow the steps as below.

### MusicQA Dataset

We use the [MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps) and the [MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) dataset to generate our training MusicQA dataset and the [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) for evaluation. You can download the generated MusicQA dataset [here](https://huggingface.co/datasets/mu-llama/MusicQA). 

To generate the dataset yourself, first download the MusicCaps, MTT and MTG datasets. Once downloaded, the directory structure would be as shown.

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
│   ├── MTG
│   │   ├── audios
│   │   │   │── 00
│   │   │   │── 01
│   │   │   │── ...
│   │   ├── raw_30s_cleantags_50artists.tsv
│   ├── MTT_process.py
│   ├── musiccaps_process.py
│   ├── MTG_process.py
│   ├── generate_dataset.py
└── ...
```

The MusicQA dataset generation is a very computationally intensive process which takes around 8 days per dataset on a Tesla V100-SXM2-32GB GPU, so it is recommended to download our generated dataset.

> &#128221; **Notes**:
> Run the following command to flatten the MTT audio file structure once downloaaded and extracted,
```
find ./MTT/audios -mindepth 2 -type f -exec mv -t ./MTT/audios -i '{}' +
```
> We only use the folders 00 to 09 from the MTG dataset


By running [***musiccaps_process.py***](./MusicQA/musiccaps_process.py), [***MTT_process.py***](./MusicQA/MTT_process.py) and [***MTG_process.py***](./MusicQA/MTG_process.py), you can generate the question answer pairs from each of the datasets and by running [***generate_dataset.py***](./MusicQA/generate_dataset.py) the final datasets for pretraining, finetuning and evaluation will be generated.

```
usage: generate_dataset.py [-h] --mtt MTT --mtg MTG --musiccaps MUSICCAPS --musicqa MUSICQA

optional arguments:
  -h, --help            show this help message and exit
  --mtt MTT             Directory of the MTT dataset
  --mtg MTG             Directory of the MTG dataset
  --musiccaps MUSICCAPS
                        Directory of the MusicCaps dataset
  --musicqa MUSICQA     Directory of the MusicQA dataset to be generated
```

### MU-LLaMA Pretraining

To pretrain the MU-LLaMA model, we use the MusicCaps part of the MusicQA dataset and the Alpaca Instruction dataset with the [***pretrain.sh***](./MU-LLaMA/pretrain.sh) script.
```
./pretrain.sh ./ckpts/LLaMA-2 ./configs/pretrain.yaml ./ckpts/MU-LLaMA_Pretrain
```

This will pretrain the MU-LLaMA model for 150 epochs. The hyperparameters can be modified in the [***pretrain.sh***](./MU-LLaMA/pretrain.sh) file. 

### MU-LLaMA Finetuning

To finetune the MU-LLaMA model, we use the MTT part of the MusicQA dataset with the [***finetune.sh***](./MU-LLaMA/finetune.sh) script.
```
./finetune.sh ./ckpts/LLaMA-2 ./ckpts/MU-LLaMA_Pretrain/checkpoint.pth ./configs/finetune.yaml ./ckpts/MU-LLaMA_Finetune
```

This will finetune the MU-LLaMA model for 20 epochs. The hyperparameters can be modified in the [***finetune.sh***](./MU-LLaMA/finetune.sh) file. The MU-LLaMA model with 7B parameters takes approximately 2 days to train on a Tesla V100-SXM2-32GB GPU. Once trained, the model can be tested using the Gradio demo.

### MU-LLaMA Inference

To test the model without Gradio, the [***inference.py***](./MU-LLaMA/inference.py) script can be used.
```
usage: inference.py [-h] [--model MODEL] [--llama_type LLAMA_TYPE] [--llama_dir LLAMA_DIR] [--mert_path MERT_PATH] --audio_path AUDIO_PATH [--question QUESTION]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Name of or path to the trained checkpoint
  --llama_type LLAMA_TYPE
                        Type of llama original weight
  --llama_dir LLAMA_DIR
                        Path to LLaMA pretrained checkpoint
  --mert_path MERT_PATH
                        Path to MERT pretrained checkpoint
  --audio_path AUDIO_PATH
                        Path to the input music file
  --question QUESTION   Question to ask the model
```

### MU-LLaMA Evaluation

Our model was compared against audio enabled models such as the Listen, Think and Understand (LTU) model and the LLaMA Adapter model trained on our MusicQA dataset. We evaluate the models using BLEU (B-U), METEOR (M-R), ROUGE<sub>L</sub> (R-L) and BERT-Score (BERT-S) which are common evaluation metrics for text generation. For the BLEU score, a weighted average of BLEU<sub>1</sub>, BLEU<sub>2</sub>, BLEU<sub>3</sub> and BLEU<sub>4</sub> (weight = 0.25 for each) is used.

The evaluation scripts are given in the ModelEvaluations folder. The generate scripts are used to generate the answers for all the questions in the dataset.

```
usage: generate_mullama.py [-h] [--model MODEL] [--knn KNN] [--llama_type LLAMA_TYPE] [--llama_dir LLAMA_DIR] [--mert_path MERT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Name of or path to the trained checkpoint
  --knn KNN             Name of or path to the directory with knn checkpoint
  --llama_type LLAMA_TYPE
                        Type of llama original weight
  --llama_dir LLAMA_DIR
                        Path to LLaMA pretrained checkpoint
  --mert_path MERT_PATH
                        Path to MERT pretrained checkpoint
```
```
usage: generate_ltu.py [-h] [--demo DEMO]

optional arguments:
  -h, --help   show this help message and exit
  --demo DEMO  Link to the LTU Demo Page
```
```
usage: generate_llama-adapter.py [-h] [--model MODEL] [--llama_dir LLAMA_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Name of or path to the trained checkpoint
  --llama_dir LLAMA_DIR
                        Path to LLaMA pretrained checkpoint
```

Once generated, [***evaluate.py***](./ModelEvaluations/evaluate.py) can be used to evaluated the generated answers for the three models. The results are shown below.

| **Model**         | **B-U &#8593;**        | **M-R &#8593;**        | **R-L &#8593;**        | **BERT-S &#8593;**        |
|-------------------|------------------------|------------------------|------------------------|---------------------------|
| LTU               | 0.242                  | 0.274                  | 0.326                  | 0.887                     |
| LLaMA Adapter     | 0.273                  | 0.334                  | 0.413                  | 0.895                     |
| **MU-LLaMA**      | **0.306**              | **0.385**              | **0.466**              | **0.901**                 |


## Acknowledgements

This code contains elements from the following repos:
- [OpenGVLab/LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter)
- [yizhilll/MERT](https://github.com/yizhilll/MERT)


## Cite our work
If you find this repo useful, please consider citing: 
```bibtex
@article{liu2023music,
  title={{Music Understanding LLaMA: Advancing Text-to-Music Generation with Question Answering and Captioning}},
  author={Liu, Shansong and Hussain, Atin Sakkeer and Sun, Chenshuo and Shan, Ying},
  journal={arXiv preprint arXiv:2308.11276},
  year={2023}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=crypto-code/MU-LLaMA&type=Date)](https://star-history.com/#crypto-code/MU-LLaMA&Date)

