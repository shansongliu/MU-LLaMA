<div>
  <h1>
    MU-LLaMA: <br>Music Understanding Large Language Model
    <img src="MU-LLaMA.png" height=100px align="right"/>
  </h1>
</div>

This is the official repository for *MU-LLaMA: Large Language Model for Music Question Answering*

## Introduction
The MU-LLaMA model is Music Understanding Language Model designed with the purpose of answering questions based on music. Our model is also designed with the purpose of captioning music files to generate Text-to-Music Generation datasets. We also provide the code for generating our MusicQA dataset from MusicCaps and the FMA dataset.

<p align="center">
  <img src="./assets/MU-LLaMA.png">
</p>

## MusicQA Dataset Generation

We use the MusicCaps and the FMA dataset to generate our MusicQA dataset. You can download the generated MusicQA dataset here. To generate the dataset yourself, first download the MusicCaps and MTT datasets. Once downloaded, the directory structure would be as shown.

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
> Run the following command to flatten the MTT audio file structure
```
find ./MTT/audios -mindepth 2 -type f -exec mv -t ./MTT/audios -i '{}' +
```


By running ***musiccaps_process.py*** and ***MTT_process.py***, you can generate the question answer pairs from each of the datasets and by running ***generate_dataset.py*** the final dataset will be generated.

```
usage: generate_dataset.py [-h] --mtt MTT --musiccaps MUSICCAPS --musicqa MUSICQA

optional arguments:
  -h, --help            show this help message and exit
  --mtt MTT             Directory of the MTT dataset
  --musiccaps MUSICCAPS
                        Directory of the MusicCaps dataset
  --musicqa MUSICQA     Directory of the MusicQA dataset to be generated
```
