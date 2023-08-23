import pandas as pd
import os
import argparse
from tqdm import tqdm
import json
from pydub import AudioSegment
from utils import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--mtt', help='Directory of the MTT dataset', required=True)
parser.add_argument('--mtg', help='Directory of the MTG dataset', required=True)
parser.add_argument('--musiccaps', help='Directory of the MusicCaps dataset', required=True)
parser.add_argument('--musicqa', help='Directory of the MusicQA dataset to be generated', required=True)
args = parser.parse_args()

mtt_df = pd.read_csv(f"{args.mtt}/MTT_AQA.csv", sep=";")
del mtt_df["Unnamed: 0"]
musiccaps_df = pd.read_csv(f"{args.musiccaps}/MusicCapsAQA.csv", sep=";")
del musiccaps_df["Unnamed: 0"]
mtg_df = pd.read_csv(f"{args.mtg}/MTG_AQA.csv", sep=";")
del musiccaps_df["Unnamed: 0"]

if not os.path.exists(f"{args.musicqa}/audios"):
    os.makedirs(f"{args.musicqa}/audios")

pretrain_musicqa = []

q_set = ["Describe the audio", "Describe the audio in detail", "What do you hear in the audio?",
         "What can be inferred from the audio?"]

count = 0
for i, row in tqdm(musiccaps_df.iterrows(), total=len(musiccaps_df)):
    if not os.path.exists(f"{args.musiccaps}/audios/{row[0]}"):
        continue
    try:
        file_format = row[0].split('.')[-1]
        filename = f"{str(count).zfill(6)}.wav"
        if file_format == "mp3":
            sound = AudioSegment.from_mp3(f"{args.musiccaps}/audios/{row[0]}")
            sound.export(f"{args.musicqa}/audios/{filename}", format="wav")
        else:
            copyfile(f"{args.musiccaps}/audios/{row[0]}", f"{args.musicqa}/audios/{filename}")
        for j in range(4):
            pretrain_musicqa.append({"audio_name": filename,
                            "conversation": [{"from": "human", "value": q_set[j]},
                                             {"from": "gpt", "value": row[j + 1]}]})
        for qa in row[5:]:
            pretrain_musicqa.append({"audio_name": filename,
                            "conversation": [{"from": "human", "value": qa.split("\t")[0].replace("Q:", "")},
                                             {"from": "gpt", "value": qa.split("\t")[1].replace("A:", "")}]})
        count += 1
    except:
        print(f"Skipping {args.musiccaps}/audios/{row[0]}")

json.dump(pretrain_musicqa, open(f"{args.musicqa}/PretrainMusicQA.json", "w"), indent=2)

finetune_musicqa = []

for i, row in tqdm(mtt_df.iterrows(), total=len(mtt_df)):
    if not os.path.exists(f"{args.mtt}/audios/{row[0]}"):
        continue
    try:
        file_format = row[0].split('.')[-1]
        filename = f"{str(count).zfill(6)}.wav"
        if file_format == "mp3":
            sound = AudioSegment.from_mp3(f"{args.mtt}/audios/{row[0]}")
            sound.export(f"{args.musicqa}/audios/{filename}", format="wav")
        else:
            copyfile(f"{args.mtt}/audios/{row[0]}", f"{args.musicqa}/audios/{filename}")
        for j in range(4):
            finetune_musicqa.append({"audio_name": filename,
                            "conversation": [{"from": "human", "value": q_set[j]},
                                             {"from": "gpt", "value": row[j + 1]}]})
        for qa in row[5:]:
            finetune_musicqa.append({"audio_name": filename,
                            "conversation": [{"from": "human", "value": qa.split("\t")[0].replace("Q:", "")},
                                             {"from": "gpt", "value": qa.split("\t")[1].replace("A:", "")}]})
        count += 1
    except:
        print(f"Skipping {args.mtt}/audios/{row[0]}")

json.dump(finetune_musicqa, open(f"{args.musicqa}/FinetuneMusicQA.json", "w"), indent=2)

eval_musicqa = []

for i, row in tqdm(mtg_df.iterrows(), total=len(mtg_df)):
    if not os.path.exists(f"{args.mtg}/{row[0]}"):
        continue
    try:
        file_format = row[0].split('.')[-1]
        filename = f"{str(count).zfill(6)}.wav"
        if file_format == "mp3":
            sound = AudioSegment.from_mp3(f"{args.mtg}/{row[0]}")
            sound.export(f"{args.musicqa}/audios/{filename}", format="wav")
        else:
            copyfile(f"{args.mtg}/{row[0]}", f"{args.musicqa}/audios/{filename}")
        for j in range(4):
            eval_musicqa.append({"audio_name": filename,
                            "conversation": [{"from": "human", "value": q_set[j]},
                                             {"from": "gpt", "value": row[j + 1]}]})
        for qa in row[5:]:
            eval_musicqa.append({"audio_name": filename,
                            "conversation": [{"from": "human", "value": qa.split("\t")[0].replace("Q:", "")},
                                             {"from": "gpt", "value": qa.split("\t")[1].replace("A:", "")}]})
        count += 1
    except:
        print(f"Skipping {args.mtg}/audios/{row[0]}")

json.dump(eval_musicqa, open(f"{args.musicqa}/EvalMusicQA.json", "w"), indent=2)
