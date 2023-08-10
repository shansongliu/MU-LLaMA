import pandas as pd
import shutil
import os
import argparse
from tqdm import tqdm
import json
import torchaudio
from pydub import AudioSegment

parser = argparse.ArgumentParser()
parser.add_argument('--mtt', help='Directory of the MTT dataset', required=True)
parser.add_argument('--musiccaps', help='Directory of the MusicCaps dataset', required=True)
parser.add_argument('--musicqa', help='Directory of the MusicQA dataset to be generated', required=True)
args = parser.parse_args()

mtt_df = pd.read_csv(f"{args.mtt}/MTT_AQA.csv", sep=";")
del mtt_df["Unnamed: 0"]
musiccaps_df = pd.read_csv(f"{args.musiccaps}/MusicCapsAQA.csv", sep=";")
del musiccaps_df["Unnamed: 0"]

if not os.path.exists(f"{args.musicqa}/audios"):
    os.makedirs(f"{args.musicqa}/audios")

musicqa = []

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
            shutil.copy(f"{args.musiccaps}/audios/{row[0]}", f"{args.musicqa}/audios/{filename}")
        _, _ = torchaudio.load(f"{args.musicqa}/audios/{filename}")
        for i in range(4):
            musicqa.append({"audio_name": filename,
                            "conversation": [{"from": "human", "value": q_set[i]},
                                             {"from": "gpt", "value": row[i + 1]}]})
        for qa in row[5:]:
            musicqa.append({"audio_name": filename,
                            "conversation": [{"from": "human", "value": qa.split("\t")[0].replace("Q:", "")},
                                             {"from": "gpt", "value": qa.split("\t")[1].replace("A:", "")}]})
        count += 1
    except:
        print(f"Skipping {args.musiccaps}/audios/{row[0]}")

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
            shutil.copy(f"{args.mtt}/audios/{row[0]}", f"{args.musicqa}/audios/{filename}")
        _, _ = torchaudio.load(f"{args.musicqa}/audios/{filename}")
        for i in range(4):
            musicqa.append({"audio_name": filename,
                            "conversation": [{"from": "human", "value": q_set[i]},
                                             {"from": "gpt", "value": row[i + 1]}]})
        for qa in row[5:]:
            musicqa.append({"audio_name": filename,
                            "conversation": [{"from": "human", "value": qa.split("\t")[0].replace("Q:", "")},
                                             {"from": "gpt", "value": qa.split("\t")[1].replace("A:", "")}]})
        count += 1
    except:
        print(f"Skipping {args.mtt}/audios/{row[0]}")

json.dump(musicqa, open(f"{args.musicqa}/MusicQA.json", "w"), indent=2)
