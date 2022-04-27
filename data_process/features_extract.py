from email import header
from ast import arg
import datasets
import os
import librosa
import numpy as np
from paddle import dtype
from pkg_resources import yield_lines
from tqdm import tqdm
import pandas as pd
from paddlenlp.datasets.dataset import MapDataset


label2id = {
    "no-music": 0,
    "music": 1
}

def split_frame(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += window_size

def normalize(x):
    
    return (x - x.min() + 1e-16) / (x.max() - x.min() + 1e-9)

def extract_chrome_features(path):
    features = pd.read_csv(path, header=None, sep=";")
    features = features.replace(0, 1e-6)
    return features.to_numpy()

def extract_log_mel(sound,
                    sr=16000, 
                    frame_shift=10,  # 帧移 ms
                    frame_length=25,  # 帧长 ms
                    n_mels=80,
                    fmin=0.0,
                    fmax=8000   
                    ):
    hop_length = int(sr * frame_shift / 1000) # 帧移 采样点表示
    win_length = int(sr * frame_length / 1000)  # 帧长 采样点表示
    S = librosa.feature.melspectrogram(y=sound, sr=sr, win_length=win_length, hop_length=hop_length,
                                        power=1, n_mels=n_mels, fmin=fmin, fmax=fmax, window="hamm")    # 梅尔谱 Mel filter bank energies
    log_mel_energy = librosa.power_to_db(S)  # 对数梅尔谱 log Mel filter bank energies
    log_mel_energy = log_mel_energy.T
    # librosa.display.specshow(log_mel_energy, y_axis='mel', fmax=8000, x_axis='time')
    # min-max normalised in range(0-1) 
    log_mel_energy_noml = normalize(log_mel_energy)
    return log_mel_energy_noml


def read_data(args, audio_dir):
    chroma_dir = os.path.join(args.chroma_dir, os.path.split(audio_dir)[-1])
    combine_features_dir = os.path.join(args.combine_dir, os.path.split(audio_dir)[-1])
    fnames = os.listdir(audio_dir)
    if not os.path.exists(combine_features_dir):
        os.makedirs(combine_features_dir)
    for fname in tqdm(fnames):
        fs = fname[:-4].split("_")
        label = label2id[fs[-1]]
        combine_feature_path = os.path.join(combine_features_dir, fname[:-4] + ".csv")
        if os.path.exists(combine_feature_path):
            features = pd.read_csv(combine_feature_path, header=None).to_numpy().astype(np.float32)
            yield features[:args.max_seq_length], label
            continue
        audio_path = os.path.join(audio_dir, fname)
        music, sr = librosa.load(audio_path)
        # mel_feature: [frames, n_mels]
        mel_feature = extract_log_mel(music, sr=sr, frame_shift=args.frame_shift, frame_length=args.frame_length, 
                        n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax)
        # chroma feature: [frames, 12]
        chroma_path = os.path.join(chroma_dir, fname[:-4] + ".txt")
        chroma_feature = extract_chrome_features(chroma_path)
        chroma_feature = normalize(chroma_feature)
        features = np.concatenate([mel_feature, chroma_feature], axis=-1).astype(np.float32)
        df_features = pd.DataFrame(features)
        df_features.to_csv(combine_feature_path, index=None, header=None)
        yield features[:args.max_seq_length], label
        
   



if __name__ == "__main__":
    wav_path = "/home/th/paddle/audio_segment/data/OpenBMAT/full_md_audio/valid/Germany_Talk_21005596_182138727_22.22_46.26_no-music.wav"
    sound, sr = librosa.load(wav_path, duration=3)   # 加载3秒音频
    log_mel_energy = extract_log_mel(sound=sound, sr=sr, n_mels=128, fmin=64)
    # chrome_features = extract_chrome_features(sound, sr=sr)
    