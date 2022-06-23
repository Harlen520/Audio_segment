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
from math import ceil


# label2id = {
#     "no-music": 0,
#     "music": 1
# }

# label2id = {
#     "no-music": 0,
#     "bg-music": 1,
#     "fg-music": 2,
# }

def split_frame(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += window_size

def normalize(x):
    x_nml = (x - x.min()) / (x.max() - x.min() + 1e-16)
    return x_nml + 1e-16

def extract_chrome_features(path):
    features = pd.read_csv(path, header=None, sep=";")
    # features = features.replace(0, 1e-12)
    return features.to_numpy()

def extract_log_mel(sound,
                    sr=16000, 
                    frame_shift=10,  # 帧移 ms
                    frame_length=25,  # 帧长 ms
                    n_mels=128,
                    fmin=64,
                    fmax=8000   
                    ):
    hop_length = ceil(sr * frame_shift / 1000) # 帧移 采样点表示
    win_length = int(sr * frame_length / 1000)  # 帧长 采样点表示
    S = librosa.feature.melspectrogram(y=sound, sr=sr, win_length=win_length, hop_length=hop_length,
                                        power=1, n_mels=n_mels, fmin=fmin, fmax=fmax, window="hamm")    # 梅尔谱 Mel filter bank energies
    log_mel_energy = librosa.power_to_db(S)  # 对数梅尔谱 log Mel filter bank energies
    log_mel_energy = log_mel_energy.T
    # librosa.display.specshow(log_mel_energy, y_axis='mel', fmax=8000, x_axis='time')
    # min-max normalised in range(0-1) 
    log_mel_energy_noml = normalize(log_mel_energy)
    return log_mel_energy_noml


def read_data(args, data_dir, label2id, k=None):
    
    features_dir = os.path.join(args.features_dir, os.path.split(data_dir)[-1])
    chroma_dir = os.path.join(args.chroma_dir, os.path.split(data_dir)[-1])
    audio_dir = os.path.join(data_dir, "audio")
    fnames = os.listdir(audio_dir)
    labels_dir = os.path.join(data_dir, "labels")
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    features_path = os.path.join(features_dir, "music_mel_features.npy")
    labels_path = os.path.join(features_dir, "music_frames_labels.npy")
    if os.path.exists(features_path):
        features, labels = load_features(features_path, labels_path)
    else:
        features = []
        labels = []
        for fname in tqdm(fnames):
            with open(os.path.join(labels_dir, fname[:-4] + ".txt"), "r", encoding="utf-8") as f:
                label = [label2id[l] for l in f.read().strip().split("\t")]
                label = np.array(label)

            audio_path = os.path.join(audio_dir, fname)
            music, sr = librosa.load(audio_path)
            
            mel_feature = extract_log_mel(music, sr=sr, frame_shift=args.frame_shift, frame_length=args.frame_length, 
                            n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax)
            if mel_feature.shape[0] != 300:
                continue
            chroma_path = os.path.join(chroma_dir, fname[:-4] + ".csv")
            chroma_feature = extract_chrome_features(chroma_path)
            feature = np.concatenate([mel_feature, chroma_feature], axis=-1)
            features.append(feature)
            labels.append(label)
        
        np.save(features_path, features)
        np.save(labels_path, labels)
    if k is None:
        k = len(features)
    assert k <= len(features)
    for i in range(k):
        yield features[i], labels[i]
        
   
def load_features(features_path, labels_path):
    features = np.load(features_path)
    labels = np.load(labels_path)
    return features, labels


if __name__ == "__main__":
    wav_path = "/home/th/paddle/audio_segment/data/OpenBMAT/full_md_audio/valid/Germany_Talk_21005596_182138727_22.22_46.26_no-music.wav"
    sound, sr = librosa.load(wav_path, duration=3)   # 加载3秒音频
    log_mel_energy = extract_log_mel(sound=sound, sr=sr, n_mels=128, fmin=64)
    # chrome_features = extract_chrome_features(sound, sr=sr)
    