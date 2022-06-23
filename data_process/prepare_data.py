import os
import pandas as pd
import utils
from tqdm import tqdm
import numpy as np
import json
import subprocess
import librosa
import soundfile
from decimal import Decimal

def window(start, end, window_size, overlap=0):
    s = start
    while s < end:
        yield s, s + window_size
        s += (window_size - overlap)

def gen_intervals(annots):
    intervals = []
    n = len(annots)
    for i in range(n):
        segment = annots[str(i)]
        cla = segment["class"]
        start = segment["start"] * 100
        end = segment["end"] * 100
        intervals.append([start, end])
    return intervals

def cut_audio_to_3s(agr_level, json_path, original_audio_dir, output_dir, fnames, seq_len=3.0, overlap=0):
    save_audio_dir = os.path.join(output_dir, "audio")
    save_label_dir = os.path.join(output_dir, "labels")
    if not os.path.exists(save_audio_dir):
        os.makedirs(save_audio_dir)
    if not os.path.exists(save_label_dir):
        os.makedirs(save_label_dir)
    with open(json_path, 'r') as f:
        data = json.load(f)
    seq_len = Decimal(str(seq_len))
    overlap = Decimal(str(overlap))
    total_num = 0
    m = 0
    for fname in tqdm(fnames):
        
        annots = data["annotations"]['annotator_a'][fname]
        intervals = gen_intervals(annots)
        fpath = os.path.join(original_audio_dir, fname + '.wav')
        
        start = Decimal(str(0))
        end = Decimal(str(60))
       
        for s, e in window(start, end, seq_len, overlap):
            if end - s < 1:
                break
            if e > end:
                s = end - seq_len
                e = end   
            assert e - s == seq_len
            labels = []
            for i in range(int(s*100), int(e*100)):
                for ix, (st, ed) in enumerate(intervals):
                    if i >= st and i <= ed:
                        labels.append(annots[str(ix)]["class"])
                        break
            if len(labels) != 300:
                continue
            clas = np.unique(labels)
            if len(clas) > 1:
                m += 1
            new_fname = '_'.join([fname, str(s), str(e)])
            output_fname =  new_fname + '.wav'
            output_fpath = os.path.join(save_audio_dir, output_fname)
            # sound, sr = librosa.load(audio_file, offset=float(s), duration=float(seq_len))
            # new_fname = "_".join(fs[:-2] + [str(s), str(e)]) + ".wav"
            # soundfile.write(os.path.join(save_audio_dir, new_fname), sound, sr)
            with open(os.path.join(save_label_dir, new_fname+".txt"), "w", encoding="utf-8") as f:
                for l in labels:
                    f.write(l + "\t")
        
            cmd = 'ffmpeg -loglevel panic -i {0} -ss {1} -to {2} {3}'.format(fpath, float(s), float(e), output_fpath)
            subprocess.call(cmd.split(' '))
            # os.system(cmd)
            total_num += 1
    print(f"total num: {total_num}")
    print(f"m: {m}")
    assert len(os.listdir(save_audio_dir)) == total_num
    assert len(os.listdir(save_label_dir)) == total_num

def split_data(split_path):
    
    df = pd.read_csv(split_path)
    train = []
    valid = []
    test = []
    for i, grop in df.groupby(["split"]):
        if i == 8:
            valid = grop
        elif i == 9:
            test = grop
        else:
            train.append(grop)
    train = pd.concat(train)
    return train, valid, test


if __name__ == "__main__":
    # bat_csv_dir = '/home/th/paddle/audio_segment/data/OpenBMAT/annotations/bat/'
    # output_dir = '/home/th/paddle/audio_segment/data/OpenBMAT/annotations/'
    # utils.generate_annotations(bat_csv_dir, output_dir)
    # # 1. 按照注释文件对原始音频数据切割，以及分成train, valid, test
    split_path = "/home/th/paddle/audio_segment/data/OpenBMAT/splits/splits.csv"
    train, valid, test = split_data(split_path)
    agr_level = 'full' # or 'partial'
    json_path = '/home/th/paddle/audio_segment/data/OpenBMAT/annotations/json/MD_mapping.json' # or any other
    audio_dir = '/home/th/paddle/audio_segment/data/OpenBMAT/audio/'
    # output_dir = '/home/th/paddle/audio_segment/data/OpenBMAT/full_md_audio/'
    save_dir = '/home/th/paddle/audio_segment/data/dataset_2class/'
    # 2. 继续将音频分割为3秒音频，因为我们的输入是3秒序列
    cut_audio_to_3s(agr_level, json_path, audio_dir, save_dir + "train/", 
                        train["file_name"].to_list(), seq_len=3, overlap=0.5)
    cut_audio_to_3s(agr_level, json_path, audio_dir, save_dir + "valid/", 
                        valid["file_name"].to_list(), seq_len=3, overlap=0.5)
    cut_audio_to_3s(agr_level, json_path, audio_dir, save_dir + "test/", 
                        test["file_name"].to_list(), seq_len=3, overlap=0.5)
    
    
    # split_audio(json_path, output_dir + "train/", save_dir + "train/", 3, overlap=0.5)
    # split_audio(json_path, output_dir + "valid/", save_dir + "valid/", 3, overlap=0.5)
    # split_audio(json_path, output_dir + "test/", save_dir + "test/", 3, overlap=0.5)
