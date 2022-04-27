import os
import pandas as pd
import utils
from tqdm import tqdm
import json
import subprocess
import librosa
import soundfile

def cut_audio_with_agreement(agr_level, json_path, audio_dir, output_dir, fnames):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for fname in tqdm(fnames):
        data['annotations']
        segs = data['agreement'][fname]['segs'][agr_level]
        fpath = os.path.join(audio_dir, fname + '.wav')
        for s in segs:
            output_fname = '_'.join([fname, str(s[0]), str(s[1]), s[2]]) + '.wav'
            output_fpath = os.path.join(output_dir, output_fname)
            cmd = 'ffmpeg -loglevel panic -i {0} -ss {1} -to {2} {3}'.format(fpath, s[0], s[1], output_fpath)
            subprocess.call(cmd.split(' '))

def split_data():
    split_path = "C:\\Users\\hua\\Desktop\\audio_segmentation\\OpenBMAT\\splits/splits.csv"
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

def window(start, end, window_size, overlap=0):
    s = start
    while s < end:
        yield s, s + window_size
        s += (window_size - overlap)

def split_audio(original_audio_dir, audio_dir, save_dir, seq_len=3.0, overlap=0):
    '''
    seq_len: 音频序列长度， 单位s
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fnames = os.listdir(audio_dir)
    total_num = 0
    for fname in tqdm(fnames):
        fs = ".".join(fname.split(".")[:-1]).split("_")
        label = fs[-1]
        start = float(fs[-3])
        end = float(fs[-2])
        if end - start < 2:
            continue
        original_fname = os.path.join(original_audio_dir, "_".join(fs[:-3]) + ".wav")
        for s, e in window(start, end, seq_len, overlap):
            if end - s < 1.5:
                break
            if e > end + 1:
                s = end + 1.0 - seq_len
                e = end + 1
            if s > 60 - seq_len:
                s = 60 - seq_len
                e = 60
            if e > 60 or s < 0:
                break
            
            sound, sr = librosa.load(original_fname, offset=s, duration=seq_len)
            new_fname = "_".join(fs[:-3] + [str(s), str(e), label]) + ".wav"
            soundfile.write(os.path.join(save_dir, new_fname), sound, sr)
            total_num += 1
    print(f"total num: {total_num}")
    

if __name__ == "__main__":
    # 1. 先执行， 重新生成注释文件
    bat_csv_dir = 'C:\\Users\\hua\\Desktop\\audio_segmentation\\OpenBMAT\\annotations/bat/'
    output_dir = 'C:\\Users\\hua\\Desktop\\audio_segmentation\\OpenBMAT\\annotations/'
    utils.generate_annotations(bat_csv_dir, output_dir)
    # 2. 按照注释文件对原始音频数据切割，以及分成train, valid, test
    train, valid, test = split_data()
    agr_level = 'full' # or 'partial'
    json_path = 'C:\\Users\\hua\\Desktop\\audio_segmentation\\OpenBMAT\\annotations/json/MD_mapping.json' # or any other
    audio_dir = 'C:\\Users\\hua\\Desktop\\audio_segmentation\\OpenBMAT/audio/'
    output_dir = 'C:\\Users\\hua\\Desktop\\audio_segmentation\\OpenBMAT\\full_md_audio/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    cut_audio_with_agreement(agr_level, json_path, audio_dir, output_dir + "train/", train["file_name"].to_list())
    cut_audio_with_agreement(agr_level, json_path, audio_dir, output_dir + "valid/", valid["file_name"].to_list())
    cut_audio_with_agreement(agr_level, json_path, audio_dir, output_dir + "test/", test["file_name"].to_list())
    # 3. 继续将音频分割为3秒音频，因为我们的输入是3秒序列
    save_dir = 'C:\\Users\\hua\\Desktop\\audio_segmentation\\OpenBMAT\\dataset/'
    split_audio(audio_dir, output_dir + "train/", save_dir + "train/", 3, overlap=0.5)
    split_audio(audio_dir, output_dir + "valid/", save_dir + "valid/", 3, overlap=0.5)
    split_audio(audio_dir, output_dir + "test/", save_dir + "test/", 3, overlap=0.5)
