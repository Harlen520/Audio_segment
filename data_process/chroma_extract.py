import os
from tqdm import tqdm
import subprocess

log_dir = 'E:\\audio_segmentation\\chroma_features/error'

def extract_chroma(audio_path, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    error_path = os.path.join(log_dir, os.path.split(output_dir)[-1] + ".txt")
    f = open(error_path, "w", encoding="utf-8")
    fnames=os.listdir(audio_path)   # 生成所有音频文件文件名的列表
    for fname in tqdm(fnames):
        audio_path_file = os.path.join(audio_path, fname)
        output_path = os.path.join(output_dir, fname[:-4] + '.txt')
        base = 'E: && cd E:/audio_segmentation/OpenSMILE/opensmile-3.0.1-win-x64/bin/ && SMILExtract -C E:/audio_segmentation/OpenSMILE/opensmile-3.0.1-win-x64/config/chroma/chroma_fft.conf -I '
        cmd = base + audio_path_file + ' -O ' + output_path
        # &&连续执行；C: 进入C盘内；进入opensmile中要执行的文件的目录下；执行文件 -C 配置文件 -I 语音文件 -O 输出到指定文件
        # os.system(cmd)
        try:
            subprocess.call(cmd, shell=True)
        except:
            f.write(fname + "\n")
    f.close()

if __name__ == "__main__":
    audio_dir = 'E:\\audio_segmentation\\OpenBMAT\\dataset/'  # 音频文件所在目录
    output_dir='E:\\audio_segmentation\\chroma_features/'   # 特征文件输出目录
    extract_chroma(audio_dir + "train", output_dir + "train")
    extract_chroma(audio_dir + "valid", output_dir + "valid")
    extract_chroma(audio_dir + "test", output_dir + "test")
    print('done~')
  
