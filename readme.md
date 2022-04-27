## Audio Segmentation

### 一、Dataset：OpenBMAT

数据来自论文《Open Broadcast Media Audio from TV: A Dataset of TV Broadcast Audio with Relative Music Loudness Annotations  》

save at ./data/OpenBMAT 

可用作 6分类、三分类、二分类 音频分类任务

获取方式：需发邮件向论文作者申请。

### 二、运行环境

nvidia driver version 440.33.01

cuda 10.2

cudnn 7.6.0 

paddlepaddle-gpu 2.2.2

paddlenlp==2.2.5

pandas

librosa

soundfile

tqdm

### 三、数据预处理

（1）特征提取之前先对音频原始数据进行数据预处理，切割出每个类别的音频片段和提取相应的类别标签。

（2）因为我们的输入是3秒序列，所以继续切割为3秒的音频文件

**运行./data_process/prepare_data.py** 

### 四、特征提取

对音频数据做两种特征提取，分别是：对数梅尔频谱（log Mel filter bank energies ） 和 chroma色度特征（chroma features   ）

（1）对数梅尔频谱

使用librosa库提供的接口提取

librosa.feature.melspectrogram()

（2）chroma色度特征

note：chroma色度特征需提前提取保存，训练前再加载与mel特征拼接。

使用OpenSMILE工具提取音频数据的chroma色度特征。由于OpenSMILE linux版本安装较为麻烦，而OpenSMILE window版本下载回来解压即用（免安装），所以本次我们使用window版本。

OpenSMILE下载链接：https://github.com/audeering/opensmile/releases（window、linux、Mac三种版本可下载）

OpenSMILE使用教程可结合以下两篇文章

https://www.it610.com/article/1290144592229900288.htm

https://zhuanlan.zhihu.com/p/69170521

**运行./data_process/chroma_extract.py** 

### 五、模型

模型代码在model.py文件下。采用两层双向RNN结构对音频特征（mel特征与色度特征拼接）进行融合，然后取最后时刻的rnn输出，再经过一层线性层+softmax预测输出。

rnn可选择 LSTM或 GRU，这两种都是RNN的变体，对普通RNN的改进。

训练和评估文件：run.py

需要分布式训练时：sh run.sh





