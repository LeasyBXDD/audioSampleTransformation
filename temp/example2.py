import librosa
import numpy as np
import soundfile as sf

# addNoise
# 加载两个音频文件
audio_data1, sample_rate1 = librosa.load('real.wav', sr=None, mono=True)
audio_data2, sample_rate2 = librosa.load('noise2.wav', sr=None, mono=True)

sin_signal = np.resize(audio_data2, len(audio_data1))
# 将两个音频文件混合在一起，设置音量比例为 2:1
mixed_signal = 4 * audio_data1 + sin_signal

# 保存混合后的音频文件
sf.write('2noise4.wav', mixed_signal, sample_rate1)