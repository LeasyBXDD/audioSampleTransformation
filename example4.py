import librosa
import soundfile as sf
from audiomentations import TimeStretch
# 拉伸
# 核心函数librosa.effects.time_stretch(y, rate)
# ，y 表示输入的音频数据，可以是一个 Numpy 数组或者一个文件名字符串；rate 表示时间拉伸的比例因子，为一个正实数。
# 如果 rate 大于 1，则表示时间轴被拉伸，音频文件的时长变长；
# 如果 rate 小于 1，则表示时间轴被压缩，音频文件的时长变短。

samples, sample_rate = librosa.load('real.wav', sr=None, mono=True)
# 会随机生成一个时间拉伸比例因子 rate，其取值范围为 min_rate 和 max_rate 之间
transform = TimeStretch(
    min_rate=0.8,
    max_rate=1.25,
    leave_length_unchanged=True,
    p=1.0
)

augmented_sound = transform(samples, sample_rate=16000)
sf.write("4noiseV1.wav", augmented_sound, sample_rate)
