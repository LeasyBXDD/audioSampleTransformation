import numpy as np
import librosa
import soundfile as sf

# from audiomentations import AddBackgroundNoise, Compose, Reverse


# class TestAddBackgroundNoise:
#     def test_add_background_noise(self):
#         # 创建一个长度为22500的正弦波信号samples和一个采样率为44100的样本率sample_rate。
#         samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 22500)).astype(np.float32)
#         sample_rate = 44100
#         # 创建一个AddBackgroundNoise类的实例augmenter
#         augmenter = AddBackgroundNoise(
#             # 背景噪声文件所在的路径；
#             sounds_path=os.path.join("noise1.mp3", "background_noises"),
#             # 背景噪声和信号之间的最小信噪比；
#             min_snr_in_db=15,
#             # 背景噪声和信号之间的最大信噪比；
#             max_snr_in_db=35,
#             # 执行增强的概率。
#             p=1.0,
#         )
#         # 调用augmenter的__call__方法，向其传递samples和sample_rate作为输入，并将输出结果存储在samples_out中。
#         samples_out = augmenter(samples=samples, sample_rate=sample_rate)
#         # 使用numpy库中的allclose函数检查samples和samples_out是否相同，并使用assert 语句确保它们不相同，并且samples_out的数据类型为np.float32。
#         assert not np.allclose(samples, samples_out)
#         assert samples_out.dtype == np.float32
#
#     def test_add_background_noise_when_noise_sound_is_too_short(self):
#         sample_rate = 44100
#         samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 14 * sample_rate)).astype(
#             np.float32
#         )
#         augmenter = AddBackgroundNoise(
#             sounds_path=os.path.join(DEMO_DIR, "background_noises"),
#             min_snr_in_db=15,
#             max_snr_in_db=35,
#             p=1.0,
#         )
#         samples_out = augmenter(samples=samples, sample_rate=sample_rate)
#         assert not np.allclose(samples, samples_out)
#         assert samples_out.dtype == np.float32
#
#     def test_try_add_almost_silent_file(self):
#         samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 30000)).astype(np.float32)
#         sample_rate = 48000
#         augmenter = AddBackgroundNoise(
#             sounds_path=os.path.join(DEMO_DIR, "almost_silent"),
#             min_snr_in_db=15,
#             max_snr_in_db=35,
#             p=1.0,
#         )
#         samples_out = augmenter(samples=samples, sample_rate=sample_rate)
#         assert not np.allclose(samples, samples_out)
#         assert samples_out.dtype == np.float32
#
#     def test_try_add_digital_silence(self):
#         samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 40000)).astype(np.float32)
#         sample_rate = 48000
#         augmenter = Compose(
#             [
#                 AddBackgroundNoise(
#                     sounds_path=os.path.join(DEMO_DIR, "digital_silence"),
#                     min_snr_in_db=15,
#                     max_snr_in_db=35,
#                     p=1.0,
#                 )
#             ]
#         )
#
#         with warnings.catch_warnings(record=True) as w:
#             # Cause all warnings to always be triggered.
#             warnings.simplefilter("always")
#             samples_out = augmenter(samples=samples, sample_rate=sample_rate)
#
#             assert "is too silent to be added as noise" in str(w[-1].message)
#
#         assert np.allclose(samples, samples_out)
#         assert samples_out.dtype == np.float32
#
#     def test_serialize_parameters(self):
#         transform = AddBackgroundNoise(
#             sounds_path=os.path.join(DEMO_DIR, "background_noises"), p=1.0
#         )
#         samples = np.random.normal(0, 1, size=1024).astype(np.float32)
#         transform.randomize_parameters(samples, sample_rate=44100)
#         json.dumps(transform.serialize_parameters())
#
#     def test_picklability(self):
#         transform = AddBackgroundNoise(
#             sounds_path=os.path.join(DEMO_DIR, "background_noises"), p=1.0
#         )
#         pickled = pickle.dumps(transform)
#         unpickled = pickle.loads(pickled)
#         assert transform.sound_file_paths == unpickled.sound_file_paths
#
#     def test_absolute_option(self):
#         samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 22500)).astype(np.float32)
#         sample_rate = 44100
#         augmenter = AddBackgroundNoise(
#             sounds_path=os.path.join(DEMO_DIR, "background_noises"),
#             noise_rms="absolute",
#             p=1.0,
#         )
#         samples_out = augmenter(samples=samples, sample_rate=sample_rate)
#         assert not np.allclose(samples, samples_out)
#
#     def test_noise_transform(self):
#         np.random.seed(3650)
#         random.seed(3650)
#         samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 22500)).astype(np.float32)
#         sample_rate = 44100
#         augmenter = AddBackgroundNoise(
#             sounds_path=os.path.join(DEMO_DIR, "background_noises"),
#             min_snr_in_db=3,
#             max_snr_in_db=6,
#             p=1.0,
#         )
#         samples_out_without_transform = augmenter(
#             samples=samples, sample_rate=sample_rate
#         )
#         augmenter.freeze_parameters()
#         augmenter.noise_transform = Reverse()
#         samples_out_with_transform = augmenter(samples=samples, sample_rate=sample_rate)
#
#         assert not np.allclose(
#             samples_out_without_transform, samples_out_with_transform
#         )


def main():
    from audiomentations import AddBackgroundNoise, PolarityInversion
    # method1  只使用正弦波信号
    # 创建一个长度为22500的正弦波信号samples和一个采样率为44100的样本率sample_rate。
    # samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 22500)).astype(np.float32)
    # samples = np.tile(samples, 3)
    # sample_rate = 16000

    # method2  使用正弦波信号和原文件叠加
    # 1.使用 Python 的音频库（如 librosa）或第三方库（如 pydub）从音频文件中加载音频数据。可以使用以下代码使用 librosa 从 WAV 文件中加载音频数据：
    # 加载音频文件
    audio_data, sample_rate = librosa.load('real.wav', sr=None, mono=True)
    # 将音频数据转换为浮点数类型
    # 在这里，audio_data变量将包含加载的音频数据，sample_rate变量将包含音频文件的采样率
    audio_data = audio_data.astype(float)

    # 2.使用numpy库生成一个正弦波信号，使其具有与音频数据相同的采样率和持续时间。可以使用以下代码生成类似于np.sin(np.linspace(0, 440 * 2 * np.pi, 22500)).astype(np.float32)
    # 的正弦波信号：
    # 设置持续时间为音频文件的持续时间
    duration = len(audio_data) / sample_rate
    # 计算采样点数量
    num_samples = int(duration * 22500)
    # 生成采样点的时间序列，从 0 秒到持续时间
    time_points = np.linspace(0, duration, num_samples, endpoint=False)
    # 计算正弦波的频率
    frequency = 440.0
    # 生成正弦波信号
    sin_signal = np.sin(2 * np.pi * frequency * time_points).astype(np.float32)
    # 调整正弦波信号的长度
    sin_signal = np.resize(sin_signal, len(audio_data))

    #3.将生成的正弦波信号与音频数据相乘，以生成一个类似于 np.sin(np.linspace(0, 440 * 2 * np.pi, 22500)).astype(np.float32) 的信号。可以使用以下代码将两个信号相乘：
    # 将正弦波信号与音频数据相乘  也可以相加，但是就是单纯的叠加，正弦信号的声音没有去掉
    mixed_signal = sin_signal * audio_data
    #在这里，mixed_signal 变量将包含相乘后的信号，它将具有类似于 np.sin(np.linspace(0, 440 * 2 * np.pi, 22500)).astype(np.float32) 的格式，
    # 但是它将包含音频数据中的实际音频信息。需要注意的是，这个信号只是一个简单的混合信号，它并不是一个真正的音频文件，无法直接保存为 WAV、MP3 或其他音频格式

    #method3
    # 加载音频文件
    # audio_data, sample_rate = librosa.load('noise.wav', sr=None, mono=True)
    # # 将音频数据转换为 np.ndarray 类型的数组
    # samples = np.array(audio_data)

    # 创建一个AddBackgroundNoise类的实例transform
    transform = AddBackgroundNoise(
        # 背景噪声文件所在的路径；
        sounds_path="noise.wav",
        # 背景噪声和信号之间的最小信噪比；
        min_snr_in_db=3.0,
        # 背景噪声和信号之间的最大信噪比；
        max_snr_in_db=30.0,
        noise_transform=PolarityInversion(),
        # 执行增强的概率。
        p=1.0
    )
    # 调用augmenter的__call__方法，向其传递samples和sample_rate作为输入，并将输出结果存储在samples_out中。
    augmented_sound = transform(samples = mixed_signal, sample_rate=sample_rate)
    print("hahaha")
    assert not np.allclose(mixed_signal, augmented_sound)
    assert augmented_sound.dtype == np.float32
    # 使用numpy库中的allclose函数检查samples和samples_out是否相同，并使用assert 语句确保它们不相同，并且samples_out的数据类型为np.float32。
    sf.write("noiseV4.mp3", augmented_sound, sample_rate)

main()