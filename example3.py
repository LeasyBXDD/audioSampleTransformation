import librosa
import numpy as np
import soundfile as sf

# shitf  平移
from audiomentations import Shift, Compose
from numpy.testing import assert_almost_equal

samples, sample_rate = librosa.load('real.wav', sr=None, mono=True)
# samples = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
# sample_rate = 16000

forward_augmenter = Compose([Shift(min_fraction=0.5, max_fraction=0.5, p=1.0)])
forward_shifted_samples = forward_augmenter(
    samples=samples, sample_rate=sample_rate
)
# assert_almost_equal(
#     forward_shifted_samples, np.array([0.25, 0.125, 1.0, 0.5], dtype=np.float32)
# )
assert forward_shifted_samples.dtype == np.float32
# assert len(forward_shifted_samples) == 4

backward_augmenter = Compose(
    [Shift(min_fraction=-0.25, max_fraction=-0.25, p=1.0)]
)
backward_shifted_samples = backward_augmenter(
    samples=samples, sample_rate=sample_rate
)
# assert_almost_equal(
#     backward_shifted_samples,
#     np.array([0.5, 0.25, 0.125, 1.0], dtype=np.float32),
# )
assert backward_shifted_samples.dtype == np.float32
# assert len(forward_shifted_samples) == 4

sf.write('3noiseV1.wav', forward_shifted_samples, sample_rate)
sf.write('3noiseV2.wav', backward_shifted_samples, sample_rate)