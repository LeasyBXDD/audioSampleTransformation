import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
from audiomentations import SpecCompose, SpecChannelShuffle, SpecFrequencyMask

from SDAfunc import SDA_view, signal, sr

fig = plt.figure(figsize=(10, 6))
# ,dpi=300
gs = gridspec.GridSpec(nrows=2, ncols=3, height_ratios=[1, 1])

M = librosa.feature.melspectrogram(y=signal, sr=sr, power=1)

augment = SpecCompose([SpecChannelShuffle(p=1)])
M1 = augment(np.array([M, M]))[0, :, :]
name1 = "C1"

M2 = augment(np.array([M, M]))[1, :, :]
name2 = "C2"

SDA_view()
