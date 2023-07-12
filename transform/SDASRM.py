from audiomentations import SpecCompose, SpecChannelShuffle, SpecFrequencyMask
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
from SDAfunc import SDA_view, signal, sr


fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(nrows=2, ncols=3, height_ratios=[1, 1])

M = librosa.feature.melspectrogram(y=signal, sr=sr, power=1)

augment = SpecCompose([SpecFrequencyMask(p=1)])
M1 = augment(M)
name1 = 'Frequency'

M2 = augment(M.transpose()).transpose()  # simulating the time mask effect
name2 = 'Time'

SDA_view()
