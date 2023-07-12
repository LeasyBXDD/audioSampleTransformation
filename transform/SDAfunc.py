import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display

signal, sr = librosa.load("../content/data/wav48/p225/p225_001.wav")
y = signal

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
M = librosa.power_to_db(S, ref=np.max)

M1 = M2 = M
it = iter([M1, M2])

name1 = "Augmented 1"  # Define name1 variable
name2 = "Augmented 2"  # Define name2 variable


def SDA_view():
    fig = plt.figure(figsize=(10, 3))  # ,dpi=300
    gs = gridspec.GridSpec(nrows=1, ncols=3)

    ax0 = fig.add_subplot(gs[0, 0])
    librosa.display.specshow(
        librosa.core.amplitude_to_db(M, ref=np.max), y_axis="log", x_axis="time", ax=ax0
    )
    ax0.set_title("Original", fontsize=16)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    librosa.display.specshow(
        librosa.core.amplitude_to_db(M1, ref=np.max),
        y_axis="log",
        x_axis="time",
        ax=ax1,
    )
    ax1.set_title(name1, fontsize=16)  # Use name1 variable
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    batch = next(it)
    librosa.display.specshow(
        librosa.core.amplitude_to_db(M2, ref=np.max),
        y_axis="log",
        x_axis="time",
        ax=ax2,
    )
    ax2.set_title(name2, fontsize=16)
    ax2.axis("off")

    plt.show()
