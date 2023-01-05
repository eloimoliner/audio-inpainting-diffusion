import soundfile as sf
import torch
import numpy as np

def load_audio(name="test_dir/0.wav",Ls=65536):
    x, fs=sf.read(name)
    x=torch.Tensor(x)
    #x=x[1*fs:2*fs]
    Ls=65536
    x=x[0:Ls]
    return x, fs

def save_wav(x, fs=22050, filename="test.wav"):
    x=x.numpy()
    sf.write(filename, x, fs)

def plot_stft(x):
    NFFT=1024
    #hamming window
    window = torch.hann_window(NFFT)
    #apply STFT to x
    X = torch.stft(x, NFFT, hop_length=NFFT//2, win_length=NFFT, window=window, center=False, normalized=False, onesided=True)

    freqs=np.fft.rfftfreq(NFFT, 1/fs)

    X_abs=(X[...,0]**2+X[...,1]**2)**0.5
    #plot absolute value of STFT using px
    fig = px.imshow(20*np.log10(X_abs.numpy()+1e-8), labels=dict(x="Time", y="Frequency", color="Magnitude"))

    fig.show()