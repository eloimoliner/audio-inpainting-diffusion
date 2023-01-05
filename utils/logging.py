import os
import torch 
import time
import numpy as np
#import torchaudio
import plotly.express as px
import soundfile as sf
#import plotly.graph_objects as go
import pandas as pd
import plotly
import scipy.signal as sig
import plotly.graph_objects as go

"""
Logging related functions that I wrote for my own use
This is quite a mess, but I'm too lazy to clean it up
"""
#from src.CQT_nsgt import CQT_cpx

def do_stft(noisy, clean=None, win_size=2048, hop_size=512, device="cpu", DC=True):
    """
        applies the stft, this an ugly old function, but I'm using it for logging and I'm to lazy to modify it
    """
    
    #window_fn = tf.signal.hamming_window

    #win_size=args.stft.win_size
    #hop_size=args.stft.hop_size
    window=torch.hamming_window(window_length=win_size)
    window=window.to(noisy.device)
    noisy=torch.cat((noisy, torch.zeros(noisy.shape[0],win_size).to(noisy.device)),1)
    stft_signal_noisy=torch.stft(noisy, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
    stft_signal_noisy=stft_signal_noisy.permute(0,3,2,1)
    #stft_signal_noisy=tf.signal.stft(noisy,frame_length=win_size, window_fn=window_fn, frame_step=hop_size)
    #stft_noisy_stacked=tf.stack( values=[tf.math.real(stft_signal_noisy), tf.math.imag(stft_signal_noisy)], axis=-1)
    
    if clean!=None:

       # stft_signal_clean=tf.signal.stft(clean,frame_length=win_size, window_fn=window_fn, frame_step=hop_size)
        clean=torch.cat((clean, torch.zeros(clean.shape[0],win_size).to(device)),1)
        stft_signal_clean=torch.stft(clean, win_size, hop_length=hop_size,window=window, center=False,return_complex=False)
        stft_signal_clean=stft_signal_clean.permute(0,3,2,1)
        #stft_clean_stacked=tf.stack( values=[tf.math.real(stft_signal_clean), tf.math.imag(stft_signal_clean)], axis=-1)


        if DC:
            return stft_signal_noisy, stft_signal_clean
        else:
            return stft_signal_noisy[...,1:], stft_signal_clean[...,1:]
    else:

        if DC:
            return stft_signal_noisy
        else:
            return stft_signal_noisy[...,1:]

def plot_norms(path, normsscores, normsguides, t, name):
    values=t.cpu().numpy()
     
    df=pd.DataFrame.from_dict(
                {"sigma": values[0:-1], "score": normsscores.cpu().numpy(), "guidance": normsguides.cpu().numpy()
                }
                )

    fig= px.line(df, x="sigma", y=["score", "guidance"],log_x=True,  log_y=True, markers=True)


    path_to_plotly_html = path+"/"+name+".html"
    
    fig.write_html(path_to_plotly_html, auto_play = False)
    return fig
    

def plot_spectral_analysis_sampling( avgspecNF,avgspecDEN, ts, fs=22050, nfft=1024):
    T,F=avgspecNF.shape
    f=torch.arange(0, F)*fs/nfft
    f=f.unsqueeze(1).repeat(1,T).view(-1)
    avgspecNF=avgspecNF.permute(1,0).reshape(-1)
    avgspecDEN=avgspecDEN.permute(1,0).reshape(-1)
    ts=ts.cpu().numpy().tolist()
    ts=513*ts
    #ts=np.repeat(ts,513)
    print(f.shape, avgspecDEN.shape, avgspecNF.shape, len(ts))
    df=pd.DataFrame.from_dict(
        {"f": f,  "noisy": avgspecNF, "denoised":  avgspecDEN, "sigma":ts
        }
        )

    fig= px.line(df, x="f", y=["noisy", "denoised"],  animation_frame="sigma", log_x=False,log_y=False,  markers=False)
    return fig

def plot_spectral_analysis(avgspecY, avgspecNF,avgspecDEN, ts, fs=22050, nfft=1024):
    T,F=avgspecNF.shape
    f=torch.arange(0, F)*fs/nfft
    f=f.unsqueeze(1).repeat(1,T).view(-1)
    avgspecY=avgspecY.squeeze(0).unsqueeze(1).repeat(1,T).view(-1)
    avgspecNF=avgspecNF.permute(1,0).reshape(-1)
    avgspecDEN=avgspecDEN.permute(1,0).reshape(-1)
    ts=ts.cpu().numpy().tolist()
    ts=513*ts
    #ts=np.repeat(ts,513)
    print(f.shape, avgspecY.shape, avgspecNF.shape, len(ts))
    df=pd.DataFrame.from_dict(
        {"f": f, "y": avgspecY, "noisy": avgspecNF, "denoised":  avgspecDEN, "sigma":ts
        }
        )

    fig= px.line(df, x="f", y=["y", "noisy", "denoised"],  animation_frame="sigma", log_x=False,log_y=False,  markers=False)
    return fig

def plot_loss_by_sigma_test_snr(average_snr, average_snr_out, t):
    #write a fancy plot to log in wandb
    values=np.array(t)
    df=pd.DataFrame.from_dict(
                {"sigma": values, "SNR": average_snr, "SNR_denoised": average_snr_out
                }
                )

    fig= px.line(df, x="sigma", y=["SNR", "SNR_denoised"],log_x=True,  markers=True)
    return fig
    



# Create and style traces

def plot_loss_by_sigma(sigma_means, sigma_stds, sigma_bins):
    df=pd.DataFrame.from_dict(
                {"sigma": sigma_bins, "loss": sigma_means, "std": sigma_stds
                }
                )

    fig= error_line('bar', data_frame=df, x="sigma", y="loss", error_y="std", log_x=True,  markers=True, range_y=[0, 2])
    
    return fig

def plot_loss_by_sigma_and_freq(sigma_freq_means, sigma_freq_stds, sigma_bins, freq_bins):
    df=pd.DataFrame.from_dict(
                {"sigma": np.tile(sigma_bins, len(freq_bins))}) 
    names=[]
    means=[]
    stds=[]
    for i in range(len(freq_bins)):
        name=[str(freq_bins[i])+"Hz" for j in range(len(sigma_bins))]
        names.extend(name)
        means.extend(sigma_freq_means[i])
        stds.extend(sigma_freq_stds[i])
    df['freq']=names
    df['means']=means
    df['stds']=stds
    print(df)


    #basically, we want to plot the loss as a function of sigma and freq. We do it by plotting a line for each freq.
    fig= error_line('bar', data_frame=df, x="sigma", y="means", error_y="stds", color="freq", log_x=True,  markers=True, range_y=[0, 2])
    
    return fig



def plot_melspectrogram(X, refr=1):
    X=X.squeeze(1) #??
    X=X.cpu().numpy()
    if refr==None:
        refr=np.max(np.abs(X))+1e-8
     
    S_db = 10*np.log10(np.abs(X)/refr)

    S_db=np.transpose(S_db, (0,2,1))
    S_db=np.flip(S_db, axis=1)

    for i in range(X.shape[0]): #iterate over batch size, shity way of ploting all the batched spectrograms
        o=S_db[i]
        if i==0:
             res=o
        else:
             res=np.concatenate((res,o), axis=1)
      
    fig=px.imshow( res,  zmin=-40, zmax=20)

    fig.update_layout(coloraxis_showscale=False)

    return fig

def plot_cpxspectrogram(X):
    X=X.squeeze(1)
    X=X.cpu().numpy()
    #Xre=X[...,0]
    #Xim=X[...,1]
    #X=np.sqrt(X[:,:,:,0]**2 + X[:,:,:,1]**2)
    #if refr==None:
    #    refr=np.max(np.abs(X))+1e-8
     
    #S_db = 10*np.log10(np.abs(X)/refr)

    #S_db=np.transpose(S_db, (0,2,1))
    #S_db=np.flip(S_db, axis=1)

    #for i in range(X.shape[0]): #iterate over batch size, shity way of ploting all the batched spectrograms
    #    o=S_db[i]
    #    if i==0:
    #         res=o
    #    else:
    #         res=np.concatenate((res,o), axis=1)
      
    fig=px.imshow(X, facet_col=3, animation_frame=0)

    fig.update_layout(coloraxis_showscale=False)

    return fig
def print_cuda_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reservedk
    print("memrylog",t,r,a,f)

def plot_spectrogram(X, refr=None):
    X=X.squeeze(1)
    X=X.cpu().numpy()
    X=np.sqrt(X[:,:,:,0]**2 + X[:,:,:,1]**2)
    if refr==None:
        refr=np.max(np.abs(X))+1e-8
     
    S_db = 10*np.log10(np.abs(X)/refr)

    S_db=np.transpose(S_db, (0,2,1))
    S_db=np.flip(S_db, axis=1)

    for i in range(X.shape[0]): #iterate over batch size, shity way of ploting all the batched spectrograms
        o=S_db[i]
        if i==0:
             res=o
        else:
             res=np.concatenate((res,o), axis=1)
      
    fig=px.imshow( res,  zmin=-40, zmax=20)

    fig.update_layout(coloraxis_showscale=False)

    return fig

def plot_mag_spectrogram(X, refr=None, path=None,name="spec"):
    #X=X.squeeze(1)
    X=X.cpu().numpy()
    #X=np.sqrt(X[:,:,:,0]**2 + X[:,:,:,1]**2)
    if refr==None:
        refr=np.max(np.abs(X))+1e-8
     
    S_db = 10*np.log10(np.abs(X)/refr)

    #S_db=np.transpose(S_db, (0,2,1))
    S_db=np.flip(S_db, axis=1)

    for i in range(X.shape[0]): #iterate over batch size, shity way of ploting all the batched spectrograms
        o=S_db[i]
        if i==0:
             res=o
        else:
             res=np.concatenate((res,o), axis=1)
      
    fig=px.imshow( res,  zmin=-40, zmax=20)

    fig.update_layout(coloraxis_showscale=False)

    path_to_plotly_png = path+"/"+name+".png"
    plotly.io.write_image(fig, path_to_plotly_png)
    
    return fig
def plot_spectrogram(X, refr=None):
    X=X.squeeze(1)
    X=X.cpu().numpy()
    X=np.sqrt(X[:,:,:,0]**2 + X[:,:,:,1]**2)
    if refr==None:
        refr=np.max(np.abs(X))+1e-8
     
    S_db = 10*np.log10(np.abs(X)/refr)

    S_db=np.transpose(S_db, (0,2,1))
    S_db=np.flip(S_db, axis=1)

    for i in range(X.shape[0]): #iterate over batch size, shity way of ploting all the batched spectrograms
        o=S_db[i]
        if i==0:
             res=o
        else:
             res=np.concatenate((res,o), axis=1)
      
    fig=px.imshow( res,  zmin=-40, zmax=20)

    fig.update_layout(coloraxis_showscale=False)

    return fig

def write_audio_file(x, sr, string: str, path='tmp', stereo=False):
    if not(os.path.exists(path)): 
        os.makedirs(path)
      
    path=os.path.join(path,string+".wav")
    if stereo:
        '''
        x has shape (B,2,T)
        '''
        x=x.permute(0,2,1) #B,T,2
        x=x.flatten(0,1) #B*T,2
        x=x.cpu().numpy()
        if np.abs(np.max(x))>=1:
            #normalize to avoid clipping
            x=x/np.abs(np.max(x))
    else:
        x=x.flatten()
        x=x.unsqueeze(1)
        x=x.cpu().numpy()
        if np.abs(np.max(x))>=1:
            #normalize to avoid clipping
            x=x/np.abs(np.max(x))
    sf.write(path,x,sr)
    return path


def plot_cpxCQT_from_raw_audio(x, args, refr=None ):
    #shape of input spectrogram:  (     ,T,F, )
    fmax=args.sample_rate/2
    fmin=fmax/(2**args.cqt.numocts)
    fbins=int(args.cqt.binsoct*args.cqt.numocts) 
    device=x.device
    CQTransform=CQT_cpx(fmin,fbins, args.sample_rate, args.audio_len, device=device, split_0_nyq=False)

    x=x
    X=CQTransform.fwd(x)
    return plot_cpxspectrogram(X)

def plot_CQT_from_raw_audio(x, args, refr=None ):
    #shape of input spectrogram:  (     ,T,F, )
    fmax=args.sample_rate/2
    fmin=fmax/(2**args.cqt.numocts)
    fbins=int(args.cqt.binsoct*args.cqt.numocts) 
    device=x.device
    CQTransform=CQT_cpx(fmin,fbins, args.sample_rate, args.audio_len, device=device, split_0_nyq=False)

    refr=3
    x=x
    X=CQTransform.fwd(x)
    return plot_spectrogram(X, refr)

def get_spectrogram_from_raw_audio(x, stft, refr=1):
    X=do_stft(x, win_size=stft.win_size, hop_size=stft.hop_size)
    X=X.permute(0,2,3,1)

    X=X.squeeze(1)
    #X=X.cpu().numpy()
    X=torch.sqrt(X[:,:,:,0]**2 + X[:,:,:,1]**2)

    #if refr==None:
    #    refr=np.max(np.abs(X))+1e-8
     
    S_db = 10*torch.log10(torch.abs(X)/refr)

    S_db=S_db.permute(0,2,1)
    #np.transpose(S_db, (0,2,1))
    S_db=torch.flip(S_db, [1])

    for i in range(X.shape[0]): #iterate over batch size, shity way of ploting all the batched spectrograms
        o=S_db[i]
        if i==0:
             res=o
        else:
             res=torch.cat((res,o), 1)
    return res

def downsample2d(inputArray, kernelSize):
    """This function downsamples a 2d numpy array by convolving with a flat
    kernel and then sub-sampling the resulting array.
    A kernel size of 2 means convolution with a 2x2 array [[1, 1], [1, 1]] and
    a resulting downsampling of 2-fold.
    :param: inputArray: 2d numpy array
    :param: kernelSize: integer
    """
    average_kernel = np.ones((kernelSize,kernelSize))

    blurred_array = sig.convolve2d(inputArray, average_kernel, mode='same')
    downsampled_array = blurred_array[::kernelSize,::kernelSize]
    return downsampled_array


def diffusion_CQT_animation(path, x ,t,  args, refr=1, name="animation_diffusion", resample_factor=1 ):
    """
        Utility for creating an animation of the cqt diffusion process
    """
    #shape of input spectrograms:  (Nsteps,B,Time,Freq)
    #print(noisy.shape)
    Nsteps=x.shape[0]
    numsteps=10
    tt=torch.linspace(0, Nsteps-1, numsteps)
    i_s=[]
    allX=None
    device=x.device

    fmax=args.sample_rate/2
    fmin=fmax/(2**args.cqt.numocts)
    fbins=int(args.cqt.binsoct*args.cqt.numocts) 
    CQTransform=CQT_cpx(fmin,fbins, args.sample_rate, args.audio_len, device=device, split_0_nyq=False)

    for i in tt:
        i=int(torch.floor(i))
        i_s.append(i)
        xx=x[i]
        X=CQTransform.fwd(xx)

        X=torch.sqrt(X[...,0]**2 + X[...,1]**2)

        S_db = 10*torch.log10(torch.abs(X)/refr)
        S_db = S_db[:,1:-1,1:-1]
       
        S_db=S_db.permute(0,2,1)
        #np.transpose(S_db, (0,2,1))
        S_db=torch.flip(S_db, [1])
        S_db=S_db.unsqueeze(0)
        S_db=torch.nn.functional.interpolate(S_db, size= (S_db.shape[2]//resample_factor, S_db.shape[3]//resample_factor), mode="bilinear")
        S_db=S_db.squeeze(0)
        S_db=S_db.cpu().numpy()
        #S_db=S_db.squeeze(0)        

        if i==0:
             allX=S_db
        else:
             allX=np.concatenate((allX,S_db), 0)
    
    #fig=px.imshow(S_db, animation_frame=0, facet_col=3, binary_compression_level=0) #I need to incorporate t here!!!
    fig=px.imshow(allX, animation_frame=0,  zmin=-40, zmax=10, binary_compression_level=0) #I need to incorporate t here!!!

    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    t=t[i_s].cpu().numpy()
       
    assert len(t)==len(fig.frames), print(len(t), len(fig.frames))
    #print(fig)
    for i, f in enumerate(fig.layout.sliders[0].steps):
        #hacky way of changing the axis values
        #f.args[0]=[str(t[i])]
        f.label=str(t[i])

    path_to_plotly_html = path+"/"+name+".html"
    
    fig.write_html(path_to_plotly_html, auto_play = False)


    return fig

def diffusion_joint_filter_animation(path, score, grads , f, t,  refr=1, name="animation_diffusion", NT=15 ):
    '''
    plot an animation of the reverse diffusion process of filters
    args:
        path: path to save the animation
        x: filters (Nsteps, F)
        f: frequencies (F)
        t: timesteps (sigma)
        name: name of the animation
    '''
    #shape of input spectrograms:  (Nsteps,B,Time,Freq)
    #print(noisy.shape)
    Nsteps=score.shape[0]
    #numsteps=min(Nsteps,NT) #hardcoded, I'll probably need more!

    #tt=torch.linspace(0, Nsteps-1, numsteps)
    #i_s=[]
    #for i in tt:
    #    i=int(torch.floor(i))
    #    i_s.append(i)
      
    print(f.shape, score.shape, grads.shape, t.shape)  #19, (100,19)
    #fig=px.line(x=f.unsqueeze(0).numpy(),y=x, animation_frame=0) #I need to incorporate t here!!!
    sigma=t
    s=score.squeeze(1)# (100,19)
    g=grads.squeeze(1)# (100,19)
    f=f# (19,)
    f=f.unsqueeze(0).expand(s.shape[0], -1).reshape(-1)
    sigma=sigma.unsqueeze(-1).expand(-1, s.shape[1]).reshape(-1)
    s=s.reshape(-1)
    g=g.reshape(-1)
    max_score=torch.max(s).item()
    df=pd.DataFrame(
        {
            "f": f.cpu().numpy(),
            "score": s.cpu().numpy(),
            "grads": g.cpu().numpy(),
            "sigma": sigma.cpu().numpy()
        }
    )
    fig=px.line(df, x="f",y=["score","grads"], animation_frame="sigma", log_x=True) #I need to incorporate t here!!!
    #set the range of th y axis
    fig.update_yaxes(range=[-max_score,max_score])
    #t=t[i_s].cpu().numpy()
    #t=t.cpu().numpy()
       
    #assert len(t)==len(fig.frames), print(len(t), len(fig.frames))
    #print(fig)
    #for i, f in enumerate(fig.layout.sliders[0].steps):
        #hacky way of changing the axis values
        #f.args[0]=[str(t[i])]
    #    f.label=str(t[i])

    path_to_plotly_html = path+"/"+name+".html"
    
    fig.write_html(path_to_plotly_html, auto_play = False)

    return fig

def diffusion_filter_animation(path, x , f, t,  refr=1, name="animation_diffusion", NT=15 ):
    '''
    plot an animation of the reverse diffusion process of filters
    args:
        path: path to save the animation
        x: filters (Nsteps, F)
        f: frequencies (F)
        t: timesteps (sigma)
        name: name of the animation
    '''
    #shape of input spectrograms:  (Nsteps,B,Time,Freq)
    #print(noisy.shape)
    Nsteps=x.shape[0]
    #numsteps=min(Nsteps,NT) #hardcoded, I'll probably need more!

    #tt=torch.linspace(0, Nsteps-1, numsteps)
    #i_s=[]
    #for i in tt:
    #    i=int(torch.floor(i))
    #    i_s.append(i)
      
    print(f.shape, x.shape, t.shape)  #19, (100,19)
    #fig=px.line(x=f.unsqueeze(0).numpy(),y=x, animation_frame=0) #I need to incorporate t here!!!
    sigma=t
    x=x.squeeze(1)# (100,19)
    f=f# (19,)
    f=f.unsqueeze(0).expand(x.shape[0], -1).reshape(-1)
    sigma=sigma.unsqueeze(-1).expand(-1, x.shape[1]).reshape(-1)
    x=x.reshape(-1)
    df=pd.DataFrame(
        {
            "f": f.cpu().numpy(),
            "x": x.cpu().numpy(),
            "sigma": sigma.cpu().numpy()
        }
    )
    fig=px.line(df, x="f",y="x", animation_frame="sigma", log_x=True) #I need to incorporate t here!!!
    #t=t[i_s].cpu().numpy()
    #t=t.cpu().numpy()
       
    #assert len(t)==len(fig.frames), print(len(t), len(fig.frames))
    #print(fig)
    #for i, f in enumerate(fig.layout.sliders[0].steps):
        #hacky way of changing the axis values
        #f.args[0]=[str(t[i])]
    #    f.label=str(t[i])

    path_to_plotly_html = path+"/"+name+".html"
    
    fig.write_html(path_to_plotly_html, auto_play = False)

    return fig

def diffusion_spec_animation(path, x ,t,  stft, refr=1, name="animation_diffusion",NT=15 ):
    '''
    plot an animation of the reverse diffusion process of filters
    args:
        path: path to save the animation
        x: input audio (N,T)
        t: timesteps (sigma)
        name: name of the animation
    '''
    #print(noisy.shape)
    Nsteps=x.shape[0]
    numsteps=min(Nsteps,NT) #hardcoded, I'll probably need more!
    tt=torch.linspace(0, Nsteps-1, numsteps)
    i_s=[]
    allX=None
    for i in tt:
        i=int(torch.floor(i))
        i_s.append(i)
        X=get_spectrogram_from_raw_audio(x[i],stft, refr)
        X=X.unsqueeze(0)
        if allX==None:
             allX=X
        else:
             allX=torch.cat((allX,X), 0)

      
    allX=allX.cpu().numpy()
    fig=px.imshow(allX, animation_frame=0,  zmin=-40, zmax=20) #I need to incorporate t here!!!

    fig.update_layout(coloraxis_showscale=False)
    
    t=t[i_s].cpu().numpy()
       
    assert len(t)==len(fig.frames), print(len(t), len(fig.frames))
    #print(fig)
    for i, f in enumerate(fig.layout.sliders[0].steps):
        #hacky way of changing the axis values
        #f.args[0]=[str(t[i])]
        f.label=str(t[i])

    path_to_plotly_html = path+"/"+name+".html"
    
    fig.write_html(path_to_plotly_html, auto_play = False)


    return fig

def plot_spectrogram_from_raw_audio(x, stft, refr=None ):
    #shape of input spectrogram:  (     ,T,F, )
    refr=3
    x=x
    X=do_stft(x, win_size=stft.win_size, hop_size=stft.hop_size)
    X=X.permute(0,2,3,1)
    return plot_spectrogram(X, refr)

def plot_spectrogram_from_cpxspec(X, refr=None):
    #shape of input spectrogram:  (     ,T,F, )
    return plot_spectrogram(X, refr)


def error_line(error_y_mode='band', **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {'bar','band','bars','bands',None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar','bars',None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band','bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = False,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig


def plot_filters( x, freqs):
    '''
    This function plots a batch of lines using plotly
    args:
        x: (B, F)
    '''
    fig = px.line(x=freqs, y=x[0,:], log_x=True)

    for i in range(1,x.shape[0]):
        fig.add_trace(go.Scatter(x=freqs, y=x[i,:]))
    return fig

def plot_batch_of_lines( x, freqs):
    '''
    This function plots a batch of lines using plotly
    args:
        x: (B, F)
    '''
    fig = px.line(x=freqs, y=x[0,:], log_x=True)

    for i in range(1,x.shape[0]):
        fig.add_trace(go.Scatter(x=freqs, y=x[i,:]))
    return fig
