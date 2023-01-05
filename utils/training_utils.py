
import torch
import torchaudio
import numpy as np
import scipy.signal
class EMAWarmup:
    """Implements an EMA warmup using an inverse decay schedule.
    If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).
    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
        max_value (float): The maximum EMA decay rate. Default: 1.
        start_at (int): The epoch to start averaging at. Default: 0.
        last_epoch (int): The index of last epoch. Default: 0.
    """

    def __init__(self, inv_gamma=1., power=1., min_value=0., max_value=1., start_at=0,
                 last_epoch=0):
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.start_at = start_at
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the class as a :class:`dict`."""
        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """Loads the class's state.
        Args:
            state_dict (dict): scaler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_value(self):
        """Gets the current EMA decay rate."""
        epoch = max(0, self.last_epoch - self.start_at)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        return 0. if epoch < 0 else min(self.max_value, max(self.min_value, value))

    def step(self):
        """Updates the step count."""
        self.last_epoch += 1


#from https://github.com/csteinmetz1/auraloss/blob/main/auraloss/perceptual.py
class FIRFilter(torch.nn.Module):
    """FIR pre-emphasis filtering module.
    Args:
        filter_type (str): Shape of the desired FIR filter ("hp", "fd", "aw"). Default: "hp"
        coef (float): Coefficient value for the filter tap (only applicable for "hp" and "fd"). Default: 0.85
        ntaps (int): Number of FIR filter taps for constructing A-weighting filters. Default: 101
        plot (bool): Plot the magnitude respond of the filter. Default: False
    Based upon the perceptual loss pre-empahsis filters proposed by
    [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).
    A-weighting filter - "aw"
    First-order highpass - "hp"
    Folded differentiator - "fd"
    Note that the default coefficeint value of 0.85 is optimized for
    a sampling rate of 44.1 kHz, considering adjusting this value at differnt sampling rates.
    """

    def __init__(self, filter_type="hp", coef=0.85, fs=44100, ntaps=101, plot=False): 
        """Initilize FIR pre-emphasis filtering module."""
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps
        self.plot = plot

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, -coef, 0]).view(1, 1, -1)
        elif filter_type == "fd":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, 0, -coef]).view(1, 1, -1)
        elif filter_type == "aw":
            # Definition of analog A-weighting filter according to IEC/CD 1672.
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997

            NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
            DENs = np.polymul(
                [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
            )
            DENs = np.polymul(
                np.polymul(DENs, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2]
            )

            # convert analog filter to digital filter
            b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

            # compute the digital filter frequency response
            w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)

            # then we fit to 101 tap FIR filter with least squares
            taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)

            # now implement this digital FIR filter as a Conv1d layer
            self.fir = torch.nn.Conv1d(
                1, 1, kernel_size=ntaps, bias=False, padding=ntaps // 2
            )
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype("float32")).view(1, 1, -1)

    def forward(self, error):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: Filtered signal.
        """
        self.fir.weight.data=self.fir.weight.data.to(error.device)
        error=error.unsqueeze(1)
        error = torch.nn.functional.conv1d(
            error, self.fir.weight.data, padding=self.ntaps // 2
        )
        error=error.squeeze(1)
        return error

def resample_batch(audio, fs, fs_target, length_target):

        device=audio.device
        dtype=audio.dtype
        B=audio.shape[0]
        #if possible resampe in a batched way
        #check if all the fs are the same and equal to 44100
        if fs_target==22050:
            if (fs==44100).all():
                 audio=torchaudio.functional.resample(audio, 2,1)
                 return audio[:, 0:length_target] #trow away the last samples
            elif (fs==48000).all():
                 #approcimate resamppleint
                 audio=torchaudio.functional.resample(audio, 160*2,147)
                 return audio[:, 0:length_target]
            else:
                #if revious is unsuccesful bccause we have examples at 441000 and 48000 in the same batch,, just iterate over the batch
                proc_batch=torch.zeros((B,length_target), device=device)
                for i, (a, f_s) in enumerate(zip(audio, fs)): #I hope this shit wll not slow down everythingh
                    if f_s==44100:
                        #resample by 2
                        a=torchaudio.functional.resample(a, 2,1)
                    elif f_s==48000:
                        a=torchaudio.functional.resample(a, 160*2,147)
                    else:
                        print("WARNING, strange fs", f_s)
           
                    proc_batch[i]=a[0:length_target]
                    return proc_batch
        elif fs_target==44100:
            if (fs==44100).all():
                 return audio[:, 0:length_target] #trow away the last samples
            elif (fs==48000).all():
                 #approcimate resamppleint
                 audio=torchaudio.functional.resample(audio, 160,147)
                 return audio[:, 0:length_target]
            else:
                #if revious is unsuccesful bccause we have examples at 441000 and 48000 in the same batch,, just iterate over the batch
                proc_batch=torch.zeros((B,length_target), device=device)
                for i, (a, f_s) in enumerate(zip(audio, fs)): #I hope this shit wll not slow down everythingh
                    if f_s==44100:
                        #resample by 2
                        pass
                    elif f_s==48000:
                        a=torchaudio.functional.resample(a, 160,147)
                    else:
                        print("WARNING, strange fs", f_s)
           
                    proc_batch[i]=a[0:length_target] 
                    return proc_batch
        else:
            print(" resampling to fs_target", fs_target)
            if (fs==44100).all():
                 audio=torchaudio.functional.resample(audio, 44100, fs_target)
                 return audio[:, 0:length_target] #trow away the last samples
            elif (fs==48000).all():
                 #approcimate resamppleint
                 audio=torchaudio.functional.resample(audio, 48000,fs_target)
                 return audio[:, 0:length_target]
            else:
                #if revious is unsuccesful bccause we have examples at 441000 and 48000 in the same batch,, just iterate over the batch
                proc_batch=torch.zeros((B,length_target), device=device)
                for i, (a, f_s) in enumerate(zip(audio, fs)): #I hope this shit wll not slow down everythingh
                    if f_s==44100:
                        #resample by 2
                        a=torchaudio.functional.resample(a, 44100,fs_target)
                    elif f_s==48000:
                        a=torchaudio.functional.resample(a, 48000,fs_target)
                    else:
                        print("WARNING, strange fs", f_s)
           
                    proc_batch[i]=a[0:length_target] 
                    return proc_batch

def load_state_dict( state_dict, network=None, ema=None, optimizer=None, log=True):
        '''
        utility for loading state dicts for different models. This function sequentially tries different strategies
        args:
            state_dict: the state dict to load
        returns:
            True if the state dict was loaded, False otherwise
        Assuming the operations are don in_place, this function will not create a copy of the network and optimizer (I hope)
        '''
        #print(state_dict)
        if log: print("Loading state dict")
        if log:
            print(state_dict.keys())
        #if there
        try:
            if log: print("Attempt 1: trying with strict=True")
            if network is not None:
                network.load_state_dict(state_dict['network'])
            if optimizer is not None:
                optimizer.load_state_dict(state_dict['optimizer'])
            if ema is not None:
                ema.load_state_dict(state_dict['ema'])
            return True
        except Exception as e:
            if log:
                print("Could not load state dict")
                print(e)
        try:
            if log: print("Attempt 2: trying with strict=False")
            if network is not None:
                network.load_state_dict(state_dict['network'], strict=False)
            #we cannot load the optimizer in this setting
            #self.optimizer.load_state_dict(state_dict['optimizer'], strict=False)
            if ema is not None:
                ema.load_state_dict(state_dict['ema'], strict=False)
            return True
        except Exception as e:
            if log:
                print("Could not load state dict")
                print(e)
                print("training from scratch")
        try:
            if log: print("Attempt 3: trying with strict=False,but making sure that the shapes are fine")
            if ema is not None:
                ema_state_dict = ema.state_dict()
            if network is not None:
                network_state_dict = network.state_dict()
            i=0 
            if network is not None:
                for name, param in state_dict['network'].items():
                    if log: print("checking",name) 
                    if name in network_state_dict.keys():
                        if network_state_dict[name].shape==param.shape:
                                network_state_dict[name]=param
                                if log:
                                    print("assigning",name)
                                i+=1
            network.load_state_dict(network_state_dict)
            if ema is not None:
                for name, param in state_dict['ema'].items():
                        if log: print("checking",name) 
                        if name in ema_state_dict.keys():
                            if ema_state_dict[name].shape==param.shape:
                                ema_state_dict[name]=param
                                if log:
                                    print("assigning",name)
                                i+=1
     
            ema.load_state_dict(ema_state_dict)
     
            if i==0:
                if log: print("WARNING, no parameters were loaded")
                raise Exception("No parameters were loaded")
            elif i>0:
                if log: print("loaded", i, "parameters")
                return True

        except Exception as e:
            print(e)
            print("the second strict=False failed")


        try:
            if log: print("Attempt 4: Assuming the naming is different, with the network and ema called 'state_dict'")
            if network is not None:
                network.load_state_dict(state_dict['state_dict'])
            if ema is not None:
                ema.load_state_dict(state_dict['state_dict'])
        except Exception as e:
            if log:
                print("Could not load state dict")
                print(e)
                print("training from scratch")
                print("It failed 3 times!! but not giving up")
            #print the names of the parameters in self.network

        try:
            if log: print("Attempt 5: trying to load with different names, now model='model' and ema='ema_weights'")
            if ema is not None:
                dic_ema = {}
                for (key, tensor) in zip(state_dict['model'].keys(), state_dict['ema_weights']):
                    dic_ema[key] = tensor
                    ema.load_state_dict(dic_ema)
                return True
        except Exception as e:
            if log:
                print(e)

        try:
            if log: print("Attempt 6: If there is something wrong with the name of the ema parameters, we can try to load them using the names of the parameters in the model")
            if ema is not None:
                dic_ema = {}
                i=0
                for (key, tensor) in zip(state_dict['model'].keys(), state_dict['model'].values()):
                    if tensor.requires_grad:
                        dic_ema[key]=state_dict['ema_weights'][i]
                        i=i+1
                    else:
                        dic_ema[key]=tensor     
                ema.load_state_dict(dic_ema)
                return True
        except Exception as e:
            if log:
                print(e)

        #try:
        #assign the parameters in state_dict to self.network using a for loop
        print("Attempt 7: Trying to load the parameters one by one. This is for the dance diffusion model, looking for parameters starting with 'diffusion.' or 'diffusion_ema.'")
        if ema is not None:
            ema_state_dict = ema.state_dict()
        if network is not None:
            network_state_dict = ema.state_dict()
        i=0 
        if network is not None:
            for name, param in state_dict['state_dict'].items():
                print("checking",name) 
                if name.startswith("diffusion."):
                    i+=1
                    name=name.replace("diffusion.","")
                    if network_state_dict[name].shape==param.shape:
                        #print(param.shape, network.state_dict()[name].shape)
                        network_state_dict[name]=param
                        #print("assigning",name)

            network.load_state_dict(network_state_dict, strict=False)

        if ema is not None:
            for name, param in state_dict['state_dict'].items():
                if name.startswith("diffusion_ema."): 
                    i+=1
                    name=name.replace("diffusion_ema.","")
                    if ema_state_dict[name].shape==param.shape:
                        if log:
                                print(param.shape, ema.state_dict()[name].shape)
                        ema_state_dict[name]=param

            ema.load_state_dict(ema_state_dict, strict=False)

        if i==0:
            print("WARNING, no parameters were loaded")
            raise Exception("No parameters were loaded")
        elif i>0:
            print("loaded", i, "parameters")
            return True
        #except Exception as e:
        #    if log:
        #        print(e)

        return False

            
