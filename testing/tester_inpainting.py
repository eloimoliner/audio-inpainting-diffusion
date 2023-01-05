from datetime import date
import re
import torch
import torchaudio
#from src.models.unet_cqt import Unet_CQT
#from src.models.unet_stft import Unet_STFT
#from src.models.unet_1d import Unet_1d
#import src.utils.setup as utils_setup
#from src.sde import  VE_Sde_Elucidating
import utils.dnnlib as dnnlib
import os

import utils.logging as utils_logging
import wandb
import copy
import soundfile as sf

from glob import glob
from tqdm import tqdm

import utils.training_utils as t_utils
import omegaconf
import numpy as np
import cv2

'''
In this tester I will test:
- unconditional sampling
- time-masked inpainting
    - large gap in the middle
    - several small gaps
- spectrogram inpainting
    - some gap in the spectrogram
- tone inpainting
- outpainting
'''

class Tester():
    def __init__(
        self, args, network, diff_params, test_set=None, device=None, it=None
    ):
        self.args=args
        self.network=network
        self.diff_params=copy.copy(diff_params)
        self.device=device
        #choose gpu as the device if possible
        if self.device is None:
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network=network

        torch.backends.cudnn.benchmark = True

        today=date.today() 
        if it is None:
            self.it=0

        mode='test' #this is hardcoded for now, I'll have to figure out how to deal with the subdirectories once I want to test conditional sampling
        self.path_sampling=os.path.join(args.model_dir,mode+today.strftime("%d_%m_%Y")+"_"+str(self.it))
        if not os.path.exists(self.path_sampling):
            os.makedirs(self.path_sampling)


        #I have to rethink if I want to create the same sampler object to do conditional and unconditional sampling
        self.setup_sampler()

        self.use_wandb=False #hardcoded for now

        #S=2
        #if S>2.1 and S<2.2:
        #    #resampling 48k to 22.05k
        #    self.resample=torchaudio.transforms.Resample(160*2,147).to(self.device)
        #elif S!=1:
        #    N=int(self.args.exp.audio_len*S)
        #    self.resample=torchaudio.transforms.Resample(N,self.args.exp.audio_len).to(self.device)
        print("test_set", test_set)

        if test_set is not None:
            self.test_set=test_set
            self.do_inpainting=True
            self.do_bwe=True
        else:
            self.test_set=None
            self.do_inpainting=False
            self.do_bwe=False #these need to be set up in the config file

        self.paths={}
        if self.do_inpainting and ("inpainting" in self.args.tester.modes):
            self.do_inpainting=True
            mode="inpainting"
            self.paths[mode], self.paths[mode+"degraded"], self.paths[mode+"original"], self.paths[mode+"reconstructed"]=self.prepare_experiment("inpainting","masked","inpainted")
            #TODO add more information in the subirectory names
        else:
            self.do_inpainting=False
        
        if ("inpainting_shortgaps" in self.args.tester.modes):
            mode="inpainting_shortgaps"
            self.paths[mode], self.paths[mode+"degraded"], self.paths[mode+"original"], self.paths[mode+"reconstructed"]=self.prepare_experiment("inpainting_shortgaps","masked","inpainted")

        if ("spectrogram_inpainting" in self.args.tester.modes):
            mode="spectrogram_inpainting"
            self.paths[mode], self.paths[mode+"degraded"], self.paths[mode+"original"], self.paths[mode+"reconstructed"]=self.prepare_experiment("spectrogram_inpainting","masked","inpainted")
        
        if ("STN_inpainting" in self.args.tester.modes):
            mode="STN_inpainting"
            self.paths[mode], self.paths[mode+"degraded"], self.paths[mode+"original"], self.paths[mode+"reconstructed"]=self.prepare_experiment("STN_inpainting","masked","inpainted")

        if ("unconditional" in self.args.tester.modes):
            mode="unconditional"
            self.paths[mode]=self.prepare_unc_experiment("unconditional")

        try:
            self.stereo=self.args.tester.stereo
        except:
            self.stereo=False

    def prepare_unc_experiment(self, str):
            path_exp=os.path.join(self.path_sampling,str)
            if not os.path.exists(path_exp):
                os.makedirs(path_exp)
            return path_exp

    def prepare_experiment(self, str, str_degraded="degraded", str_reconstruced="recosntucted"):
            path_exp=os.path.join(self.path_sampling,str)
            if not os.path.exists(path_exp):
                os.makedirs(path_exp)

            n=str_degraded
            path_degraded=os.path.join(path_exp, n) #path for the lowpassed 
            #ensure the path exists
            if not os.path.exists(path_degraded):
                os.makedirs(path_degraded)
            
            path_original=os.path.join(path_exp, "original") #this will need a better organization
            #ensure the path exists
            if not os.path.exists(path_original):
                os.makedirs(path_original)
            
            n=str_reconstruced
            path_reconstructed=os.path.join(path_exp, n) #path for the clipped outputs
            #ensure the path exists
            if not os.path.exists(path_reconstructed):
                os.makedirs(path_reconstructed)

            return path_exp, path_degraded, path_original, path_reconstructed


    def setup_wandb(self):
        """
        Configure wandb, open a new run and log the configuration.
        """
        config=omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        self.wandb_run=wandb.init(project="testing"+self.args.exp.wandb.project, entity=self.args.exp.wandb.entity, config=config)
        wandb.watch(self.network, log_freq=self.args.logging.heavy_log_interval) #wanb.watch is used to log the gradients and parameters of the model to wandb. And it is used to log the model architecture and the model summary and the model graph and the model weights and the model hyperparameters and the model performance metrics.
        self.wandb_run.name=os.path.basename(self.args.model_dir)+"_"+self.args.exp.exp_name+"_"+self.wandb_run.id #adding the experiment number to the run name, bery important, I hope this does not crash
        self.use_wandb=True

    def setup_wandb_run(self, run):
        #get the wandb run object from outside (in trainer.py or somewhere else)
        self.wandb_run=run
        self.use_wandb=True

    def setup_sampler(self):
        self.rid=False
        self.sampler=dnnlib.call_func_by_name(func_name=self.args.tester.sampler_callable, model=self.network, diff_params=self.diff_params, args=self.args, rid=self.rid)
    
    def load_latest_checkpoint(self ):
        #load the latest checkpoint from self.args.model_dir
        try:
            # find latest checkpoint_id
            save_basename = f"{self.args.exp.exp_name}-*.pt"
            save_name = f"{self.args.model_dir}/{save_basename}"
            list_weights = glob(save_name)
            id_regex = re.compile(f"{self.args.exp.exp_name}-(\d*)\.pt")
            list_ids = [int(id_regex.search(weight_path).groups()[0])
                        for weight_path in list_weights]
            checkpoint_id = max(list_ids)

            state_dict = torch.load(
                f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt", map_location=self.device)
            try:
                self.network.load_state_dict(state_dict['ema'])
            except Exception as e:
                print(e)
                print("Failed to load in strict mode, trying again without strict mode")
                self.network.load_state_dict(state_dict['model'], strict=False)

            print(f"Loaded checkpoint {checkpoint_id}")
            return True
        except (FileNotFoundError, ValueError):
            raise ValueError("No checkpoint found")

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device)
        try:
            self.it=state_dict['it']
        except:
            self.it=0
        print("loading checkpoint")
        return t_utils.load_state_dict(state_dict, ema=self.network)

    def log_audio(self,preds, mode:str):
        string=mode+"_"+self.args.tester.name
        audio_path=utils_logging.write_audio_file(preds,self.args.exp.sample_rate, string,path=os.path.join(self.args.model_dir, self.paths[mode]),stereo=self.stereo)
        print(audio_path)
        if self.use_wandb:
            self.wandb_run.log({"audio_"+str(string): wandb.Audio(audio_path, sample_rate=self.args.exp.sample_rate)},step=self.it)
        #TODO: log spectrogram of the audio file to wandb
        spec_sample=utils_logging.plot_spectrogram_from_raw_audio(preds, self.args.logging.stft)
        if self.use_wandb:
            self.wandb_run.log({"spec_"+str(string): spec_sample}, step=self.it)

    def sample_unconditional(self):
        #the audio length is specified in the args.exp, doesnt depend on the tester
        if self.stereo: 
            shape=[self.args.tester.unconditional.num_samples,2, self.args.exp.audio_len]
        else:
            shape=[self.args.tester.unconditional.num_samples, self.args.exp.audio_len]
        #TODO assert that the audio_len is consistent with the model
        preds=self.sampler.predict_unconditional(shape, self.device)
        if self.use_wandb:
            self.log_audio(preds, "unconditional")
        else:
            #TODO do something else if wandb is not used, like saving the audio file to the model directory
            pass

        return preds

    def prepare_mask(self):
        inpainting_mask=torch.ones((1,self.args.exp.audio_len)).to(self.device) #assume between 5 and 6s of total length
        if self.args.tester.inpainting.mask_mode=="long":
            gap=int(self.args.tester.inpainting.long.gap_length*self.args.exp.sample_rate/1000)      
        
            if self.args.tester.inpainting.long.start_gap_idx =="None": #we were crashing here!
                #the gap is placed at the center
                start_gap_index=int(self.args.exp.audio_len//2 - gap//2) 
            else:
                start_gap_index=int(self.args.tester.inpainting.long.start_gap_idx*self.args.exp.sample_rate/1000)
        
            inpainting_mask[...,start_gap_index:(start_gap_index+gap)]=0
        elif self.args.tester.inpainting.mask_mode=="short":
            num_gaps=int(self.args.tester.inpainting.short.num_gaps)
            gap_len=int(self.args.tester.inpainting.short.gap_length*self.args.exp.sample_rate/1000)
            if self.args.tester.inpainting.short.start_gap_idx =="None":
                #randomize the gaps positions
                start_gap_index=torch.randint(0,self.args.exp.audio_len-gap_len,(num_gaps,))
                for i in range(num_gaps):
                    inpainting_mask[...,start_gap_index[i]:(start_gap_index[i]+gap_len)]=0
            else:
                raise NotImplementedError
        
        return inpainting_mask

    def prepare_spectral_mask(self):
        #prepare the mask for the spectral inpainting
        #The mask is defined in the stft domain
        A_trial=torch.ones((1,self.args.exp.audio_len))
        if self.args.tester.spectrogram_inpainting.stft.window=="hann":
            window=torch.hann_window(self.args.tester.spectrogram_inpainting.stft.win_length).to(A_trial.device)
        else:
            raise NotImplementedError("Only hann window is implemented for now")

        n_fft=self.args.tester.spectrogram_inpainting.stft.n_fft

        #add padding to the signal to make it a multiple of n_fft
        A_trial=torch.nn.functional.pad(A_trial, (0, n_fft-A_trial.shape[-1]%n_fft), mode='constant', value=0)
        A_trial_stft=torch.stft(A_trial, n_fft, self.args.tester.spectrogram_inpainting.stft.hop_length, self.args.tester.spectrogram_inpainting.stft.win_length, window, return_complex=True) 
        print(A_trial_stft.shape)
        B, F, T=A_trial_stft.shape
        print(B, F, T)
        A=torch.ones((F,T))
        #get frequency values in Hz
        freqs=torch.fft.fftfreq(n_fft, d=1/self.args.exp.sample_rate)
        print(freqs)
        fmin=self.args.tester.spectrogram_inpainting.min_masked_freq #In Hz
        fmax=self.args.tester.spectrogram_inpainting.max_masked_freq #in Hz
        #get the indices of the frequencies to mask
        fmin_idx=torch.argmin(torch.abs(freqs-fmin))
        fmax_idx=torch.argmin(torch.abs(freqs-fmax))

        gap=int(self.args.tester.spectrogram_inpainting.time_mask_length*self.args.exp.sample_rate/1000)      
        if self.args.tester.spectrogram_inpainting.time_start_idx =="None":
            start_gap_index=int(self.args.exp.audio_len//2 - gap//2)//self.args.tester.spectrogram_inpainting.stft.hop_length
        else:
            start_gap_index=int(self.args.tester.spectrogram_inpainting.time_start_idx*self.args.exp.sample_rate/1000)//self.args.tester.spectrogram_inpainting.stft.hop_length
        end_gap_index=start_gap_index+gap//self.args.tester.spectrogram_inpainting.stft.hop_length

        A[fmin_idx:fmax_idx,start_gap_index:end_gap_index]=0
        A=A.to(self.device)
        #print what we have done
        print("rectangular spectral mask with shape ", A.shape," created. The time indexes are :", start_gap_index, end_gap_index, "and the frequency indexes are :", fmin_idx, fmax_idx)
        return A




    def apply_spectral_mask(self,x,mask):
        """
        x: the input signal with shape (B, T)
        mask: the spectral mask to apply with shape (F, T)
        """
        if self.args.tester.spectrogram_inpainting.stft.window=="hann":
            window=torch.hann_window(self.args.tester.spectrogram_inpainting.stft.win_length).to(x.device)
        else:
            raise NotImplementedError("Only hann window is implemented for now")

        n_fft=self.args.tester.spectrogram_inpainting.stft.n_fft
        #add padding to the signal
        input_shape=x.shape
        x=torch.nn.functional.pad(x, (0, n_fft-x.shape[-1]%n_fft), mode='constant', value=0)
        X=torch.stft(x, n_fft, self.args.tester.spectrogram_inpainting.stft.hop_length, self.args.tester.spectrogram_inpainting.stft.win_length, window, return_complex=True) 
        #X.shape=(B, F, T)
        X_masked=X*mask.unsqueeze(0)
        #apply the inverse stft
        x_masked=torch.istft(X_masked, n_fft, self.args.tester.spectrogram_inpainting.stft.hop_length, self.args.tester.spectrogram_inpainting.stft.win_length, window, return_complex=False)
        print(x_masked.shape, input_shape)
        #why is the size different? Because of the padding?
        x_masked=x_masked[...,0:input_shape[-1]]
        print(x_masked.shape, input_shape)

        return x_masked

    def get_spectrogram_image(self, x):

        if self.args.tester.spectrogram_inpainting.stft.window=="hann":
            window=torch.hann_window(self.args.tester.spectrogram_inpainting.stft.win_length).to(x.device)
        else:
            raise NotImplementedError("Only hann window is implemented for now")

        n_fft=self.args.tester.spectrogram_inpainting.stft.n_fft
        #add padding to the signal
        input_shape=x.shape
        x=torch.nn.functional.pad(x, (0, n_fft-x.shape[-1]%n_fft), mode='constant', value=0)
        X=torch.stft(x, n_fft, self.args.tester.spectrogram_inpainting.stft.hop_length, self.args.tester.spectrogram_inpainting.stft.win_length, window, return_complex=True) 
        X=X.abs()
        #in db
        X=20*torch.log10(X+1e-6)
        X=X.detach().cpu().numpy()
        X=X[0]
        
        return X




    def test_inpainting_fordamushra(self):
        #if not self.do_inpainting or self.test_set is None:
        #    print("No test set specified, skipping inpainting test")
        #    return

        #assert self.test_set is not None

        gap_length=371 #ms
        #gap_length=743 #ms
        #gap_length=1486 #ms
        #gap_length=2962 #ms

        inpainting_mask=self.prepare_mask()


        if len(self.test_set) == 0:
            print("No samples found in test set")
        
        res=torch.zeros((len(self.test_set),self.args.exp.audio_len))


        path_test_set="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/inpainting_test/long_gaps/original"
        path_masked="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/inpainting_test/long_gaps/masked/"+str(gap_length)
        if not os.path.exists(path_masked):
            os.makedirs(path_masked)
        path_output="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/inpainting_test/long_gaps/output/"+str(gap_length)
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        filenames=glob(os.path.join(path_test_set,"*.wav"))
        print(filenames)
        #the conditional sampling uses batch_size=1, so we need to loop over the test set. This is done for simplicity, but it is not the most efficient way to do it.
        #for i, (original, fs, filename) in enumerate(tqdm(self.test_set)):

        audio_len=self.args.exp.audio_len #the length that the model works on
        for i, filename in enumerate(tqdm(filenames)):
            n=os.path.splitext(os.path.basename(filename))[0]
            original, fs=sf.read(filename)
            #let's hope that the shape is correct
            original=torch.Tensor(original).unsqueeze(0)

            original=original.float().to(self.device)
            #seg=self.resample_audio(original, fs)
            #seg=audio #the fs should also be ok

            #seg=torchaudio.functional.resample(seg, self.args.exp.resample_factor, 1)
            #utils_logging.write_audio_file(seg, self.args.exp.sample_rate, n, path=self.paths["inpainting"+"original"])
            print(original.shape)

            #the mask should go in the middle
            inpainting_mask=torch.ones((1,original.shape[-1])).to(self.device) #assume between 5 and 6s of total length
            gap=int(gap_length*fs/1000)      
        
            #the gap is placed at the center
            start_gap_index=int(original.shape[-1]//2 - gap//2) 
        
            inpainting_mask[...,start_gap_index:(start_gap_index+gap)]=0

            masked=original*inpainting_mask

            print(n,path_masked)
            utils_logging.write_audio_file(masked, self.args.exp.sample_rate, n, path=path_masked)

            start_seg_index=int(original.shape[-1]//2 - audio_len//2) 
            seg=original[...,start_seg_index:(start_seg_index+audio_len)]
            seg_mask=inpainting_mask[...,start_seg_index:(start_seg_index+audio_len)]
               
            pred=self.sampler.predict_inpainting(seg*seg_mask, seg_mask)

            print(original.shape, pred.shape)
            result=torch.cat((original[...,0:start_seg_index],pred,original[...,start_seg_index+audio_len:]),-1)

            utils_logging.write_audio_file(result, self.args.exp.sample_rate, n, path=path_output)
            #res[i,:]=pred

        #if self.use_wandb:
        #self.log_audio(res, "inpainting")
        
        #TODO save the files in the subdirectory inpainting of the model directory

    def test_inpainting_short_gaps(self):
        #if not self.do_inpainting or self.test_set is None:
        #    print("No test set specified, skipping inpainting test")
        #    return

        #assert self.test_set is not None


        #inpainting_mask=self.prepare_mask()

        if len(self.test_set) == 0:
            print("No samples found in test set")
        
        res=torch.zeros((len(self.test_set),self.args.exp.audio_len))
        #the conditional sampling uses batch_size=1, so we need to loop over the test set. This is done for simplicity, but it is not the most efficient way to do it.
        for i, (original, mask, fs, filename) in enumerate(tqdm(self.test_set)):
            n=os.path.splitext(filename[0])[0]

            mask=mask.float().to(self.device).squeeze(-1)
            original=original.float().to(self.device)
            print(original.shape, mask.shape)
            seg=self.resample_audio(original, fs)
            #seg=torchaudio.functional.resample(seg, self.args.exp.resample_factor, 1)
            utils_logging.write_audio_file(seg, self.args.exp.sample_rate, n, path=self.paths["inpainting_shortgaps"+"original"])
            masked=seg*mask
            utils_logging.write_audio_file(masked, self.args.exp.sample_rate, n, path=self.paths["inpainting_shortgaps"+"degraded"])
            pred=self.sampler.predict_inpainting(masked, mask)
            utils_logging.write_audio_file(pred, self.args.exp.sample_rate, n, path=self.paths["inpainting_shortgaps"+"reconstructed"])
            utils_logging.write_audio_file(pred, self.args.exp.sample_rate, n, path=self.args.dset.test.path_output)
            res[i,:]=pred

        #if self.use_wandb:
        #    self.log_audio(res, "inpainting")
        
        #TODO save the files in the subdirectory inpainting of the model directory

    def test_spectrogram_inpainting(self):

        assert self.test_set is not None

        inpainting_mask=self.prepare_spectral_mask()

        if len(self.test_set) == 0:
            print("No samples found in test set")
        
        res=torch.zeros((len(self.test_set),self.args.exp.audio_len))
        #the conditional sampling uses batch_size=1, so we need to loop over the test set. This is done for simplicity, but it is not the most efficient way to do it.
        for i, (original, fs, filename) in enumerate(tqdm(self.test_set)):
            n=os.path.splitext(filename[0])[0]
            original=original.float().to(self.device)
            seg=self.resample_audio(original, fs)
            #seg=torchaudio.functional.resample(seg, self.args.exp.resample_factor, 1)
            utils_logging.write_audio_file(seg, self.args.exp.sample_rate, n, path=self.paths["spectrogram_inpainting"+"original"])
            masked=self.apply_spectral_mask(seg, inpainting_mask)

            utils_logging.write_audio_file(masked, self.args.exp.sample_rate, n, path=self.paths["spectrogram_inpainting"+"degraded"])

            pred=self.sampler.predict_spectrogram_inpainting(masked, inpainting_mask)

            utils_logging.write_audio_file(pred, self.args.exp.sample_rate, n, path=self.paths["spectrogram_inpainting"+"reconstructed"])
            res[i,:]=pred

        if self.use_wandb:
            self.log_audio(res, "spectrogram_inpainting")
        
        #TODO save the files in the subdirectory inpainting of the model directory

    def interactive_spectrogram_inpainting(self, audio, fs, mask):
        #audio: torch.tensor of shape (1, audio_len)
        #mask: spectral mask

        audio=audio.float().to(self.device)
        audio=self.resample_audio(audio, fs)

        inpainting_mask=mask.to(self.device)

        masked=self.apply_spectral_mask(audio, inpainting_mask)
        pred=self.sampler.predict_spectrogram_inpainting(masked, inpainting_mask)

        return pred

    def test_inpainting(self):
        if not self.do_inpainting or self.test_set is None:
            print("No test set specified, skipping inpainting test")
            return

        assert self.test_set is not None


        inpainting_mask=self.prepare_mask()

        if len(self.test_set) == 0:
            print("No samples found in test set")
        
        res=torch.zeros((len(self.test_set),self.args.exp.audio_len))
        #the conditional sampling uses batch_size=1, so we need to loop over the test set. This is done for simplicity, but it is not the most efficient way to do it.
        for i, (original, fs, filename) in enumerate(tqdm(self.test_set)):
            n=os.path.splitext(filename[0])[0]
            original=original.float().to(self.device)
            seg=self.resample_audio(original, fs)
            #seg=torchaudio.functional.resample(seg, self.args.exp.resample_factor, 1)
            utils_logging.write_audio_file(seg, self.args.exp.sample_rate, n, path=self.paths["inpainting"+"original"])
            masked=seg*inpainting_mask
            utils_logging.write_audio_file(masked, self.args.exp.sample_rate, n, path=self.paths["inpainting"+"degraded"])
            if not(self.rid):
                pred=self.sampler.predict_inpainting(masked, inpainting_mask)
            else:
                pred, rid_denoised, rid_grads, rid_grad_update, rid_pocs, rid_xt, rid_xt2, t =self.sampler.predict_inpainting(masked, inpainting_mask)
                rid_denoised=rid_denoised.cpu().numpy()
                np.save(os.path.join(self.paths["inpainting"],"rid_denoised.npy"), rid_denoised)
                rid_grads=rid_grads.cpu().numpy()
                np.save(os.path.join(self.paths["inpainting"],"rid_grads.npy"), rid_grads)
                rid_grad_update=rid_grad_update.cpu().numpy()
                np.save(os.path.join(self.paths["inpainting"],"rid_grad_update.npy"), rid_grad_update)
                rid_pocs=rid_pocs.cpu().numpy()
                np.save(os.path.join(self.paths["inpainting"],"rid_pocs.npy"), rid_pocs)
                rid_xt=rid_xt.cpu().numpy()
                np.save(os.path.join(self.paths["inpainting"],"rid_xt.npy"), rid_xt)
                rif_xt2=rid_xt2.cpu().numpy()
                np.save(os.path.join(self.paths["inpainting"],"rid_xt2.npy"), rid_xt2)


            utils_logging.write_audio_file(pred, self.args.exp.sample_rate, n, path=self.paths["inpainting"+"reconstructed"])
            res[i,:]=pred

        #if self.use_wandb:
        #    self.log_audio(res, "inpainting")
        
        #TODO save the files in the subdirectory inpainting of the model directory

    def resample_audio(self, audio, fs):
        #this has been reused from the trainer.py
        return t_utils.resample_batch(audio, fs, self.args.exp.sample_rate, self.args.exp.audio_len)

    def sample_inpainting(self, y, mask):

        y_masked=y*mask
        #shape=[self.args.tester.unconditional.num_samples, self.args.tester.unconditional.audio_len]
        #TODO assert that the audio_len is consistent with the model
        preds=self.sampler.predict_inpainting(y_masked, mask)

        return preds
    


    def dodajob(self):
        self.setup_wandb()
        if "unconditional" in self.args.tester.modes:
            print("testing unconditional")
            self.sample_unconditional()
        self.it+=1
        if "inpainting" in self.args.tester.modes:
            self.test_inpainting()
        if "inpainting_fordamushra" in self.args.tester.modes:
            self.test_inpainting_fordamushra()
        if "inpainting_shortgaps" in self.args.tester.modes:
            self.test_inpainting_short_gaps()
        if "spectrogram_inpainting" in self.args.tester.modes:
            self.test_spectrogram_inpainting()
        if "STN_inpainting" in self.args.tester.modes:
            self.test_STN_inpainting()
        self.it+=1

        #do I want to save this audio file locally? I think I do, but I'll have to figure out how to do it


