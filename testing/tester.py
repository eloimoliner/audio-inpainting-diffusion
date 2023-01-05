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

from glob import glob
from tqdm import tqdm

import utils.bandwidth_extension as utils_bwe
import utils.training_utils as t_utils
import omegaconf


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

        if self.do_bwe and ("bwe" in self.args.tester.modes):
            self.do_bwe=True
            mode="bwe"
            self.paths[mode], self.paths[mode+"degraded"], self.paths[mode+"original"], self.paths[mode+"reconstructed"]=self.prepare_experiment("bwe","lowpassed","bwe")
            #TODO add more information in the subirectory names
        else:
            self.do_bwe=False

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
        self.sampler=dnnlib.call_func_by_name(func_name=self.args.tester.sampler_callable, model=self.network, diff_params=self.diff_params, args=self.args)
    
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

    def load_checkpoint_legacy(self, path):
        state_dict = torch.load(path, map_location=self.device)

        try:
            print("load try 1")
            self.network.load_state_dict(state_dict['ema'])
        except:
            #self.network.load_state_dict(state_dict['model'])
            try:
                print("load try 2")
                dic_ema = {}
                for (key, tensor) in zip(state_dict['model'].keys(), state_dict['ema_weights']):
                    dic_ema[key] = tensor
                self.network.load_state_dict(dic_ema)
            except:
                print("load try 3")
                dic_ema = {}
                i=0
                for (key, tensor) in zip(state_dict['model'].keys(), state_dict['model'].values()):
                    if tensor.requires_grad:
                        dic_ema[key]=state_dict['ema_weights'][i]
                        i=i+1
                    else:
                        dic_ema[key]=tensor     
                self.network.load_state_dict(dic_ema)
        try:
            self.it=state_dict['it']
        except:
            self.it=0

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

    def test_inpainting(self):
        if not self.do_inpainting or self.test_set is None:
            print("No test set specified, skipping inpainting test")
            return

        assert self.test_set is not None

        self.inpainting_mask=torch.ones((1,self.args.exp.audio_len)).to(self.device) #assume between 5 and 6s of total length
        gap=int(self.args.tester.inpainting.gap_length*self.args.exp.sample_rate/1000)      

        if self.args.tester.inpainting.start_gap_idx =="None": #we were crashing here!
            #the gap is placed at the center
            start_gap_index=int(self.args.exp.audio_len//2 - gap//2) 
        else:
            start_gap_index=int(self.args.tester.inpainting.start_gap_idx*self.args.exp.sample_rate/1000)
        self.inpainting_mask[...,start_gap_index:(start_gap_index+gap)]=0

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
            masked=seg*self.inpainting_mask
            utils_logging.write_audio_file(masked, self.args.exp.sample_rate, n, path=self.paths["inpainting"+"degraded"])
            pred=self.sampler.predict_inpainting(masked, self.inpainting_mask)
            utils_logging.write_audio_file(pred, self.args.exp.sample_rate, n, path=self.paths["inpainting"+"reconstructed"])
            res[i,:]=pred

        if self.use_wandb:
            self.log_audio(res, "inpainting")
        
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
    
    def test_bwe(self, typefilter="whateverIignoreit"):
        if not self.do_bwe or self.test_set is None:
            print("No test set specified, skipping inpainting test")
            return

        assert self.test_set is not None

        if len(self.test_set) == 0:
            print("No samples found in test set")

        #prepare lowpass filters
        self.filter=utils_bwe.prepare_filter(self.args, self.args.exp.sample_rate)
        
        res=torch.zeros((len(self.test_set),self.args.exp.audio_len))
        #the conditional sampling uses batch_size=1, so we need to loop over the test set. This is done for simplicity, but it is not the most efficient way to do it.
        for i, (original, fs, filename) in enumerate(tqdm(self.test_set)):
            n=os.path.splitext(filename[0])[0]
            original=original.float().to(self.device)
            seg=self.resample_audio(original, fs)

            utils_logging.write_audio_file(seg, self.args.exp.sample_rate, n, path=self.paths["bwe"+"original"])

            y=utils_bwe.apply_low_pass(seg, self.filter, self.args.tester.bandwidth_extension.filter.type) 

            if self.args.tester.noise_in_observations_SNR != "None":
                SNR=10**(self.args.tester.noise_in_observations_SNR/10)
                sigma2_s=torch.var(y, -1)
                sigma=torch.sqrt(sigma2_s/SNR)
                y+=sigma*torch.randn(y.shape).to(y.device)

            utils_logging.write_audio_file(y, self.args.exp.sample_rate, n, path=self.paths["bwe"+"degraded"])

            pred=self.sampler.predict_bwe(y, self.filter, self.args.tester.bandwidth_extension.filter.type)
            utils_logging.write_audio_file(pred, self.args.exp.sample_rate, n, path=self.paths["bwe"+"reconstructed"])
            res[i,:]=pred

        if self.use_wandb:
            self.log_audio(res, "bwe")

            #preprocess the audio file if necessary


    def dodajob(self):
        self.setup_wandb()
        if "unconditional" in self.args.tester.modes:
            print("testing unconditional")
            self.sample_unconditional()
        self.it+=1
        if "blind_bwe" in self.args.tester.modes:
            print("testing blind bwe")
            #tester.test_blind_bwe(typefilter="whatever")
            self.tester.test_blind_bwe(typefilter="3rdoct")
        self.it+=1
        if "filter_bwe" in self.args.tester.modes:
            print("testing filter bwe")
            self.test_filter_bwe(typefilter="3rdoct")
        self.it+=1
        if "unconditional_operator" in self.args.tester.modes:
            print("testing unconditional operator")
            self.sample_unconditional_operator()
        self.it+=1
        if "bwe" in self.args.tester.modes:
            print("testing bwe")
            self.test_bwe(typefilter="3rdoct")
        self.it+=1
        if "inpainting" in self.args.tester.modes:
            self.test_inpainting()
        self.it+=1

        #do I want to save this audio file locally? I think I do, but I'll have to figure out how to do it


