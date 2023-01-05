"""Main training loop."""

import os
import time
import copy

import numpy as np
import torch
import torchaudio

from utils.torch_utils import training_stats
from utils.torch_utils import misc

import librosa
from glob import glob
import re

import wandb

import utils.logging as utils_logging
from torch.profiler import tensorboard_trace_handler

import omegaconf

import utils.training_utils as t_utils

#----------------------------------------------------------------------------
class Trainer():
    def __init__(self, args, dset, network, optimizer, diff_params, tester=None, device='cpu'):
        self.args=args
        self.dset=dset
        self.network=network
        self.optimizer=optimizer
        self.diff_params=diff_params
        self.device=device

        #testing means generating demos by sampling from the model
        self.tester=tester
        if self.tester is None or not(self.args.tester.do_test):
            self.do_test=False
        else:
            self.do_test=True

        #these are settings set by karras. I am not sure what they do
        #np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
        torch.manual_seed(np.random.randint(1 << 31))
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False

        self.total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print("total_params: ",self.total_params/1e6, "M")

        self.ema = copy.deepcopy(self.network).eval().requires_grad_(False)
        
        #resume from checkpoint
        self.latest_checkpoint=None
        resuming=False
        if self.args.exp.resume:
            if self.args.exp.resume_checkpoint != "None":
                resuming =self.resume_from_checkpoint(checkpoint_path=self.args.exp.resume_checkpoint)
            else:
                resuming =self.resume_from_checkpoint()
            if not resuming:
                print("Could not resume from checkpoint")
                print("training from scratch")
            else:
                print("Resuming from iteration {}".format(self.it))

        if not resuming:
            self.it=0
            self.latest_checkpoint=None

        if self.args.logging.print_model_summary:
            #if dist.get_rank() == 0:
             with torch.no_grad():
                 audio=torch.zeros([args.exp.batch,args.exp.audio_len], device=device)
                 sigma = torch.ones([args.exp.batch], device=device).unsqueeze(-1)
                 misc.print_module_summary(self.network, [audio, sigma ], max_nesting=2)

        
        if self.args.logging.log:
            #assert self.args.logging.heavy_log_interval % self.args.logging.save_interval == 0 #sorry for that, I just want to make sure that you are not wasting your time by logging too often, as the tester is only updated with the ema weights from a checkpoint
            self.setup_wandb()
            if self.do_test:
               self.tester.setup_wandb_run(self.wandb_run)
            self.setup_logging_variables()

        self.profile=False
        if self.args.logging.profiling.enabled:
            try:
                print("Profiling is being enabled")
                wait=self.args.logging.profiling.wait
                warmup=self.args.logging.profiling.warmup
                active=self.args.logging.profiling.active
                repeat=self.args.logging.profiling.repeat

                schedule =  torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=repeat)
                self.profiler = torch.profiler.profile(
                schedule=schedule, on_trace_ready=tensorboard_trace_handler("wandb/latest-run/tbprofile"), profile_memory=True, with_stack=False)
                self.profile=True
                self.profile_total_steps = (wait + warmup + active) * (1 + repeat)
            except Exception as e:

                print("Could not setup profiler")
                print(e)
                self.profile=False
                

    def setup_wandb(self):
        """
        Configure wandb, open a new run and log the configuration.
        """
        config=omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        config["total_params"]=self.total_params
        self.wandb_run=wandb.init(project=self.args.exp.wandb.project, entity=self.args.exp.wandb.entity, config=config)
        wandb.watch(self.network, log="all", log_freq=self.args.logging.heavy_log_interval) #wanb.watch is used to log the gradients and parameters of the model to wandb. And it is used to log the model architecture and the model summary and the model graph and the model weights and the model hyperparameters and the model performance metrics.
        self.wandb_run.name=os.path.basename(self.args.model_dir)+"_"+self.args.exp.exp_name+"_"+self.wandb_run.id #adding the experiment number to the run name, bery important, I hope this does not crash

    
    def setup_logging_variables(self):

        self.sigma_bins = np.logspace(np.log10(self.args.diff_params.sigma_min), np.log10(self.args.diff_params.sigma_max), num=self.args.logging.num_sigma_bins, base=10)

        #logarithmically spaced bins for the frequency logging
        self.freq_bins=np.logspace(np.log2(self.args.logging.cqt.fmin), np.log2(self.args.logging.cqt.fmin*2**(self.args.logging.cqt.num_octs)), num=self.args.logging.cqt.num_octs*self.args.logging.cqt.bins_per_oct, base=2)
        self.freq_bins=self.freq_bins.astype(int)



    def load_state_dict(self, state_dict):
        #print(state_dict)
        return t_utils.load_state_dict(state_dict, network=self.network, ema=self.ema, optimizer=self.optimizer)


    def resume_from_checkpoint(self, checkpoint_path=None, checkpoint_id=None):
        # Resume training from latest checkpoint available in the output director
        if checkpoint_path is not None:
            try:
                checkpoint=torch.load(checkpoint_path, map_location=self.device)
                print(checkpoint.keys())
                #if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it=157007 #large number to mean that we loaded somethin, but it is arbitrary
                return self.load_state_dict(checkpoint)
            except Exception as e:
                print("Could not resume from checkpoint")
                print(e)
                print("training from scratch")
                self.it=0
                return False
        else:
            try:
                print("trying to load a project checkpoint")
                print("checkpoint_id", checkpoint_id)
                if checkpoint_id is None:
                    # find latest checkpoint_id
                    save_basename = f"{self.args.exp.exp_name}-*.pt"
                    save_name = f"{self.args.model_dir}/{save_basename}"
                    print(save_name)
                    list_weights = glob(save_name)
                    id_regex = re.compile(f"{self.args.exp.exp_name}-(\d*)\.pt")
                    list_ids = [int(id_regex.search(weight_path).groups()[0])
                                for weight_path in list_weights]
                    checkpoint_id = max(list_ids)
                    print(checkpoint_id)
    
                checkpoint = torch.load(
                    f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt", map_location=self.device)
                #if it is possible, retrieve the iteration number from the checkpoint
                try:
                    self.it = checkpoint['it']
                except:
                    self.it=159000 #large number to mean that we loaded somethin, but it is arbitrary
                self.load_state_dict(checkpoint)
                return True
            except Exception as e:
                print(e)
                return False


    def state_dict(self):
        return {
            'it': self.it,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
            'args': self.args,
        }

    def save_checkpoint(self):
        save_basename = f"{self.args.exp.exp_name}-{self.it}.pt"
        save_name = f"{self.args.model_dir}/{save_basename}"
        torch.save(self.state_dict(), save_name)
        print("saving",save_name)
        if self.args.logging.remove_last_checkpoint:
            try:
                os.remove(self.latest_checkpoint)
                print("removed last checkpoint", self.latest_checkpoint)
            except:
                print("could not remove last checkpoint", self.latest_checkpoint)
        self.latest_checkpoint=save_name


    def process_loss_for_logging(self, error: torch.Tensor, sigma: torch.Tensor):
        """
        This function is used to process the loss for logging. It is used to group the losses by the values of sigma and report them using training_stats.
        args:
            error: the error tensor with shape [batch, audio_len]
            sigma: the sigma tensor with shape [batch]
        """
        #sigma values are ranged between self.args.diff_params.sigma_min and self.args.diff_params.sigma_max. We need to quantize the values of sigma into 10 logarithmically spaced bins between self.args.diff_params.sigma_min and self.args.diff_params.sigma_max
        torch.nan_to_num(error) #not tested might crash
        error=error.detach().cpu().numpy()

        for i in range(len(self.sigma_bins)):
            if i == 0:
                mask = sigma <= self.sigma_bins[i]
            elif i == len(self.sigma_bins)-1:
                mask = (sigma <= self.sigma_bins[i]) & (sigma > self.sigma_bins[i-1])

            else:
                mask = (sigma <= self.sigma_bins[i]) & (sigma > self.sigma_bins[i-1])
            mask=mask.squeeze(-1).cpu()
            if mask.sum() > 0:
                #find the index of the first element of the mask
                idx = np.where(mask==True)[0][0]

                training_stats.report('error_sigma_'+str(self.sigma_bins[i]),error[idx].mean())

    def get_batch(self):
        #load the data batch
        if self.args.dset.name == "maestro_allyears":
            #this dataset has data sampled at different frequencies, so we need to resample it. The dataset returns a tuple (audio, fs), where fs is the sampling frequency of the given audio sample. Moreover, the size of the audio tensor is [B, dset.load_len], where dset.load_len is an arbitrary number designed to be sufficiently large so that the 48kHz audio samples can be loaded without any problem. We need to resample the audio tensor to the desired sampling frequency, and then crop it to the desired length.

            audio, fs = next(self.dset)
            audio=audio.to(self.device).to(torch.float32)

            return t_utils.resample_batch(audio, fs, self.args.exp.sample_rate, self.args.exp.audio_len)
        else: 
            audio = next(self.dset)
            audio=audio.to(self.device).to(torch.float32)
            #do resampling if needed
            if self.args.exp.resample_factor != 1:
                audio=torchaudio.functional.resample(audio, self.args.exp.resample_factor, 1)

            return audio
    def train_step(self):
        # Train step
        it_start_time = time.time()
        #self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.zero_grad()
        st_time=time.time()
        for round_idx in range(self.args.exp.num_accumulation_rounds):
            #with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
            audio=self.get_batch()

            #print(audio.shape, self.args.exp.audio_len)
            error, sigma = self.diff_params.loss_fn(self.network, audio)
            loss=error.mean()
            loss.backward() #TODO: take care of the loss scaling if using mixed precision
            #do I want to call this at every round? It will slow down the training. I will try it and see what happens



        if self.it <= self.args.exp.lr_rampup_it:
            for g in self.optimizer.param_groups:
                #learning rate ramp up
                g['lr'] = self.args.exp.lr * min(self.it / max(self.args.exp.lr_rampup_it, 1e-8), 1)


        if self.args.exp.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.exp.max_grad_norm)

        # Update weights.
        self.optimizer.step()

        end_time=time.time()
        if self.args.logging.log:
            self.process_loss_for_logging(error, sigma)

        it_end_time = time.time()
        print("it :",self.it, "time:, ",end_time-st_time, "total_time: ",training_stats.report('it_time',it_end_time-it_start_time) ,"loss: ", training_stats.report('loss', loss.item())) #TODO: take care of the logging


    def update_ema(self):
        """Update exponential moving average of self.network weights."""

        ema_rampup = self.args.exp.ema_rampup  #ema_rampup should be set to 10000 in the config file
        ema_rate=self.args.exp.ema_rate #ema_rate should be set to 0.9999 in the config file
        t = self.it * self.args.exp.batch
        with torch.no_grad():
            if t < ema_rampup:
                s = np.clip(t / ema_rampup, 0.0, ema_rate)
                for dst, src in zip(self.ema.parameters(), self.network.parameters()):
                    dst.copy_(dst * s + src * (1-s))
            else:
                for dst, src in zip(self.ema.parameters(), self.network.parameters()):
                    dst.copy_(dst * ema_rate + src * (1-ema_rate))

    def easy_logging(self):
        """
         Do the simplest logging here. This will be called every 1000 iterations or so
        I will use the training_stats.report function for this, and aim to report the means and stds of the losses in wandb
        """
        training_stats.default_collector.update()
        #Is it a good idea to log the stds of the losses? I think it is not.
        loss_mean=training_stats.default_collector.mean('loss')
        self.wandb_run.log({'loss':loss_mean}, step=self.it)
        loss_std=training_stats.default_collector.std('loss')
        self.wandb_run.log({'loss_std':loss_std}, step=self.it)

        it_time_mean=training_stats.default_collector.mean('it_time')
        self.wandb_run.log({'it_time_mean':it_time_mean}, step=self.it)
        it_time_std=training_stats.default_collector.std('it_time')
        self.wandb_run.log({'it_time_std':it_time_std}, step=self.it)
        
        #here reporting the error respect to sigma. I should make a fancier plot too, with mean and std
        sigma_means=[]
        sigma_stds=[]
        for i in range(len(self.sigma_bins)):
            a=training_stats.default_collector.mean('error_sigma_'+str(self.sigma_bins[i]))
            sigma_means.append(a)
            self.wandb_run.log({'error_sigma_'+str(self.sigma_bins[i]):a}, step=self.it)
            a=training_stats.default_collector.std('error_sigma_'+str(self.sigma_bins[i]))
            sigma_stds.append(a)

        
        figure=utils_logging.plot_loss_by_sigma(sigma_means,sigma_stds, self.sigma_bins)
        wandb.log({"loss_dependent_on_sigma": figure}, step=self.it, commit=True)


    def heavy_logging(self):
        """
        Do the heavy logging here. This will be called every 10000 iterations or so
        """
        if self.do_test:

            if self.latest_checkpoint is not None:
                self.tester.load_checkpoint(self.latest_checkpoint)

            preds=self.tester.sample_unconditional()
            preds=self.tester.test_inpainting()

    def log_audio(self,x, name):
        string=name+"_"+self.args.tester.name
        audio_path=utils_logging.write_audio_file(x,self.args.exp.sample_rate, string,path=self.args.model_dir)
        self.wandb_run.log({"audio_"+str(string): wandb.Audio(audio_path, sample_rate=self.args.exp.sample_rate)},step=self.it)
        #TODO: log spectrogram of the audio file to wandb
        spec_sample=utils_logging.plot_spectrogram_from_raw_audio(x, self.args.logging.stft)
        self.wandb_run.log({"spec_"+str(string): spec_sample}, step=self.it)


    def training_loop(self):
        
        # Initialize.

        #ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)

        while True:
            # Accumulate gradients.

            self.train_step()

            self.update_ema()
            
            if self.profile and self.args.logging.log:
                print(self.profile, self.profile_total_steps, self.it)
                if self.it<self.profile_total_steps:
                    self.profiler.step()
                elif self.it==self.profile_total_steps +1:
                    #log trace as an artifact in wandb
                    profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
                    profile_art.add_file(glob("wandb/latest-run/tbprofile/*.pt.trace.json")[0], "trace.pt.trace.json")
                    wandb.log_artifact(profile_art)
                    print("proiling done")
                elif self.it>self.profile_total_steps +1:
                    self.profile=False



            if self.it>0 and self.it%self.args.logging.save_interval==0 and self.args.logging.save_model:
                #self.save_snapshot() #are the snapshots necessary? I think they are not.
                self.save_checkpoint()


            if self.it>0 and self.it%self.args.logging.heavy_log_interval==0 and self.args.logging.log:
                self.heavy_logging()
                #self.conditional_demos()

            if self.it>0 and self.it%self.args.logging.log_interval==0 and self.args.logging.log:
                self.easy_logging()


            # Update state.
            self.it += 1


    #----------------------------------------------------------------------------
