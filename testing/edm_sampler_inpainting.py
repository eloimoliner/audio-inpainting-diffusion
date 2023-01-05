from tqdm import tqdm
import torch
import torchaudio
#import scipy.signal
#import numpy as np
#from utils.decSTN_pytorch import apply_STN_mask

class Sampler():

    def __init__(self, model, diff_params, args, rid=False):

        self.model = model
        self.diff_params = diff_params #same as training, useful if we need to apply a wrapper or something
        self.args=args
        if not(self.args.tester.diff_params.same_as_training):
            self.update_diff_params()


        self.order=self.args.tester.order
        self.xi=self.args.tester.posterior_sampling.xi
         #hyperparameter for the reconstruction guidance
        self.data_consistency=self.args.tester.data_consistency.use and self.args.tester.data_consistency.type=="always"

        self.data_consistency_end=self.args.tester.data_consistency.use and self.args.tester.data_consistency.type=="end"
        if self.data_consistency or self.data_consistency_end:
            if self.args.tester.data_consistency.smooth:
                self.smooth=True
            else:
                self.smooth=False

        #use reconstruction gudance without replacement
        self.nb_steps=self.args.tester.T

        #self.treshold_on_grads=args.tester.inference.max_thresh_grads
        self.rid=rid #this is for logging, ignore for now

        #try:
        #    self.stereo=self.args.tester.stereo
        #except:
        #    self.stereo=False


    def update_diff_params(self):
        #the parameters for testing might not be necesarily the same as the ones used for training
        self.diff_params.sigma_min=self.args.tester.diff_params.sigma_min
        self.diff_params.sigma_max =self.args.tester.diff_params.sigma_max
        self.diff_params.ro=self.args.tester.diff_params.ro
        self.diff_params.sigma_data=self.args.tester.diff_params.sigma_data
        #par.diff_params.meters stochastic sampling
        self.diff_params.Schurn=self.args.tester.diff_params.Schurn
        self.diff_params.Stmin=self.args.tester.diff_params.Stmin
        self.diff_params.Stmax=self.args.tester.diff_params.Stmax
        self.diff_params.Snoise=self.args.tester.diff_params.Snoise



    def get_score_rec_guidance(self, x, y, t_i, degradation):

        x.requires_grad_()
        x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))

        if self.args.tester.filter_out_cqt_DC_Nyq:
            x_hat=self.model.CQTransform.apply_hpf_DC(x_hat)

        den_rec= degradation(x_hat) 

        if len(y.shape)==3:
            dim=(1,2)
        elif len(y.shape)==2:
            dim=1

        if self.args.tester.posterior_sampling.norm=="smoothl1":
            norm=torch.nn.functional.smooth_l1_loss(y, den_rec, reduction='sum', beta=self.args.tester.posterior_sampling.smoothl1_beta)
        else:
            norm=torch.linalg.norm(y-den_rec,dim=dim, ord=self.args.tester.posterior_sampling.norm)

        
        rec_grads=torch.autograd.grad(outputs=norm,
                                      inputs=x)

        rec_grads=rec_grads[0]
        
        normguide=torch.linalg.norm(rec_grads)/self.args.exp.audio_len**0.5
        
        #normalize scaling
        #s=self.xi/(normguide*t_i+1e-6)
        s=t_i*self.xi/(normguide+1e-6)
        
        #optionally apply a treshold to the gradients
        if False:
            #pply tresholding to the gradients. It is a dirty trick but helps avoiding bad artifacts 
            rec_grads=torch.clip(rec_grads, min=-self.treshold_on_grads, max=self.treshold_on_grads)
        
        if self.rid:
            x_hat_old=x_hat.detach().clone()

        x_hat=x_hat-s*rec_grads #apply gradients

        if self.rid:
            x_hat_old_2=x_hat.detach().clone()

        if self.data_consistency:
            x_hat=self.proj_convex_set(x_hat.detach())

        score=(x_hat.detach()-x)/t_i**2

        #apply scaled guidance to the score
        #score=score-s*rec_grads
        if self.rid:
            #score, denoised esimate,  s*rec_grads, denoised_estimate minus gradients, x_hat after pocs
            return score, x_hat_old, s*rec_grads, x_hat_old_2, x_hat
        else:
            return score

    def get_score(self,x, y, t_i, degradation):
        if y==None:
            assert degradation==None
            #unconditional sampling
            with torch.no_grad():
                #print("In sampling", x.shape, t_i.shape)
                x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                if self.args.tester.filter_out_cqt_DC_Nyq:
                    x_hat=self.model.CQTransform.apply_hpf_DC(x_hat)
                score=(x_hat-x)/t_i**2
            return score
        else:
            if self.xi>0:
                #apply rec. guidance
                score=self.get_score_rec_guidance(x, y, t_i, degradation)
    
                #optionally apply replacement or consistency step
                #if self.data_consistency:
                #    #convert score to denoised estimate using Tweedie's formula
                #    x_hat=score*t_i**2+x
    
                #    x_hat=self.proj_convex_set(x_hat)
    
                #    #convert back to score
                #    score=(x_hat-x)/t_i**2
    
            else:
                #denoised with replacement method
                with torch.no_grad():
                    x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                    x_hat=self.proj_convex_set(x_hat.detach())

                    score=(x_hat.detach()-x)/t_i**2
                        
                    #x_hat=self.data_consistency_step(x_hat,y, degradation)
        
                    #score=(x_hat-x)/t_i**2
    
            return score

    def predict_unconditional(
        self,
        shape,  #observations (lowpssed signal) Tensor with shape ??
        device
    ):
        self.y=None
        self.degradation=None
        return self.predict(shape, device)

    def predict_resample(
        self,
        y,  #observations (lowpssed signal) Tensor with shape ??
        shape,
        degradation, #lambda function
    ):
        self.degradation=degradation 
        self.y=y
        #print(shape)
        return self.predict(shape, y.device)




    def predict(
        self,
        shape,  #observations (lowpssed signal) Tensor with shape ??
        device, #lambda function
    ):

        if self.rid:
            rid_xt=torch.zeros((self.nb_steps,shape[0], shape[1]))
            rid_grads=torch.zeros((self.nb_steps,shape[0], shape[1]))
            rid_denoised=torch.zeros((self.nb_steps,shape[0], shape[1]))
            rid_grad_update=torch.zeros((self.nb_steps,shape[0], shape[1]))
            rid_pocs=torch.zeros((self.nb_steps,shape[0], shape[1]))
            rid_xt2=torch.zeros((self.nb_steps,shape[0], shape[1]))

        #get the noise schedule
        t = self.diff_params.create_schedule(self.nb_steps).to(device)
        #sample from gaussian distribution with sigma_max variance
        x = self.diff_params.sample_prior(shape,t[0]).to(device)

        #parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic
        gamma=self.diff_params.get_gamma(t).to(device)


        for i in tqdm(range(0, self.nb_steps, 1)):
            #print("sampling step ",i," from ",self.nb_steps)

            if gamma[i]==0:
                #deterministic sampling, do nothing
                t_hat=t[i] 
                x_hat=x
            else:
                #stochastic sampling
                #move timestep
                t_hat=t[i]+gamma[i]*t[i] 
                #sample noise, Snoise is 1 by default
                epsilon=torch.randn(shape).to(device)*self.diff_params.Snoise
                #add extra noise
                x_hat=x+((t_hat**2 - t[i]**2)**(1/2))*epsilon 

            if self.rid:
                rid_xt[i]=x_hat

            score=self.get_score(x_hat, self.y, t_hat, self.degradation)    
            if self.rid:
                score, x_hat1, grads, x_hat2, x_hat3=score
                rid_denoised[i]=x_hat1
                rid_grads[i]=grads
                rid_grad_update[i]=x_hat2
                rid_pocs[i]=x_hat3


            #d=-t_hat*((denoised-x_hat)/t_hat**2)
            d=-t_hat*score
            
            #apply second order correction
            h=t[i+1]-t_hat


            if t[i+1]!=0 and self.order==2:  #always except last step
                #second order correction2
                #h=t[i+1]-t_hat
                t_prime=t[i+1]
                x_prime=x_hat+h*d
                score=self.get_score(x_prime, self.y, t_prime, self.degradation)
                if self.rid:
                    score, x_hat1, grads, x_hat2, x_hat3=score

                d_prime=-t_prime*score

                x=(x_hat+h*((1/2)*d +(1/2)*d_prime))

            elif t[i+1]==0 or self.order==1: #first condition  is to avoid dividing by 0
                #first order Euler step
                x=x_hat+h*d

            if self.rid: 
                rid_xt2[i]=x
            
        if self.data_consistency_end:
            x=self.proj_convex_set(x)

        if self.rid:
            return x.detach(), rid_denoised.detach(), rid_grads.detach(), rid_grad_update.detach(), rid_pocs.detach(), rid_xt.detach(), rid_xt2.detach(), t.detach()
        else:
            return x.detach()

    def apply_mask(self, x, mask=None):
        
        if mask is None:
            mask=self.mask

        return mask*x

    def apply_spectral_mask(self, x, mask=None):
        if mask is None:
            mask=self.mask

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
        #why is the size different? Because of the padding?
        x_masked=x_masked[...,0:input_shape[-1]]

        return x_masked


    #def proj_convex_set(self, x_hat, y, degradation):
    #    """
    #    Simple replacement method, used for inpainting and FIR bwe
    #    """
    #    #get reconstruction estimate
    #    den_rec= degradation(x_hat)     
    #    #apply replacment (valid for linear degradations)
    #    return y+x_hat-den_rec 

    def prepare_smooth_mask(self, mask, size=10):
        hann=torch.hann_window(size*2)
        hann_left=hann[0:size]
        hann_right=hann[size::]
        B,N=mask.shape
        mask=mask[0]
        prev=1
        new_mask=mask.clone()
        #print(hann.shape)
        for i in range(len(mask)):
            if mask[i] != prev:
                #print(i, mask.shape, mask[i], prev)
                #transition
                if mask[i]==0:
                   #gap encountered, apply hann right before
                   new_mask[i-size:i]=hann_right
                if mask[i]==1:
                   #gap encountered, apply hann left after
                   new_mask[i:i+size]=hann_left
                #print(mask[i-2*size:i+2*size])
                #print(new_mask[i-2*size:i+2*size])
                
            prev=mask[i]
        return new_mask.unsqueeze(0).expand(B,-1)

    def predict_inpainting(
        self,
        y_masked,
        mask
        ):
        self.mask=mask.to(y_masked.device)

        self.y=y_masked

        self.degradation=lambda x: self.apply_mask(x)
        if self.data_consistency or self.data_consistency_end:
            if self.smooth:
                smooth_mask=self.prepare_smooth_mask(mask, self.args.tester.data_consistency.hann_size)
            else:
                smooth_mask=mask

            self.proj_convex_set= lambda x: smooth_mask*y_masked+(1-smooth_mask)*x #will this work? I am too scared about lambdas


        return self.predict(self.y.shape, self.y.device)

    def predict_spectrogram_inpainting(
        self,
        y_masked,
        mask
        ):
        self.mask=mask.to(y_masked.device)

        self.y=y_masked

        self.degradation=lambda x: self.apply_spectral_mask(x)
        if self.data_consistency or self.data_consistency_end:
            smooth_mask=mask #I assume for now that the smoothness is not required for spectrogram inpainting

            self.proj_convex_set= lambda x: y_masked+x-self.apply_spectral_mask(x) #If this fails, consider using a more appropiate projection

        return self.predict(self.y.shape, self.y.device)


    def predict_STN_inpainting(
        self,
        y_masked,
        mask
        ):
        self.mask=mask.to(y_masked.device)

        self.y=y_masked

        self.degradation=lambda x: self.apply_STN_mask(x, self.mask)
        if self.data_consistency or self.data_consistency_end:
            smooth_mask=mask #I assume for now that the smoothness is not required for spectrogram inpainting

            self.proj_convex_set= lambda x: y_masked+self.apply_STN_mask(x,1-self.mask) #If this fails, consider using a more appropiate projection

        return self.predict(self.y.shape, self.y.device)
