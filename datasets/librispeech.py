# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
#import PIL.Image
import json
import torch
import utils.dnnlib as dnnlib
import random
import pandas as pd
import glob
import soundfile as sf

#try:
#    import pyspng
#except ImportError:
#    pyspng = None

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.
class LibrispeechTrain(torch.utils.data.IterableDataset):
    def __init__(self,
        dset_args,
        fs=44100,
        seg_len=131072,
        overfit=False,
        seed=42 ):
        self.overfit=overfit

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path=dset_args.path
        filelist=[]
        for d in dset_args.train_dirs:
            filelist.extend(glob.glob(os.path.join(path,d,"*/*/*.flac")))

        assert len(filelist)>0 , "error in dataloading: empty or nonexistent folder"

        self.train_samples=filelist
       
        self.seg_len=int(seg_len)
        self.fs=fs
        if self.overfit:
            file=self.train_samples[0]
            data, samplerate = sf.read(file)
            if len(data.shape)>1 :
                data=np.mean(data,axis=1)
            self.overfit_sample=data[10*samplerate:60*samplerate] #use only 50s

    def __iter__(self):
        if self.overfit:
           data_clean=self.overfit_sample
        while True:
            if not self.overfit:
                num=random.randint(0,len(self.train_samples)-1)
                #for file in self.train_samples:  
                file=self.train_samples[num]
                data, samplerate = sf.read(file)
                assert(samplerate==self.fs, "wrong sampling rate")
                segment=data
                #Stereo to mono
                if len(data.shape)>1 :
                    segment=np.mean(segment,axis=1)

            #normalize
            #no normalization!!
            #data_clean=data_clean/np.max(np.abs(data_clean))
            L=len(segment)
            #print(L, self.seg_len)
            if L>self.seg_len:
                #get random segment
                idx=np.random.randint(0,L-self.seg_len)
                segment=segment[idx:idx+self.seg_len]
            elif L<=self.seg_len:
                #pad with zeros to get to the right length randomly
                idx=np.random.randint(0,self.seg_len-L)
                #segment=np.pad(segment,(idx,self.seg_len-L-idx),'constant')
                #copy segment to get to the right length
                segment=np.pad(segment,(idx,self.seg_len-L-idx),'wrap')

                #print the std of the segment
                #print(np.std(segment))


                yield  segment
            #else:
            #    pass



class LibrispeechTest(torch.utils.data.Dataset):
    def __init__(self,
        dset_args,
        fs=44100,
        seg_len=131072,
        num_samples=4,
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path=dset_args.test.path


        filelist=glob.glob(os.path.join(path,"*/*/*.flac"))

        assert len(filelist)>0 , "error in dataloading: empty or nonexistent folder"

        self.train_samples=filelist
        self.seg_len=int(seg_len)
        self.fs=fs

        self.test_samples=[]
        self.filenames=[]
        self._fs=[]
        for i in range(num_samples):
            file=self.train_samples[i]
            self.filenames.append(os.path.basename(file))
            segment, samplerate = sf.read(file)
            print(self.fs, samplerate)
            assert samplerate==self.fs, "wrong sampling rate"
            if len(segment.shape)>1 :
                segment=np.mean(segment,axis=1)
            L=len(segment)
            if L>self.seg_len:
                #get random segment
                idx=0
                segment=segment[idx:idx+self.seg_len]
            elif L<=self.seg_len:
                #pad with zeros to get to the right length randomly
                idx=0
                #segment=np.pad(segment,(idx,self.seg_len-L-idx),'constant')
                segment=np.pad(segment,(idx,self.seg_len-L-idx),'wrap')
            self._fs.append(samplerate)

            self.test_samples.append(segment) #use only 50s

    def __getitem__(self, idx):
        #return self.test_samples[idx]
        return self.test_samples[idx],self._fs[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)
