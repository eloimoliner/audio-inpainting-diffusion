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
class AudioFolderDatasetTest(torch.utils.data.Dataset):
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

        filelist=glob.glob(os.path.join(path,"*.wav"))
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
            data, samplerate = sf.read(file)
            self._fs.append(samplerate)
            if len(data.shape)>1 :
                data=np.mean(data,axis=1)
            self.test_samples.append(data[2*samplerate:2*samplerate+self.seg_len]) #use only 50s


    def __getitem__(self, idx):
        #return self.test_samples[idx]
        return self.test_samples[idx], self._fs[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)

