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
class MaestroDatasetTestChunks(torch.utils.data.Dataset):
    def __init__(self,
        dset_args,
        num_samples=4,
        seed=42 ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path=dset_args.path
        years=dset_args.years

        self.seg_len=int(dset_args.load_len)

        metadata_file=os.path.join(path,"maestro-v3.0.0.csv")
        metadata=pd.read_csv(metadata_file)

        metadata=metadata[metadata["year"].isin(years)]
        metadata=metadata[metadata["split"]=="test"]
        filelist=metadata["audio_filename"]

        filelist=filelist.map(lambda x:  os.path.join(path,x)     , na_action='ignore')


        self.filelist=filelist.to_list()

        self.test_samples=[]
        self.filenames=[]
        self.f_s=[]
        for i in range(num_samples):
            file=self.filelist[i]
            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            if len(data.shape)>1 :
                data=np.mean(data,axis=1)

            self.test_samples.append(data[10*samplerate:10*samplerate+self.seg_len]) #use only 50s
            self.f_s.append(samplerate)
       

    def __getitem__(self, idx):
        return self.test_samples[idx], self.f_s[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)


