import os
import torch
import torch.nn as nn
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


class InstanceDataset(Dataset):
    """ Instance dataset"""
    def __init__(self,metadata_file,img_root_dir,instance_root_dir,transform=None):
        self.metadata_file=metadata_file
        self.img_root_dir=img_root_dir
        self.instance_root_dir=instance_root_dir
        metadata_f=open(metadata_file)
        metadata_lines=metadata_f.readlines()

        self.names=[]
        self.numbers=[]
        for metadata_line in metadata_lines:
            name,number= metadata_line.split(' ')
            number = int(number.strip())
            self.names.append(name)
            self.numbers.append(number)

    def __len__(self):
        return len(self.names)

    def __getitem__(self,idx):
        name=self.names[idx]
        number=self.numbers[idx]
        city=name.split('_')[0]
        img_name=os.path.join(self.img_root_dir,city,name)
        img=io.imread(img_name)
        img=img.clip(0,1)
        inst_imgs=[]
        for i in range(number):
            inst_img_name=os.path.join(self.instance_root_dir,city,str(i)+'_'+name)
            inst_img=io.imread(inst_img_name)
            inst_imgs.append(inst_img)

        sample={ 'image':img , 'instance_images':inst_imgs  }
        return sample
