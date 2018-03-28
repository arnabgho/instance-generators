import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .infogan_instance_base_model import BaseModel
from . import networks
import random

class InfoganInstance(BaseModel):
    def name(self):
        return 'InfoganInstance'

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
	self.z=Variable(torch.zeros(self.max_masks,self.ndiscrete+self.ncontinuous+self.nnoise))

    def _noise_samples(self):
	assert self.ndiscrete<=10
	activated=np.random.choice(10,self.ndiscrete,replace=False)
	discrete=np.zeros(10)
	discrete[activated]=1

	discrete=Variable(torch.Tensor(discrete))

	continuous=torch.Tensor(np.zeros(self.ncontinuous))
	continuous=continuous.normal_(0,1)

	noise = torch.Tensor(np.zeros(self.nnoise))
	noise=noise.normal_(0,1)
	return torch.cat([discrete,continuous,noise],1)


    def forward(self):
	assert self.nmasks<self.max_masks

	self.random_z_ids=np.random.choice(self.max_masks,self.nmasks,replace=False)
        self.input=Variable(torch.zeros(1 ,self.ndiscrete + self.ncontinuous + self.nnoise , self.height , self.width))

        i=0
	for z_id in self.random_z_ids:
            self.z[z_id]=self._noise_samples()
            input_z_id=torch.zeros(1 ,self.ndiscrete + self.ncontinuous + self.nnoise , self.height , self.width)
            input_z_id=self.z[z_id].view(1,self.ndiscrete+self.ncontinuous+self.nnoise,1,1).expand_as(self.input)
            mask_reshaped=self.masks[i].view(1,1,self.height,self.width).expand_as(self.input)
            input_z_id=input_z_id.mul(mask_reshaped)

            input=input + input_z_id

            i=i+1
            if i==self.nmasks:
                break

       self.output_image=self.netG.forward(input)
       return self.output_image


    def backward_D(self,netD,real,fake):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake=netD.forward(self.FE.forward(fake.detach()))
        # Real
        pred_real=netD.forwad(self.FE.forward(real))
        loss_D_fake,losses_D_fake=self.criterionGAN(pred_fake,False)
        loss_D_real,losses_D_real=self.criterionGAN(pred_real,True)
        # Combined loss
        loss_D=loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake,loss_D_real]


    def backward_G_GAN(self,fake,netD=None):
        pred_fake=netD.forward(self.FE.forward(fake))
        loss_G_GAN,losses_G_GAN=self.criterionGAN(pred_fake,True)
        loss_G_GAN.backward()
        return loss_G_GAN


    def backward_G_Q(self,fake):
        loss_Q=0
        for i in range(self.nmasks):
            id_z = self.random_z_ids[i]
            mask_reshaped=self.masks[i].view(1,1,self.height,self.width).expand_as(fake)
            mask_output = fake.mul(mask_reshaped)

            q_logits, q_mu, q_var = self.Q(self.FE(mask_output))

            target_logits= self.z[id_z][0][0:ndiscrete]
            continuous_target = self.z[id_z][0][ndiscrete:ndiscrete+ncontinuous]

            discrete_loss = criterionQ_dis(q_logits, target_logits)
            continuous_loss = criterionQ_con(continuous_target, q_mu, q_var)

            continuous_loss.backward()
