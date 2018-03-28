import os
import torch
import util.util as util
from torch.autograd import Variable
from pdb import set_trace as st
from . import networks
import itertools


class log_gaussian():
    def __call__(self, x, mu, var):
	logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
		(x-mu).pow(2).div(var.mul(2.0)+1e-6)
    	return logli.sum(1).mean().mul(-1)


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)


    def init_data(self,opt,use_D=True,use_Q=True,max_instances=10):
        # Inputs
        self.max_instances=max_instances
        self.input_imgs = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize,opt.fineSize)
        self.masks = self.Tensor(opt.batchSize, opt.max_instances, opt.fineSize, opt.fineSize)
        self.ninstances=0

        # Criterion
        # if opt.isTrain:
        use_sigmoid = opt.gan_mode == 'dcgan'


        self.criterionGAN=networks.GANLoss(mse_loss= not use_sigmoid, tensor=self.Tensor)  #nn.BCELoss().cuda()
        self.criterionQ_dis = nn.CrossEntropyLoss().cuda()
        self.criterionQ_con = log_gaussian()

        # Optimizers

        self.optimizers=[]
        self.optimD = optim.Adam([{'params':self.FE.parameters()}, {'params':self.netD.parameters()}], lr=opt.lr, betas=(opt.beta1, 0.99))
        self.optimizers.append(self.optimD)
        self.optimG = optim.Adam([{'params':self.netG.parameters()}, {'params':self.Q.parameters()}], lr=opt.lr, betas=(opt.beta1, 0.99))
        self.optimizers.append(self.optimG)

        for optimizer in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def is_skip(self):
        return False

    def forward(self):
        pass

    def eval(self):
        pass

    def set_requires_grad(self, net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad  # to avoid computation

    def balance(self):
        pass

    def get_image_paths(self):
        pass

    def update_D(self, data):
        pass

    def update_G(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device_id=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def load_network_test(self, network, network_path):
        save_path = os.path.join(self.save_dir, network_path)
        network.load_state_dict(torch.load(network_path))

    def update_learning_rate(self):
        loss = self.get_measurement()
        for scheduler in self.schedulers:
            scheduler.step(loss)
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_measurement(self):
        return None

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = self.Tensor(batchSize, nz)
        if random_type == 'uni':
            z.copy_(torch.rand(batchSize, nz) * 2.0 - 1.0)
        elif random_type == 'gauss':
            z.copy_(torch.randn(batchSize, nz))
        z = Variable(z)
        return z

    # testing models
    #def set_input(self, input):
    #    # get direciton
    #    AtoB = self.opt.which_direction == 'AtoB'
    #    # set input images
    #    input_A = input['A' if AtoB else 'B']
    #    input_B = input['B' if AtoB else 'A']
    #    self.input_A.resize_(input_A.size()).copy_(input_A)
    #    self.input_B.resize_(input_B.size()).copy_(input_B)
    #    # get image paths
    #    self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input(self, sample):
        self.ninstances=random.randint(1,sample.num_instances)
        self.selected_instances=random.sample(self.ninstances)
        self.input_imgs= sample.image

        for i in range(sample.ninstances):
            self.masks[0][i]=sample.instance_images[self.selected_instances [i]]

    def get_image_paths(self):
        return self.image_paths

    def test(self, z_sample):  # need to have input set already
        self.real_A = Variable(self.input_A, volatile=True)
        batchSize = self.input_A.size(0)
        z = self.Tensor(batchSize, self.opt.nz)
        z_torch = torch.from_numpy(z_sample)
        z.copy_(z_torch)
        # st()
        self.z = Variable(z, volatile=True)
        self.fake_B = self.netG.forward(self.real_A, self.z)
        self.real_B = Variable(self.input_B, volatile=True)

    def encode(self, input_data):
        return self.netE.forward(Variable(input_data, volatile=True))

    def encode_real_B(self):
        self.z_encoded = self.encode(self.input_B)
        return util.tensor2vec(self.z_encoded)

    def real_data(self, input=None):
        if input is not None:
            self.set_input(input)
        return util.tensor2im(self.input_A), util.tensor2im(self.input_B)

    def test_simple(self, z_sample, input=None, encode_real_B=False):
        if input is not None:
            self.set_input(input)

        if encode_real_B:  # use encoded z
            z_sample = self.encode_real_B()

        self.test(z_sample)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return self.image_paths, real_A, fake_B, real_B, z_sample
