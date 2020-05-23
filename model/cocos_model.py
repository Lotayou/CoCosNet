from model.base_model import BaseModel
import torch
from torch import nn
import torch.nn.functional as F
from model import networks
from model.translation_net import TranslationNet
from model.correspondence_net import CorrepondenceNet
form model.discriminator import Discriminator

'''
    Cross-Domian Correpondence Model
'''
class CoCosModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser
        
    def __init__(self, opt):
        super().__init__(opt)
        
        # make a folder for save images
        self.image_dir = os.path.join(self.save_dir, 'images')
        if not os.path.isdir(self.image_dir):
            os.mkdir(self.image_dir)
            
        # initialize networks
        self.netC = CorrepondenceNet(opt)
        self.netT = TranslationNet(opt)
        if opt.isTrain:
            self.netD = Discriminator(opt)
            
        if opt.continue_train:
            self.load_networks()
                
        if opt.isTrain:
            # assign losses
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionZ = torch.nn.L1Loss()
            
            # initialize optimizers
            self.optT = torch.optim.Adam(self.netT.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optC = torch.optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optD = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optC, self.optT, self.optD]

        # Finally, load checkpoints and recover schedulers
        self.setup()
            
        
    def set_input(self, batch):
        pass
        
    def forward(self):
        pass
        
    def optimize_parameters(self):
        pass
        
    ### Standalone utility functions
    def log_loss(self, epoch, iter):
        pass
        
    def log_visual(self, epoch, iter):
        
        