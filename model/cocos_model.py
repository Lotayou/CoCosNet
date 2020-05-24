from model.base_model import BaseModel
import torch
from torch import nn
import torch.nn.functional as F
from model import networks
from model.translation_net import TranslationNet
from model.correspondence_net import CorrepondenceNet
from model.discriminator import Discriminator

from model.contextual_loss import symmetric_CX_loss
import itertools
'''
    Cross-Domian Correpondence Model
'''
class CoCosModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser
        
    def __init__(self, opt):
        super().__init__(opt)
        self.w = opt.image_size
        # make a folder for save images
        self.image_dir = os.path.join(self.save_dir, 'images')
        if not os.path.isdir(self.image_dir):
            os.mkdir(self.image_dir)
            
        # initialize networks
        self.model_names ['C', 'T']
        self.netC = CorrepondenceNet(opt)
        self.netT = TranslationNet(opt)
        if opt.isTrain:
            self.model_names.append('D')
            self.netD = Discriminator(opt)
                
        if opt.isTrain:
            # assign losses
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionPerc = 
            self.criterionContext = symmetric_CX_loss
            self.criterionAdv = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)
            self.criterionDomain = nn.L1Loss()
            self.criterionReg = torch.nn.L1Loss()
            self.criterion
            
            
            # initialize optimizers
            self.optT = torch.optim.Adam(self.netT.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optC = torch.optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optD = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optC, self.optT, self.optD]

        # Finally, load checkpoints and recover schedulers
        self.setup()
            
    def set_input(self, batch):
        # expecting 'a' -> 'b_gt', 'a_exemplar' -> 'b_exemplar', ('b_deform')
        # for human pose transfer, 'b_deform' is already 'b_exemplar'
        for k, v in batch.items():
            setattr(self, k, v.to(self.device))
        
    def forward(self):
        self.sa, self.sb, self.fb_warp, self.b_warp = self.netC(self.a, self.b_exemplar)  # 3*HW*HW
        self.b_gen = self.netT(self.b_warp)
        # self.b_gen = self.netT(self.fb_warp) retain original feature or use warped rgb?
        
        # TODO: Implement backward warping (maybe we should adjust the input size?)
        _, _, _, self.b_reg = self.netC(self.a_exemplar, 
            F.interpolate(self.b_warp, (self.w, self.w), mode='bilinear')
        )
        
    def backward_G(self):
        self.optG.zero_grad()
        # Damn, do we really need 6 losses?
        # 1. Perc loss
        self.loss_perc
        # 2. domain loss
        self.loss_domain = self.opt.lambda_domain * self.criterionDomain(self.sa, self.sb)
        # 3. losses for pseudo exemplar pairs
        self.loss_feat = self.opt.lambda_feat * self.criterionFeat(self.b_gen, self.b_gt)
        # 4. Contextural loss
        self.loss_context = self.opt.lambda_context * self.criterionContext(self.b_gen, self.b_exemplar)
        # 5. Reg loss
        self.loss_reg = self.opt.lambda_reg * self.criterionReg(self.b_reg, self.b_exemplar)
        # 6. GAN loss
        # TODO
        self.loss_adv = 0
        g_loss = self.loss_perc + self.loss_domain + self.loss_feat \
            + self.loss_context + self.loss_reg + self.loss_adv
            
        g_loss.backward()
        self.optG.step()
    
    def backward_D(self):
        pass
        
    def optimize_parameters(self):
        pass
        
    ### Standalone utility functions
    def log_loss(self, epoch, iter):
        pass
        
    def log_visual(self, epoch, iter):
        
