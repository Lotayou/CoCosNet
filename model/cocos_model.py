from model.base_model import BaseModel
import torch
from torch import nn
import torch.nn.functional as F
from model import networks
from model.translation_net import TranslationNet
from model.correspondence_net import CorrepondenceNet
from model.discriminator import Discriminator

from model.loss import VGGLoss, GANLoss
import itertools
'''
    Cross-Domian Correpondence Model
'''
class CoCosModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser
    
    @staticmethod
    def torch2numpy(x):
        # from [-1,1] to [0,255]
        return ((x.detach().cpu().numpy().transpose(1,2,0) + 1) * 127.5).astype(np.uint8)
        
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
                
        self.visual_names = ['b_exemplar', 'a', 'b_gen', 'b_gt']  # HPT convention
        
        if opt.isTrain:
            # assign losses
            self.loss_names = ['perc', 'domain', 'feat', 'context', 'reg', 'adv']
            self.visual_names += ['b_warp']
            self.criterionFeat = torch.nn.L1Loss()
            # Both interface for VGG and perceptual loss
            # call with different mode and layer params
            self.criterionVGG = VGGLoss(self.device)
            # Support hinge loss
            self.criterionAdv = GANLoss(gan_mode=opt.gan_mode).to(self.device)
            self.criterionDomain = nn.L1Loss()
            self.criterionReg = torch.nn.L1Loss()
            
            
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
        
    def test(self):
        with torch.no_grad():
            _, _, _, self.b_warp = self.netC(self.a, self.b_exemplar)  # 3*HW*HW
            self.b_gen = self.netT(self.b_warp)
        
    def backward_G(self):
        self.optG.zero_grad()
        # Damn, do we really need 6 losses?
        # 1. Perc loss(For human pose transfer we abandon it, it's all in the criterion Feat)
        self.loss_perc = 0 
        # 2. domain loss
        self.loss_domain = self.opt.lambda_domain * self.criterionDomain(self.sa, self.sb)
        # 3. losses for pseudo exemplar pairs
        self.loss_feat = self.opt.lambda_feat * self.criterionVGG(self.b_gen, self.b_gt, mode='perceptual')
        # 4. Contextural loss
        self.loss_context = self.opt.lambda_context * self.criterionVGG(self.b_gen, self.b_exemplar, mode='contextual', layers=[2,3,4,5])
        # 5. Reg loss
        self.loss_reg = self.opt.lambda_reg * self.criterionReg(self.b_reg, self.b_exemplar)
        # 6. GAN loss
        pred_real, pred_fake = self.discriminate(self.b_gt, self.b_gen)
        self.loss_adv = self.opt.lambda_adv * self.criterionAdv(pred_fake, True, for_discriminator=False)
        
        g_loss = self.loss_perc + self.loss_domain + self.loss_feat \
            + self.loss_context + self.loss_reg + self.loss_adv
            
        g_loss.backward()
        self.optG.step()
    
    def discriminate(self, real, fake):
        fake_and_real = torch.cat([fake, real], dim=0)
        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    
    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if isinstance(pred, list):
            fake = [p[:tensor.size(0) // 2] for p in pred])
            real = [p[tensor.size(0) // 2:] for p in pred])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real
    
    def backward_D(self):
        self.optD.zero_grad()
        # test, run under no_grad mode
        self.test()

        pred_fake, pred_real = self.discriminate(self.b_gt, self.b_gen)

        self.d_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        self.d_real = self.criterionGAN(pred_real, True, for_discriminator=True)
        
        d_loss = (self.d_fake + self.d_real) / 2
        d_loss.backward()
        self.optD.step()
        
    def optimize_parameters(self):
        # must call self.set_input(data) first
        self.forward()
        self.backward_G()
        self.backward_D()
        
    ### Standalone utility functions
    def log_loss(self, epoch, iter):
        msg = 'Epoch %d iter %d\n  ' % (epoch, iter)
        for name in self.loss_names:
            val = getattr(self, 'loss_%s' % name).item()
            msg += '%s: %.4f, ' % (name, val)
        print(msg)
        
    def log_visual(self, epoch, iter):
        save_path = os.path.join(self.save_image_dir, 'epoch%03d_iter%05d.png' % (epoch, iter))
        # warped image is not the same resolution, need scaling
        self.b_warp = F.interpolate(self.b_warp, (self.w, self.w), mode='bicubic')
        pack = torch.cat(
            [getattr(self, name) for name in self.visual_names], dim=3
        )[0] # only save one example
        cv2.imwrite(save_path, self.torch2numpy(pack))
        
    def update_learning_rate(self):
        '''
            Update learning rates for all the networks;
            called at the end of every epoch by train.py
        '''
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate updated to %.7f' % lr)
