import torch
from torch import nn
import torch.nn.functional as F
import model.networks

'''
    CorrespondenceNet: Align images in different domains
        into a shared domain S, and compute the correlation
        matrix (vectorized)
        
    Note that a is guidance, b is exemplar
    e.g. for human pose transfer
    a is the target pose, b is the source image
    
    output: b_warp: a 3*H*W image
    -----------------
    # TODO: Add sychonized batchnorm to support multi-GPU training
'''
class CorrespondenceNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        print('Making a CorrespondenceNet')
        # domain adaptors are not shared
        self.domainA_adaptor = self.create_adaptor(opt.ncA, opt.ngf)
        self.domainB_adaptor = self.create_adaptor(opt.ncB, opt.ngf)
        self.softmax_alpha = 100
        ada_blocks = []
        for i in range(4):
            ada_blocks += [BasicBlock(ngf*4, ngf*4)]
            
        ada_blocks += [nn.Conv2d(ngf*4, ngf*4, kernel_size=1, stride=1, padding=0)]
        self.adaptive_feature_block = nn.Sequential(*ada_blocks)
        
        self.to_rgb = nn.Conv2d(ngf*4, 3, kernel_size=1, stride=1, padding=0)
    
    @staticmethod
    def warp(fa, fb, b_raw):
        '''
            calculate correspondence matrix and warp the exemplar features
        '''
        assert fa.shape == fb.shape, \
            'Feature shape must match. Got %s in a and %s in b)' % (a.shape, b.shape)
        n,c,h,w = a.shape
        # subtract mean
        fa -= torch.mean(fa, dim=(2,3), keepdim=True)
        fb -= torch.mean(fb, dim=(2,3), keepdim=True)
        
        # vectorize (merge dim H, W) and normalize channelwise vectors
        fa = fa.view(n, c, -1)
        fb = fb.view(n, c, -1)
        fa /= torch.norm(fa, dim=1, keepdim=True)
        fb /= torch.norm(fb, dim=1, keepdim=True)
        
        # correlation matrix, gonna be huge (4096*4096)
        # use matrix multiplication for CUDA speed up
        corr_ab = torch.bmm(fa.transpose(-2,-1), fb) # n*HW*C @ n*C*HW -> n*HW*HW
        
        # warp the exemplar features b, taking softmax along the b dimension
        softmax_weights = F.softMax(corr_ab * self.softmax_alpha, dim=2)
        b_warp = torch.bmm(softmax_weights, b_raw.view(n, c, h*w, 1)) # n*HW*1
        return b_warp.view(n,c,h,w)
        
    def create_adaptor(self, nc, ngf):
        model = self.combo(nc, ngf, 3, 1, 1) \
            + self.combo(ngf, ngf*2, 4, 2, 1) \
            + self.combo(ngf*2, ngf*4, 3, 1, 1) \
            + self.combo(ngf*4, ngf*8, 4, 2, 1) \
            + self.combo(ngf*8, ngf*8, 3, 1, 1)
            + [BasicBlock(ngf*8, ngf*4)] \
            + [BasicBlock(ngf*4, ngf*4)] \
            + [BasicBlock(ngf*4, ngf*4)]
            
        return nn.Sequential(*model)
        
    def combo(self, cin, cout, kw, stride, padw):
        layers [
            nn.Conv2d(cin, cout, kernel_size=kw, stride=stride, padding=padw),
            nn.InstanceNorm2D(cout),
            nn.LeakyReLU(0.2),
        ]
        return layers
        
    def forward(self, a, b):
        sa = self.domainA_adaptor(a)
        sb = self.domainB_adaptor(b)
        fa = self.adaptive_feature_block(sa)
        fb = self.adaptive_feature_block(fb)
        # This should be sb, but who knows?
        b_warp = self.warp(fa, fb, b_raw=sb)
        b_img = F.tanh(self.to_rgb(b_warp))
        return sa, sb, b_warp, b_img
        
# Basic residual block
class BasicBlock(nn.Module):
    def __init__(self, cin, cout):
        super(BasicBlock, self).__init__()
        layers = [
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2D(cout),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2D(cout),
        ]
        self.conv = nn.Sequential(*layers)
        if cin != cout:
            self.shortcut = nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = lambda x:x

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out