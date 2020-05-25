import torch
from torch import nn
import torch.nn.functional as F
import model.networks
import itertools

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

    20200525: Potential Bug: Insufficient memory to support 4096*4096 correspondence, retreat to 1024*1024 instead
'''
class CorrespondenceNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        print('Making a CorrespondenceNet')
        # domain adaptors are not shared
        ngf = opt.ngf
        self.domainA_adaptor = self.create_adaptor(opt.ncA, ngf)
        self.domainB_adaptor = self.create_adaptor(opt.ncB, ngf)
        self.softmax_alpha = 100
        ada_blocks = []
        for i in range(4):
            ada_blocks += [BasicBlock(ngf*4, ngf*4)]
            
        ada_blocks += [nn.Conv2d(ngf*4, ngf*4, kernel_size=1, stride=1, padding=0)]
        self.adaptive_feature_block = nn.Sequential(*ada_blocks)
        
        self.to_rgb = nn.Conv2d(ngf*4, 3, kernel_size=1, stride=1, padding=0)
    
    @staticmethod
    def warp(fa, fb, b_raw, alpha):
        '''
            calculate correspondence matrix and warp the exemplar features
        '''
        assert fa.shape == fb.shape, \
            'Feature shape must match. Got %s in a and %s in b)' % (a.shape, b.shape)
        n,c,h,w = fa.shape
        # subtract mean
        fa = fa - torch.mean(fa, dim=(2,3), keepdim=True)
        fb = fb - torch.mean(fb, dim=(2,3), keepdim=True)
        
        # vectorize (merge dim H, W) and normalize channelwise vectors
        fa = fa.view(n, c, -1)
        fb = fb.view(n, c, -1)
        fa = fa / torch.norm(fa, dim=1, keepdim=True)
        fb = fb / torch.norm(fb, dim=1, keepdim=True)
        
        # correlation matrix, gonna be huge (4096*4096)
        # use matrix multiplication for CUDA speed up
        # Also, calculate the transpose of the atob correlation

        # warp the exemplar features b, taking softmax along the b dimension
        corr_ab_T = F.softmax(torch.bmm(fb.transpose(-2,-1), fa), dim=2) # n*HW*C @ n*C*HW -> n*HW*HW
        #print(corr_ab_T.shape)
        #print(softmax_weights.shape, b_raw.shape)
        b_warp = torch.bmm(b_raw.view(n, c, h*w), corr_ab_T) # n*HW*1
        return b_warp.view(n,c,h,w)
        
    def create_adaptor(self, nc, ngf):
        model_parts = [self.combo(nc, ngf, 3, 1, 1),
            self.combo(ngf, ngf*2, 4, 2, 1),
            self.combo(ngf*2, ngf*4, 3, 1, 1),
            self.combo(ngf*4, ngf*8, 4, 2, 1),
            self.combo(ngf*8, ngf*8, 3, 1, 1),
            # The following line shrinks the spatial dimension to 32*32
            self.combo(ngf*8, ngf*8, 4, 2, 1),
            [BasicBlock(ngf*8, ngf*4)],
            [BasicBlock(ngf*4, ngf*4)],
            [BasicBlock(ngf*4, ngf*4)]
        ]
        model = itertools.chain(*model_parts)
        return nn.Sequential(*model)
        
    def combo(self, cin, cout, kw, stride, padw):
        layers = [
            nn.Conv2d(cin, cout, kernel_size=kw, stride=stride, padding=padw),
            nn.InstanceNorm2d(cout),
            nn.LeakyReLU(0.2),
        ]
        return layers
        
    def forward(self, a, b):
        sa = self.domainA_adaptor(a)
        sb = self.domainB_adaptor(b)
        fa = self.adaptive_feature_block(sa)
        fb = self.adaptive_feature_block(sb)
        # This should be sb, but who knows?
        b_warp = self.warp(fa, fb, b_raw=sb, alpha=self.softmax_alpha)
        b_img = F.tanh(self.to_rgb(b_warp))
        return sa, sb, b_warp, b_img
        
# Basic residual block
class BasicBlock(nn.Module):
    def __init__(self, cin, cout):
        super(BasicBlock, self).__init__()
        layers = [
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(cout),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(cout),
        ]
        self.conv = nn.Sequential(*layers)
        if cin != cout:
            self.shortcut = nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = lambda x:x

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out
