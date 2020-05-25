import torch
from torch import nn
import torch.nn.functional as F
import model.networks
from torch.nn.utils import spectral_norm

# Also, we figure it would be better to inject the warped
# guidance at the beginning rather than a constant tensor

class TranslationNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        print('Making a TranslationNet')
        self.fc = nn.Conv2d(3, 16 * opt.ngf, 3, padding=1)
        self.sw = opt.image_size // (2**5)  # fixed, 5 upsample layers
        self.head = SPADEResBlk(16 * opt.ngf, 16 * opt.ngf, opt.seg_dim)
        self.G_middle_0 = SPADEResBlk(16 * opt.ngf, 16 * opt.ngf, opt.seg_dim)
        self.G_middle_1 = SPADEResBlk(16 * opt.ngf, 16 * opt.ngf, opt.seg_dim)
        self.up_0 = SPADEResBlk(16 * opt.ngf, 8 * opt.ngf, opt.seg_dim)
        self.up_1 = SPADEResBlk(8 * opt.ngf, 4 * opt.ngf, opt.seg_dim)
        self.non_local = NonLocalLayer(opt.ngf*4)
        self.up_2 = SPADEResBlk(4 * opt.ngf, 2 * opt.ngf, opt.seg_dim)
        self.up_3 = SPADEResBlk(2 * opt.ngf, 1 * opt.ngf, opt.seg_dim)
        
        self.conv_img = nn.Conv2d(opt.ngf, 3, kernel_size=3, stride=1, padding=1)
    
    @staticmethod
    def up(x): 
        return F.interpolate(x, scale_factor=2, mode='bilinear')
        
    def forward(self, x, seg=None):
        if seg is None:
            seg = x
        # separate execute
        x = F.interpolate(x, (self.sw, self.sw), mode='bilinear') # how can I forget this one?
        x = self.fc(x)
        x = self.head(x, seg)

        x = self.up(x)    # 16
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)

        x = self.up(x)    # 32
        x = self.up_0(x, seg)
        x = self.up(x)    # 64
        x = self.up_1(x, seg)
        x = self.up(x)    # 128
        
        # 20200525: Critical Bug:
        # Using non-local layer with such a huge spatial resolution (128*128)
        # occupied way too much memory (as the intermediate tensor is O(h ** 4) memory)
        # I sincerely hope it's an honest mistake:)
        # x = self.non_local(x)
        
        x = self.up_2(x, seg)
        x = self.up(x)    # 256
        x = self.up_3(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x


# NOTE: The SPADE implementation will slightly 
# differ from the original https://github.com/NVlabs/SPADE
# where BN will be replaced with PN.
class SPADE(nn.Module):
    def __init__(self, cin, seg_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(seg_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.alpha = nn.Conv2d(128, cin,
            kernel_size=3, stride=1, padding=1)
        self.beta = nn.Conv2d(128, cin,
            kernel_size=3, stride=1, padding=1)
            
    @staticmethod
    def PN(x):
        '''
            positional normalization: normalize each positional vector along the channel dimension
        '''
        assert len(x.shape) == 4, 'Only works for 4D(image) tensor'
        x = x - x.mean(dim=1, keepdim=True)
        x_norm = x.norm(dim=1, keepdim=True) + 1e-6
        x = x / x_norm
        return x
        
    def DPN(self, x, s):
        h, w = x.shape[2], x.shape[3]
        s = F.interpolate(s, (h, w), mode='bilinear')
        s = self.conv(s)
        a = self.alpha(s)
        b  = self.beta(s)
        return x * (1 + a) + b

    def forward(self, x, s):
        x_out = self.DPN(self.PN(x), s)
        return x_out
        
class SPADEResBlk(nn.Module):
    def __init__(self, fin, fout, seg_fin):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(fin, seg_fin)
        self.norm_1 = SPADE(fmiddle, seg_fin)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, seg_fin)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class NonLocalLayer(nn.Module):
    # Non-local layer for 2D shape
    def __init__(self, cin):
        super().__init__()
        self.cinter = cin // 2
        self.theta = nn.Conv2d(cin, self.cinter, 
            kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(cin, self.cinter, 
            kernel_size=1, stride=1, padding=0)
        self.g = nn.Conv2d(cin, self.cinter, 
            kernel_size=1, stride=1, padding=0)
        
        self.w = nn.Conv2d(self.cinter, cin,
            kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        n, c, h, w = x.shape
        g_x = self.g(x).view(n, self.cinter, -1)
        phi_x = self.phi(x).view(n, self.cinter, -1)
        theta_x = self.theta(x).view(n, self.cinter, -1)
        # This non-local layer here occupies too much memory...
        print(phi_x.shape, theta_x.shape)
        f_x = torch.bmm(phi_x.transpose(-1,-2), theta_x) # note the transpose here
        f_x = F.softmax(f_x, dim=-1)
        res_x = self.w(torch.bmm(g_x, f_x))  # inverse order to save a permute of g_x
        return x + res_x
