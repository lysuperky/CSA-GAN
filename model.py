# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from collections import OrderedDict
from sync_batchnorm import SynchronizedBatchNorm2d
from torch.autograd import Variable
from GlobalAttention import SpatialAttentionGeneral as SPATIAL_ATT
from GlobalAttention import ChannelAttention as CHANNEL_ATT
from GlobalAttention import DCMChannelAttention as DCM_CHANNEL_ATT
from GlobalAttention import GlobalAttentionGeneral as ATT_NET
import torchvision as tv
from thop import clever_format
from thop import profile
import torch
from torch.autograd import Variable

BatchNorm = SynchronizedBatchNorm2d
def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)

class ACM(nn.Module):
    def __init__(self, channel_num):
        super(ACM, self).__init__()
        self.conv = conv3x3(64, 128)
        self.conv_weight = conv3x3(128, channel_num)    # weight
        self.conv_bias = conv3x3(128, channel_num)      # bias

    def forward(self, x, img):
        out_code = self.conv(img)
        out_code_weight = self.conv_weight(out_code)
        out_code_bias = self.conv_bias(out_code)
        return x * out_code_weight + out_code_bias

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.InstanceNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.InstanceNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

# class DCM_NEXT_STAGE(nn.Module):
#     def __init__(self, ngf, nef, ncf):
#         super(DCM_NEXT_STAGE, self).__init__()
#         self.gf_dim = ngf
#         self.ef_dim = nef
#         self.cf_dim = ncf
#         self.num_residual = 2
#         self.define_module()

#     def _make_layer(self, block, channel_num):
#         layers = []
#         for i in range(2):
#             layers.append(block(channel_num))
#         return nn.Sequential(*layers)

#     def define_module(self):
#         ngf = self.gf_dim
#         self.att = SPATIAL_ATT(ngf, self.ef_dim)
#         self.color_channel_att = DCM_CHANNEL_ATT(ngf, self.ef_dim)
#         self.residual = self._make_layer(ResBlock, ngf * 3)

#         self.block = nn.Sequential(
#             conv3x3(ngf * 3, ngf * 2),
#             nn.InstanceNorm2d(ngf * 2),
#             GLU())

#         self.SAIN = ACM(ngf * 3)

#     def forward(self, h_code, c_code, word_embs, mask, img):

#         self.att.applyMask(mask)
#         c_code, att = self.att(h_code, word_embs)
#         c_code_channel, att_channel = self.color_channel_att(c_code, word_embs, h_code.size(2), h_code.size(3))
#         c_code = c_code.view(word_embs.size(0), -1, h_code.size(2), h_code.size(3))

#         h_c_code = torch.cat((h_code, c_code), 1)
#         h_c_c_code = torch.cat((h_c_code, c_code_channel), 1)
#         h_c_c_img_code = self.SAIN(h_c_c_code, img)

#         out_code = self.residual(h_c_c_img_code)
#         out_code = self.block(out_code)

#         return out_code


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = 100 + ncf
        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64




class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.InstanceNorm2d(out_planes * 2),
        GLU())
    return block


class DCM(nn.Module):
    def __init__(self):
        super(DCM, self).__init__()
        ngf = 32
        nef = 256
        ncf = 100
        self.img_net = GET_IMAGE_G(ngf)
        self.h_net = DCM_NEXT_STAGE(ngf, nef, ncf)
        self.SAIN = ACM(ngf)
        self.upsample = upBlock(nef//2, ngf)
    

    def forward(self,x, rel_feat, sent_emb, word_embs, mask, c_code):
        r_code = self.upsample(rel_feat)
        tmp_code = self.h_net(x, c_code, words_embs, mask, r_code)
        tmp_r_code = self,SAIN(tmp_code, r_code)
        fake_img = self.img_net(tmp_r_code)

        return fake_img







class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])



class CA(nn.Module):
    def __init__(self):
        super(CA, self).__init__()
        self.t_dim = 256
        self.c_dim = 100
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias = True)
        self.relu = GLU()
    

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, :self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar
    

        
class NetG(nn.Module):
    def __init__(self, ngf=64, nz=100):
        super(NetG, self).__init__()
        self.ngf = ngf

        self.conv_mask = nn.Sequential(nn.Conv2d(8 * ngf, 100, 3, 1, 1),
                                       BatchNorm(100),
                                       nn.ReLU(),
                                       nn.Conv2d(100, 1, 1, 1, 0))

        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)
        self.block0 = G_Block(ngf * 8, ngf * 8)  # 4x4
        self.block1 = G_Block(ngf * 8, ngf * 8)  # 8x8
        self.block2 = G_Block(ngf * 8, ngf * 8)  # 16x16
        self.block3 = G_Block(ngf * 8, ngf * 8)  # 32x32
        self.block4 = G_Block(ngf * 8, ngf * 4)  # 64x64
        self.block5 = G_Block(ngf * 4, ngf * 2)  # 128x128
        self.block6 = G_Block(ngf * 2, ngf * 1, predict_mask=False)  # 256x256

 
        # self.alpha = alpha
        # print(x)#其实查询的是x.data,是个tensor
        self.conv_img = nn.Sequential(
            BatchNorm(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )
        ########## TODO
        '''
        in that paper and code, args are noise, sent_emb, words_embs, mask,
                                             cnn_code, region_features

        and in paper which mentioned moudle named ACM,which is a part to combinate text feature and image 
        feature , text-image affine combination module (ACM)
        fake_imgs, attention_maps, _, _, _, _ = netG(noise, sent_emb, words_embs, mask,
                                             cnn_code, region_features)
        noise.data.normal_(0, 1)


        fake, _ = netG(noise, sent_emb, cnn_code, region_features, c_code,mask)

        '''
        
        self.h_net = INIT_STAGE_G(ngf * 16 , 100)
        self.acm_net = ACM(self.ngf)
        self.img_net = GET_IMAGE_G(self.ngf)
        self.imgUpSample = upBlock(ngf , ngf*8)
        # fake, _ = netG(noise, sent_emb, cnn_code, region_features, c_code,mask)

    # def forward(self,x, c, cnn_code, region_features, c_code,mask):
    def forward(self, x, c, c_code):
        
        
        out = self.fc(x)
        out = out.view(x.size(0), 8 * self.ngf, 4, 4)
        hh, ww = out.size(2), out.size(3)
        stage_mask = self.conv_mask(out)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_4 = fusion_mask
        out, stage_mask = self.block0(out, c, fusion_mask)

        out = F.interpolate(out, scale_factor=2)
        hh, ww = out.size(2), out.size(3)
        stage_mask = F.interpolate(stage_mask, size=(hh, ww), mode='bilinear', align_corners=True)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_8 = fusion_mask
        out, stage_mask = self.block1(out, c, fusion_mask)

        out = F.interpolate(out, scale_factor=2)
        hh, ww = out.size(2), out.size(3)
        stage_mask = F.interpolate(stage_mask, size=(hh, ww), mode='bilinear', align_corners=True)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_16 = fusion_mask
        out, stage_mask = self.block2(out, c, fusion_mask)

        out = F.interpolate(out, scale_factor=2)
        hh, ww = out.size(2), out.size(3)
        stage_mask = F.interpolate(stage_mask, size=(hh, ww), mode='bilinear', align_corners=True)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_32 = fusion_mask
        out, stage_mask = self.block3(out, c, fusion_mask)

        out = F.interpolate(out, scale_factor=2)
        hh, ww = out.size(2), out.size(3)
        stage_mask = F.interpolate(stage_mask, size=(hh, ww), mode='bilinear', align_corners=True)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_64 = fusion_mask
        out, stage_mask = self.block4(out, c, fusion_mask)

        out = F.interpolate(out, scale_factor=2)
        hh, ww = out.size(2), out.size(3)
        stage_mask = F.interpolate(stage_mask, size=(hh, ww), mode='bilinear', align_corners=True)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_128 = fusion_mask
        # stage_mask2 = torch.sigmoid(1-stage_mask)
        # am = stage_mask2.squeeze().cpu().numpy()
        # am = cv2.resize(am, (256, 256))
        # # am[0:45,50:] = 0.02
        # am = 255 * (am - np.min(am)) / (
        #     np.max(am) - np.min(am) + 1e-12
        # )
        # am = np.uint8(np.floor(am))
        # am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
        # cv2.imwrite("128_i.png",am)
        out, stage_mask = self.block5(out, c, fusion_mask)

        out = F.interpolate(out, scale_factor=2)
        hh, ww = out.size(2), out.size(3)
        stage_mask = F.interpolate(stage_mask, size=(hh, ww), mode='bilinear', align_corners=True)
        fusion_mask = torch.sigmoid(stage_mask)
        stage_mask_256 = fusion_mask
        # am = torch.sigmoid(stage_mask).squeeze().cpu().numpy()
        # am = cv2.resize(am, (256, 256))
        # am = 255 * (am - np.min(am)) / (
        #     np.max(am) - np.min(am) + 1e-12
        # )
        # am = np.uint8(np.floor(am))
        # am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
        # cv2.imwrite("256_i.png",am)
        out, _ = self.block6(out, c, fusion_mask)

        out = self.conv_img(out)
        # flops, params = profile(self.conv_img, inputs=(out))
        # flops, params = clever_format([flops, params], "%.3f")
        # print("conv_img: ",flops, params )
        h_code = self.h_net(x, c_code)
        flops, params = profile(self.h_net, inputs=(x, c_code))
        flops, params = clever_format([flops, params], "%.3f")
        print("h_net: ", flops, params)
        # model = tv.models.inception_v3()
        # img_code64 = self.imgUpSample(region_features)
        h_code_img1 = self.acm_net(h_code, h_code)
        flops, params = profile(self.acm_net, inputs=(h_code, h_code))
        flops, params = clever_format([flops, params], "%.3f")
        print("acm_net: ", flops, params )

        fake_img = self.img_net(h_code_img1)
        flops, params = profile(self.img_net, inputs=(h_code_img1))
        flops, params = clever_format([flops, params], "%.3f")
        print("img_net: ", flops, params )
        fake_img = F.interpolate(fake_img, scale_factor=4)

        # out = ( 1- 0.022)* out + 0.022 * fake_img
        out = out + 0.012 * fake_img
        # tv.utils.save_image(fake_img,"tmp2.png")
        # img = out.squeeze().cpu().numpy()
        # img = 0.8 *  255 *np.swapaxes(img ,2,0)+ 0.2 * am
        # cv2.imwrite("tmp.png",img)
        # am = fusion_mask.squeeze().cpu().numpy()
        # # am
        # am = cv2.resize(am, (256, 256))
        # am =  (am - np.min(am)) / (
        #     np.max(am) - np.min(am) + 1e-12
        # )
        # am = np.uint8(np.floor(am))
        # am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
        # cv2.imwrite("256_d.png",am)

        
        # return out, fusion_mask
        return out, [stage_mask_4, stage_mask_8, stage_mask_16, stage_mask_32,
                     stage_mask_64, stage_mask_128, stage_mask_256]


class G_Block(nn.Module):

    def __init__(self, in_ch, out_ch, num_w=256, predict_mask=True):
        super(G_Block, self).__init__()

        self.learnable_sc = in_ch != out_ch
        self.predict_mask = predict_mask
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch)
        #self.affine1 = affine(in_ch)
        self.affine2 = affine(out_ch)
        #self.affine3 = affine(out_ch)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

        if self.predict_mask:
            self.conv_mask = nn.Sequential(nn.Conv2d(out_ch, 100, 3, 1, 1),
                                           BatchNorm(100),
                                           nn.ReLU(),
                                           nn.Conv2d(100, 1, 1, 1, 0))

    def forward(self, x, y=None, fusion_mask=None):
        out = self.shortcut(x) + self.gamma * self.residual(x, y, fusion_mask)

        if self.predict_mask:
            mask = self.conv_mask(out)
        else:
            mask = None

        return out, mask

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, y=None, fusion_mask=None):
        h = self.affine0(x, y, fusion_mask)
        h = nn.ReLU(inplace=True)(h)
        h = self.c1(h)

        h = self.affine2(h, y, fusion_mask)
        h = nn.ReLU(inplace=True)(h)
        return self.c2(h)


class affine(nn.Module):

    def __init__(self, num_features):
        super(affine, self).__init__()

        self.batch_norm2d = BatchNorm(num_features, affine=False)

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(256, num_features)),
        ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.zeros_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None, fusion_mask=None):
        x = self.batch_norm2d(x)
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        weight = weight * fusion_mask + 1
        bias = bias * fusion_mask
        return weight * x + bias


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16 + 256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, y):
        y = y.view(-1, 256, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)  # 128
        self.block0 = resD(ndf * 1, ndf * 2)  # 64
        self.block1 = resD(ndf * 2, ndf * 4)  # 32
        self.block2 = resD(ndf * 4, ndf * 8)  # 16
        self.block3 = resD(ndf * 8, ndf * 16)  # 8
        self.block4 = resD(ndf * 16, ndf * 16)  # 4
        self.block5 = resD(ndf * 16, ndf * 16)  # 4

        self.COND_DNET = D_GET_LOGITS(ndf)

    def forward(self, x):

        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        return out


class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x) + self.gamma * self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=True, spectral_norm=False):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding, bias=bias)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv


def linear(in_feat, out_feat, bias=True, spectral_norm=False):
    lin = nn.Linear(in_feat, out_feat, bias=bias)
    if spectral_norm:
        return nn.utils.spectral_norm(lin)
    else:
        return lin
