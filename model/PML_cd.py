import torch
import torch.nn as nn
import math
from einops import rearrange
from argparse import Namespace
from model.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.vit_seg_modeling import VisionTransformer
from model.vit_seg_modeling_L2HNet import L2HNet
from utils_pack.utils import make_coord
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out
class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class mini_model(nn.Module):
    def __init__(self, n_channel, scale_factor, in_channel, kernel_size, padding, groups):
        super(mini_model, self).__init__()
        self.n_channels = n_channel
        self.in_channel = in_channel


        self.conv1 = nn.Conv2d(in_channel, n_channel // 2, kernel_size, 1, padding, groups=groups)
        self.bn1 = nn.BatchNorm2d(n_channel // 2)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv2d(n_channel // 2, n_channel, 3, 1,
                               padding=2,
                               dilation=2,
                               groups=groups)
        self.bn2 = nn.BatchNorm2d(n_channel)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)


        self.eca = ECALayer(n_channel)


        self.ic_layer = IC_layer(n_channel, 0.3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu', a=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)


        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.eca(x)

        x = self.ic_layer(x)
        return x

class IC_layer(nn.Module):
    def __init__(self, n_channel, drop_rate):
        super(IC_layer, self).__init__()
        self.batch_norm = nn.BatchNorm2d(n_channel)
        self.drop_rate = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.drop_rate(x)
        return x


class SpatialMask(nn.Module):
    def __init__(self, mask_ratio=0.8, patch_size=4):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def forward(self, x):

        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"{H}x{W} cannot be divided by {self.patch_size}"


        x_patch = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                            p1=self.patch_size, p2=self.patch_size)


        num_patches = x_patch.size(1)
        num_keep = int(num_patches * (1 - self.mask_ratio))


        rand_indices = torch.rand(B, num_patches, device=x.device).argsort(dim=1)
        mask = torch.zeros(B, num_patches, device=x.device)
        mask[:, :num_keep] = 1
        mask = torch.gather(mask, 1, rand_indices.argsort(1))


        x_masked = x_patch * mask.unsqueeze(-1)

        x_masked = rearrange(x_masked, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                             h=H // self.patch_size, w=W // self.patch_size,
                             p1=self.patch_size, p2=self.patch_size)
        return x_masked, mask

class SpatialDecoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        self.recon_head = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 2, 3, padding=1)
        )
    def forward(self, x):
        return self.recon_head(x)
class UNO(nn.Module):
    def __init__(self, height=32, width=32, use_exf=False, scale_factor=4,
                 channels=128, sub_region=4, scaler_X=1, scaler_Y=1, args=None):
        super(UNO, self).__init__()
        self.height = height
        self.width = width
        self.masker = SpatialMask(mask_ratio=0.75, patch_size=4)
        self.decoder = SpatialDecoder(embed_dim=2)
        self.fg_height = height * scale_factor
        self.fg_width = width * scale_factor

        self.use_exf = use_exf
        self.n_channels = channels
        self.scale_factor = scale_factor
        self.out_channel = 2
        self.sub_region = scale_factor
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.args = args

        self.ic_layer = IC_layer(64, 0.3)
        self.config = CONFIGS_ViT_seg["ViT-B_16"]
        self.config.patches.grid = (int(16 / 16), int(16 / 16))
        self.con_v = VisionTransformer(self.config, backbone=L2HNet(width=96, image_band=3, length=5,
                                                                    ratios=[1, 0.5, 0.25],
                                                                    bn_momentum=0.1), img_size=16,
                                       num_classes=64).cuda()
        self.con_v.load_from(weights=np.load('../pycharm_project_916/model/pre-train_model/imagenet21k/ViT-B_16.npz'))
        time_span = 15

        if use_exf:
            self.time_emb_region = nn.Embedding(time_span,
                                                self.sub_region ** 2)
            self.time_emb_global = nn.Embedding(time_span, (self.fg_width * self.fg_height))

            self.embed_day = nn.Embedding(8, 2)
            self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23]
            self.embed_weather = nn.Embedding(15, 3)  # ignore 0, thus use 18

            self.ext2lr = nn.Sequential(
                nn.Linear(10, 64),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.sub_region ** 2),
                nn.ReLU(inplace=True)
            )

            self.ext2lr_global = nn.Sequential(
                nn.Linear(10, 64),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(64, int(self.fg_width * self.fg_height)),
                nn.ReLU(inplace=True)
            )

            self.global_model = mini_model(self.n_channels, self.scale_factor, 67, 9, 4, 1)  # 6 8
            self.local_sub_model = mini_model(self.n_channels * (int(self.fg_height / self.sub_region) ** 2),
                                              self.scale_factor, 66 * (int(self.fg_height / self.sub_region) ** 2), 3,
                                              1, int(self.fg_height / self.sub_region) ** 2)  # 6144 8192

        else:
            self.global_model = mini_model(self.n_channels, self.scale_factor, 64, 9, 4, 1)

            self.local_sub_model = mini_model(self.n_channels * (sub_region ** 2),
                                              self.scale_factor, 1024, 3, 1, sub_region ** 2)
        self.relu = nn.ReLU()
        time_conv = []
        for i in range(time_span):
            time_conv.append(nn.Conv2d(256, self.out_channel, 3, 1, 1))
        self.time_conv = nn.Sequential(*time_conv)

        self.time_my = nn.Conv2d(256, 2, 3, 1, 1)

    def embed_ext(self, ext):
        ext_out1 = self.embed_day(ext[:, 0].long().view(-1, 1)).view(-1, 2)
        ext_out2 = self.embed_hour(
            ext[:, 1].long().view(-1, 1)).view(-1, 3)
        ext_out3 = self.embed_weather(
            ext[:, 4].long().view(-1, 1)).view(-1, 3)
        ext_out4 = ext[:, 2:4]

        return torch.cat([ext_out1, ext_out2, ext_out3, ext_out4], dim=1)
    def expand_channels(self,x):

        return torch.cat([
            x,
            x[:, [0]]
        ], dim=1)
    def normalization(self, x, save_x):
        w = (nn.AvgPool2d(self.scale_factor)(x)) * self.scale_factor ** 2
        w = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')(w)
        w = torch.divide(x, w + 1e-7)
        up_c = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')(save_x)
        x = torch.multiply(w, up_c)
        return x


    def forward(self, x, eif,road_map, is_pretrain=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.squeeze(1)
        x_yuan = x
        if is_pretrain:
            x_masked, mask = self.masker(x)
            x = x_masked
        x = self.expand_channels(x)
        x = x.to(device)
        y, x = self.con_v(x)

        x = y+x
        save_x = x

        coor_hr = make_coord([self.height * self.scale_factor, self.width * self.scale_factor],
                             flatten=False).cuda().unsqueeze(0).expand(x.shape[0], self.height * self.scale_factor,
                                                                       self.width * self.scale_factor,
                                                                       2)


        if self.use_exf:

            x = self.relu(nn.functional.grid_sample(x, coor_hr.flip(-1), mode='bilinear', align_corners=False))  # [N, C, hr, hr]


            x = self.ic_layer(x)

            global_x = x

            x = rearrange(x, 'b c (ph h) (pw w) -> (ph pw) b c h w', ph=int(self.fg_height / self.sub_region),
                          pw=int(self.fg_width / self.sub_region))


            ext_emb = self.embed_ext(eif)
            t = eif[:, 4].long().view(-1, 1)
            if self.args.dataset == 'TaxiBJ':
                t -= 7
            time_emb_region = self.time_emb_region(t).view(-1, 1,
                                                           int(self.sub_region),
                                                           int(self.sub_region))

            time_emb_global = self.time_emb_global(t).view(-1, 1,
                                                           self.fg_height, self.fg_width)

            ext_out = self.ext2lr(ext_emb).view(-1, 1, int(self.sub_region),
                                                int(self.sub_region))

            ext_out_global = self.ext2lr_global(ext_emb).view(-1, 1, self.fg_width, self.fg_height)

            output_x = list(map(lambda x: torch.cat([x, ext_out, time_emb_region], dim=1).unsqueeze(0), x))
            output_x = torch.cat(output_x, dim=0)


            local_c = rearrange(output_x, '(ph pw) b c h w -> b (ph pw c) h w',
                                ph=int(self.fg_height / self.sub_region), pw=int(self.fg_width / self.sub_region))


            output = self.local_sub_model(local_c)

            local_f = rearrange(output, 'b (ph pw c) h w -> b c (ph h) (pw w)',
                                ph=int(self.fg_height / self.sub_region), pw=int(self.fg_width / self.sub_region))

            expanded_map = road_map.repeat(global_x.shape[0], 1, 1, 1)

            global_f = self.global_model(torch.cat([global_x, ext_out_global, time_emb_global,expanded_map], dim=1))


        else:

            local_c = rearrange(x, 'b c (ph h) (pw w) -> b (ph pw c) h w',
                                ph=self.sub_region, pw=self.sub_region)
            output = self.local_sub_model(local_c)
            local_f = rearrange(output, 'b (ph pw c) h w -> b c (ph h) (pw w)',
                                ph=self.sub_region, pw=self.sub_region)
            global_f = self.global_model(save_x)

        x = torch.cat([local_f, global_f], dim=1)

        output = []
        if self.use_exf:
            for i in range(x.size(0)):
                t = int(eif[i, 4].cpu().detach().numpy())
                if self.args.dataset == 'TaxiBJ':
                    t -= 7
                output.append(self.relu(self.time_conv[t](x[i].unsqueeze(0))))
        else:
            for i in range(x.size(0)):
                output.append(self.relu(self.time_my(x[i].unsqueeze(0))))
        x = torch.cat(output, dim=0)


        x = self.normalization(x, x_yuan * self.scaler_X / self.scaler_Y)
        if is_pretrain :
           sr_output = self.decoder(x)
           return x,sr_output,mask
        else:
           return x
