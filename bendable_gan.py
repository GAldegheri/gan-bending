import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.io import read_image
import torchvision.transforms as T
import numpy as np
from huggingface_hub import hf_hub_download
from math import log2
from huggan.pytorch.huggan_mixin import HugGANModelHubMixin
from huggan.pytorch.lightweight_gan.lightweight_gan import is_power_of_two, default, \
    PreNorm, LinearAttention, FCANet, GlobalContext, upsample, exists, Discriminator, \
    EMA, set_requires_grad, AugWrapper, LightweightGAN
from einops import rearrange
from collections import OrderedDict


class BendedGenerator(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            bending_module=None,
            bending_idx=None,
            train_bending=False,
            latent_dim=256,
            fmap_max=512,
            fmap_inverse_coef=12,
            transparent=False,
            greyscale=False,
            attn_res_layers=[32],
            freq_chan_attn=False
    ):
        super().__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.bending_module = bending_module
        self.bending_idx = bending_idx
        self.train_bending = train_bending
        
        norm_class = nn.BatchNorm2d
        Blur = nn.Identity
        
        resolution = log2(image_size)
        assert is_power_of_two(image_size), 'image size must be a power of 2'

        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3
            
        fmap_max = default(fmap_max, latent_dim)

        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            nn.BatchNorm2d(latent_dim * 2),
            nn.GLU(dim=1)
        )

        num_layers = int(resolution) - 2
        features = list(map(lambda n: (n, 2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2)))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))
        features = [latent_dim, *features]

        in_out_features = list(zip(features[:-1], features[1:]))

        self.res_layers = range(2, num_layers + 2)
        self.layers = nn.ModuleList([])
        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))

        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = list(filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map))
        self.sle_map = dict(self.sle_map)

        self.num_layers_spatial_res = 1

        for (res, (chan_in, chan_out)) in zip(self.res_layers, in_out_features):
            image_width = 2 ** res

            attn = None
            if image_width in attn_res_layers:
                attn = PreNorm(chan_in, LinearAttention(chan_in))

            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]

                if freq_chan_attn:
                    sle = FCANet(
                        chan_in=chan_out,
                        chan_out=sle_chan_out,
                        width=2 ** (res + 1)
                    )
                else:
                    sle = GlobalContext(
                        chan_in=chan_out,
                        chan_out=sle_chan_out
                    )

            layer = nn.ModuleList([
                nn.Sequential(
                    upsample(),
                    Blur(),
                    nn.Conv2d(chan_in, chan_out * 2, 3, padding=1),
                    norm_class(chan_out * 2),
                    nn.GLU(dim=1)
                ),
                sle,
                attn
            ])
            
            self.layers.append(layer)

        # Check that the channels of the bending module match
        # those of the target layer
        if bending_idx and bending_module:
            bentchannels = self.layers[bending_idx][0][2].out_channels
            assert bending_module.in_channels == bending_module.out_channels == bentchannels
            
        self.out_conv = nn.Conv2d(features[-1], init_channel, 3, padding=1)

    def forward(self, x, bend=True, return_inout=False):
        x = rearrange(x, 'b c -> b c () ()')
        x = self.initial_conv(x)
        x = F.normalize(x, dim=1)
        
        bend_in = None
        bend_out = None
        
        residuals = dict()
        
        for (i, res, (up, sle, attn)) in zip(range(len(self.layers)), self.res_layers, self.layers):
            
            if exists(attn):
                x = attn(x) + x

            x = up[:3](x)
            # Just to get the tensor sizes
            #print('Layer:', i, 'Input shape:', x.shape, 
            #      'bending:', i == self.bending_idx)
            if bend and i == self.bending_idx and self.bending_module is not None:
                bend_in = x
                x = self.bending_module(x)
                bend_out = x
            
            x = up[3:](x)

            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]
        if return_inout:
            return self.out_conv(x), bend_in, bend_out
        else:
            return self.out_conv(x)
    
    @classmethod
    def from_pretrained(cls, 
                        model_id,
                        image_size=512,
                        bending_module=None,
                        bending_idx=None,
                        train_bending=False,
                        map_location="cpu"):
        '''overrides method from ModelHubMixin'''
        
        model_file = hf_hub_download(
            repo_id=str(model_id),
            filename='model.pt',
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            token=None,
            local_files_only=False
        )
        
        state_dict = torch.load(model_file, 
                                map_location=map_location)['GAN']
        genstatedict = OrderedDict()
        for k, v in state_dict.items():
            if k[:2]=='G.':
                newk = k.replace('G.', '')
                genstatedict[newk] = v
        
        model = cls(image_size=image_size,
                    bending_module=bending_module,
                    bending_idx=bending_idx,
                    train_bending=train_bending)
        model.load_state_dict(genstatedict, strict=False)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        if train_bending==True and bending_module is not None:
            for param in model.bending_module.parameters():
                param.requires_grad = True
        
        return model
    
    def to(self, device):
        new_self = super(BendedGenerator, self).to(device)
        if new_self.bending_module is not None:
            new_self.bending_module = new_self.bending_module.to(device)
        return new_self
    
if __name__=="__main__":
    bendedgen = BendedGenerator.from_pretrained("ceyda/butterfly_cropped_uniq1K_512")