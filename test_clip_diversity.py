import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from bendable_gan import BendedGenerator
from bending_modules import BendingConvModule, BendingConvModule_XY, BendingCPPN
from losses import compute_diversity_loss
from utils import generate_image, image_grid
from clip import TextPrompt

device = 'cuda'

if __name__=="__main__":
    # Create new bending module to optimize
    # with CLIP loss

    numchans = [1024, 1024, 512, 256, 128, 64, 6]

    bending_idx = 5

    bendingmod_clip = BendingConvModule(numchans[bending_idx],
                                        act_fn='relu')

    bend_generator_clip = BendedGenerator.from_pretrained("ceyda/butterfly_cropped_uniq1K_512",
                                                    bending_module=bendingmod_clip,
                                                    bending_idx=bending_idx,
                                                    train_bending=True)
    bend_generator_clip = bend_generator_clip.to(device)

    tgt_text = 'Neon Genesis Evangelion particle system'
    text_prompt = TextPrompt(tgt_text, device=device)
    
    torch.cuda.empty_cache()

    batch_size = 32

    n_iter = 1000

    div_loss = False
    div_weight = 16.
    div_loss_clip = True
    div_clip_weight = 6.

    opt = Adam(bendingmod_clip.parameters(), 1e-3)

    loss_log = []

    for i in tqdm(range(n_iter)):
        
        noise_input = torch.randn(batch_size, 
                        bend_generator_clip.latent_dim, 
                        device=device)
        
        out, b_in, _ = bend_generator_clip(noise_input, return_inout=True)
        out = out.clamp_(0., 1.)
            
        if div_loss_clip:
            loss, clip_div = text_prompt(out, diversity=True)
        else:
            loss = text_prompt(out)
        if div_loss:
            loss += div_weight * compute_diversity_loss(out, b_in)
        
        loss_log.append(loss.detach().cpu().numpy())

        with torch.no_grad():
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        
    plt.plot(range(n_iter), loss_log)   