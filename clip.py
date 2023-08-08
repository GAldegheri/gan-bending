import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import open_clip
import numpy as np

class TextPrompt(nn.Module):
    def __init__(self, prompt_text, device='cpu'):
        super(TextPrompt, self).__init__()
        
        self.clip_model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32-quickgelu', pretrained='laion400m_e32'
        )
        self.clip_model.to(device)
        
        # Create transforms to feed images to CLIP:
        self.clip_tfms = T.Compose(preprocess.transforms[:2]+preprocess.transforms[-1:])
        
        self.prompt_text = prompt_text
        with torch.no_grad():
            tokenized_text = open_clip.tokenize([prompt_text]).to(device)
            self.prompt_embed = self.clip_model.encode_text(tokenized_text)
            
        # As a bonus, we can do some augmentation
        self.aug_tfms = T.Compose([
            T.RandomResizedCrop(480),
            T.RandomAffine(5),
            T.ColorJitter(),
            T.GaussianBlur(5)
        ])
        
    def forward(self, x, augment=True, return_mean=True,
                diversity=False):
        """
        Take a batch of images (x), encode them with clip_model
        and score each with the prompt using Squared Great Circle Distance
        (Lower is better).
        """
        if augment:
            x = self.aug_tfms(x)
        image_embeds = self.clip_model.encode_image(self.clip_tfms(x))
        input_normed = F.normalize(image_embeds.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.prompt_embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        
        if diversity:
            batch_size = x.shape[0]
            assert batch_size % 2 == 0
            img_embeds1 = input_normed[np.arange(0, batch_size, 2), ...]
            img_embeds2 = input_normed[np.arange(1, batch_size, 2), ...]
            x1 = x[np.arange(0, batch_size, 2), 
                   ...].reshape(batch_size//2, -1)
            x2 = x[np.arange(1, batch_size, 2), 
                    ...].reshape(batch_size//2, -1)
            div_latents = img_embeds1.sub(img_embeds2).norm(dim=2).div(2).arcsin().pow(2).mul(2)
            div_inputs = F.normalize(torch.mean(torch.abs(x1 - x2), axis=1), dim=0).reshape(-1, 1)
            
            diversities = div_latents#/div_inputs
            if return_mean:
                return dists.mean(), diversities.mean()
            return dists, diversities
        
        if return_mean:
            return dists.mean()
        return dists
        