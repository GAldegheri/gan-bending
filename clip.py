import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import open_clip
import numpy as np
from infonce import InfoNCE

class CLIP(torch.nn.Module):
    """
    Generic CLIP parent class
    """
    def __init__(self, prompt_text, device='cpu'):
        super(CLIP, self).__init__()
        
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
        
    def get_embeddings(self, x, augment=True, normalize=True):
        if augment:
            x = self.aug_tfms(x)
        image_embeds = self.clip_model.encode_image(self.clip_tfms(x))
        if normalize:
            image_embeds = F.normalize(image_embeds, dim=1)
            prompt_embed = F.normalize(self.prompt_embed, dim=1)
        
        return image_embeds, prompt_embed
            

class NCELoss(CLIP):
    def __init__(self, prompt_text, temperature=0.1, device='cpu'):
        super(NCELoss, self).__init__(prompt_text, device=device)
        
        self.InfoNCE = InfoNCE(temperature=temperature,
                               negative_mode='paired')
    
    def filter_crossval(self, x, batch_size):
        """
        Simple utility to remove first element from the first
        (repeated) batch, second from second batch etc.
        """
        
        total_size = x.shape[0]
        removerows = torch.arange(batch_size) * (batch_size + 1)
        keeprows = torch.LongTensor([i for i in \
            torch.arange(total_size) if i not in removerows])
        
        return x[keeprows, :]
    
    def forward(self, x, augment=True, normalize=True):
        
        image_embeds, prompt_embed = self.get_embeddings(
            x, augment=augment, normalize=normalize)
        batch_size = x.shape[0]
        
        pos_key = prompt_embed.repeat(batch_size, 1)
        neg_keys = image_embeds.repeat(batch_size, 1)
        neg_keys = self.filter_crossval(neg_keys, batch_size).reshape(
            batch_size, batch_size-1, -1) # [N, M, D]
        
        return self.InfoNCE(image_embeds, pos_key, negative_keys=neg_keys)
        
        
class TextPrompt(CLIP):
    def __init__(self, prompt_text, device='cpu'):
        super(TextPrompt, self).__init__(prompt_text, device=device)
        
    def forward(self, x, augment=True, return_mean=True,
                normalize=True, diversity=False):
        """
        Take a batch of images (x), encode them with clip_model
        and score each with the prompt using Squared Great Circle Distance
        (Lower is better).
        """
        image_embeds, prompt_embed = self.get_embeddings(
            x, augment=augment, normalize=normalize)
        image_embeds = image_embeds.unsqueeze(1)
        prompt_embed = prompt_embed.unsqueeze(0)
            
        dists = image_embeds.sub(prompt_embed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        
        if diversity:
            batch_size = x.shape[0]
            assert batch_size % 2 == 0
            img_embeds1 = image_embeds[np.arange(0, batch_size, 2), ...]
            img_embeds2 = image_embeds[np.arange(1, batch_size, 2), ...]
            x1 = x[np.arange(0, batch_size, 2), 
                   ...].reshape(batch_size//2, -1)
            x2 = x[np.arange(1, batch_size, 2), 
                    ...].reshape(batch_size//2, -1)
            div_latents = img_embeds1.sub(img_embeds2).norm(dim=2).div(2).arcsin().pow(2).mul(2)
            #div_inputs = F.normalize(torch.mean(torch.abs(x1 - x2), axis=1), dim=0).reshape(-1, 1)
            
            diversities = 1./div_latents#/div_inputs
            if return_mean:
                return dists.mean(), diversities.mean()
            return dists, diversities
        
        if return_mean:
            return dists.mean()
        return dists
    
if __name__=="__main__":
    x = torch.randn(20, 3, 512, 512).cuda()
    model = NCELoss('A laughing pumpkin', device='cuda')
    y = model(x)
        