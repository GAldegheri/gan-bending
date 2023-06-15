import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import open_clip

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
        
    def forward(self, x, augment=True, return_mean=True):
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
        if return_mean:
            return dists.mean()
        return dists
        