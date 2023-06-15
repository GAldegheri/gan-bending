import torch
from PIL import Image
import numpy as np

def generate_image(model, bend=True):
  model = model.to('cpu')
  with torch.no_grad():
        ims = model(torch.randn(1, model.latent_dim),
                    bend=bend)
        ims = ims.permute(0,2,3,1).clamp_(0., 1.)  * 255.
        ims = ims.detach().numpy().astype(np.uint8)
        # ims is [BxWxHxC] call Image.fromarray(ims[0])
  return Image.fromarray(ims[0])

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid