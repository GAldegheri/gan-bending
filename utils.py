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
  

def permute_tensor_batchwise(A, order_tensor, dim):
    """
    Batch-wise permute tensor A along the given dimension using order_tensor.
    
    Args:
    - A (torch.Tensor): The tensor to permute.
    - order_tensor (torch.Tensor): The tensor specifying the order of permutation for each batch.
    - dim (int): The dimension along which to permute.
    
    Returns:
    - torch.Tensor: The permuted tensor.
    """
    
    # Check if the first dimension of A matches the first dimension of order_tensor
    assert A.shape[0] == order_tensor.shape[0], "Mismatch in batch size between A and order_tensor"
    
    # Check if the shape of A along the given dim matches the shape of order_tensor
    assert A.shape[dim] == order_tensor.shape[1], "Mismatch in shapes of A and order_tensor along the specified dimension"
    
    # Create a tensor of indices that matches the shape of A
    indices = [torch.arange(s).view(*([-1 if i == d else 1 for i in range(A.dim())])) for d, s in enumerate(A.shape)]
    
    # Reshape the order_tensor to fit A's shape for indexing
    for i in range(1, A.dim()):
        if i == dim:
            continue
        order_tensor = order_tensor.unsqueeze(i)
    
    # Replace the indices for the specified dim with order_tensor
    indices[dim] = order_tensor.expand_as(A)
    
    # Use advanced indexing to permute the tensor
    return A[tuple(indices)]
