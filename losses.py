import torch
import torch.nn.functional as F
import numpy as np

def compute_diversity_loss(output_batch, noise_batch):
    
    assert output_batch.shape[0] == noise_batch.shape[0] and \
        output_batch.shape[0] % 2 == 0
    
    batch_size = output_batch.shape[0]
    
    outputs1 = output_batch[np.arange(0, batch_size, 2), ...].reshape(
            batch_size//2, -1)
    outputs2 = output_batch[np.arange(1, batch_size, 2), ...].reshape(
            batch_size//2, -1)
    noises1 = noise_batch[np.arange(0, batch_size, 2), ...].reshape(
            batch_size//2, -1)
    noises2 = noise_batch[np.arange(1, batch_size, 2), ...].reshape(
            batch_size//2, -1)
    
    noisesdiff = F.normalize(torch.mean(torch.abs(noises2 - noises1), axis=1), dim=0)
    outputsdiff = F.normalize(torch.mean(torch.abs(outputs2 - outputs1), axis=1), dim=0)
    
    # mode seeking loss
    loss_lz = torch.mean(noisesdiff/outputsdiff)
    
    return loss_lz