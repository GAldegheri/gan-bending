import torch
import torch.nn as nn
import torch.nn.functional as F

class BendingConvModule(nn.Module):
    def __init__(self, n_channels):
        super(BendingConvModule, self).__init__()
        self.in_channels = self.out_channels = n_channels
        self.hid_channels = n_channels
        self.w1 = nn.Conv2d(self.in_channels, 
                            self.hid_channels, 3, 
                            padding='same')
        self.w2 = nn.Conv2d(self.hid_channels,
                            self.out_channels, 3,
                            padding='same')
    
    def forward(self, x):
        x = self.w1(x)
        x = F.relu(x)
        return self.w2(x)
    
class BendingConvModule_XY(nn.Module):
    def __init__(self, n_channels, input_size, device='cuda'):
        super(BendingConvModule_XY, self).__init__()
        self.device = device
        self.input_size = input_size
        self.in_channels = n_channels
        self.out_channels = self.hid_channels = n_channels
        self.w1 = nn.Conv2d(self.in_channels + 3, 
                            self.hid_channels, 3, 
                            padding='same')
        self.w2 = nn.Conv2d(self.hid_channels,
                            self.out_channels, 3,
                            padding='same')
        self.x, self.y = self._create_grid(input_size)
        self.r = self._create_radial(input_size)
        self.sinx = torch.sin(self.x)
        self.siny = torch.sin(self.y)
    
    def _create_grid(self, input_size):
        x, y = torch.meshgrid(torch.arange(input_size), 
                              torch.arange(input_size), 
                              indexing='xy')
        return x.float().to(self.device), y.float().to(self.device)
    
    def _create_radial(self, input_size):
        center = input_size // 2
        rx, ry = torch.meshgrid(torch.arange(input_size) - center, 
                                torch.arange(input_size) - center, 
                                indexing='xy')
        radial = torch.sqrt(rx ** 2 + ry ** 2)
        return radial.to(self.device)
    
    def to(self, device):
        new_self = super(BendingConvModule_XY, self).to(device) 
        new_self.x = new_self.x.to(device)
        new_self.y = new_self.y.to(device)
        new_self.r = new_self.r.to(device)
        new_self.sinx = new_self.sinx.to(device)
        new_self.siny = new_self.siny.to(device)
        
        return  new_self  
    
    def forward(self, inp):
        batch_size = inp.shape[0]
        in_size = inp.shape[2]
        assert in_size == self.input_size
        
        x = torch.tile(self.x[None, None, ...], # add batch and channel dims 
                       (batch_size, 1, 1, 1))
        y = torch.tile(self.y[None, None, ...],
                       (batch_size, 1, 1, 1))
        r = torch.tile(self.r[None, None, ...],
                       (batch_size, 1, 1, 1))
        sinx = torch.tile(self.sinx[None, None, ...],
                       (batch_size, 1, 1, 1))
        siny = torch.tile(self.siny[None, None, ...],
                       (batch_size, 1, 1, 1))
        
        inp = torch.cat((x, y, r, inp), dim=1)
        
        inp = self.w1(inp)
        inp = F.relu(inp)
        return self.w2(inp)

class BendingCPPN(nn.Module):
    def __init__(self, n_channels, input_size, device='cuda'):
        super(BendingCPPN, self).__init__()
        
        self.input_size = input_size
        self.device = device
        
        self.in_channels = self.out_channels = n_channels
        self.hid_channels = n_channels * 2
        
        self.w_in = nn.Conv2d(self.in_channels,
                              self.hid_channels, 1,
                              padding='same')
        self.w_xyr = nn.Conv2d(3,
                               self.hid_channels, 1,
                               padding='same')
        # self.w_x = nn.Conv2d(1, 
        #                     self.hid_channels, 1, 
        #                     padding='same')
        # self.w_y = nn.Conv2d(1,
        #                     self.hid_channels, 1,
        #                     padding='same')
        # self.w_r = nn.Conv2d(1,
        #                      self.hid_channels, 1,
        #                      padding='same')
        
        self.w2 = nn.Conv2d(self.hid_channels,
                            self.out_channels, 1,
                            padding='same')
        
        self.x, self.y = self._create_grid(input_size)
        self.r = self._create_radial(input_size)
        
    def _create_grid(self, input_size):
        x, y = torch.meshgrid(torch.arange(input_size), 
                              torch.arange(input_size), 
                              indexing='xy')
        return x.float().to(self.device), y.float().to(self.device)
    
    def _create_radial(self, input_size):
        center = input_size // 2
        rx, ry = torch.meshgrid(torch.arange(input_size) - center, 
                                torch.arange(input_size) - center, 
                                indexing='xy')
        radial = torch.sqrt(rx ** 2 + ry ** 2)
        return radial.to(self.device)
        
    def forward(self, inp, ablate_input=False):
        batch_size = inp.shape[0]
        in_size = inp.shape[2]
        assert in_size == self.input_size
        
        x = torch.tile(self.x[None, None, ...], # add batch and channel dims 
                       (batch_size, 1, 1, 1))
        y = torch.tile(self.y[None, None, ...],
                       (batch_size, 1, 1, 1))
        r = torch.tile(self.r[None, None, ...],
                       (batch_size, 1, 1, 1))
        xyr = torch.cat([x, y, r], dim=1)
        
        
        # u = float(ablate_input) * 50. * self.w_in(inp) + \
        #     self.w_x(x) + self.w_y(y) + self.w_r(r)
        u = float(ablate_input) * self.w_in(inp) + \
            self.w_xyr(xyr)
            
        out = self.w2(torch.tanh(u))
        
        return out  
    
    def to(self, device):
        new_self = super(BendingCPPN, self).to(device) 
        new_self.x = new_self.x.to(device)
        new_self.y = new_self.y.to(device)
        new_self.r = new_self.r.to(device)
        
        return  new_self  