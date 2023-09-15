import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import operator
from diffsort import DiffSortNet
import ipdb

class ConcatenatedModules(nn.Module):
    def __init__(self, modules):
        super(ConcatenatedModules, self).__init__()
        self.mod_list = nn.ModuleList(modules)
        self.mods = nn.Sequential(*self.mod_list)
        self.in_channels = modules[0].in_channels
        self.out_channels = modules[-1].out_channels
        
    def to(self, device):
        new_self = super(ConcatenatedModules, self).to(device)
        for m in self.mod_list:
            m.to(device)
        new_self.mods = nn.Sequential(*self.mod_list)
        
        return new_self
        
    def forward(self, x):
        return self.mods(x)
    

class BendingDiffSort_XY(nn.Module):
    def __init__(self, n_channels, input_size, 
                 perm_rows=True, perm_cols=True, steepness=50):
        super(BendingDiffSort_XY, self).__init__()
        self.in_channels = self.out_channels = n_channels
        self.hid_channels = n_channels * 2
        self.input_size = input_size
        
        if not perm_rows and not perm_cols:
            raise ValueError('At least one of perm_rows and perm_cols needs to be true!')
        self.perm_rows = perm_rows
        self.perm_cols = perm_cols
        
        self.steepness = steepness
        
        self.feat_extractor = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hid_channels, 
                      1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hid_channels, self.hid_channels, 
                      1, padding='same'),
            nn.ReLU(inplace=True)
            )
        
        if self.perm_rows:
            self.rowwise = nn.Conv1d(self.hid_channels * self.input_size,
                                    1, 1)
            
            self.row_sorter = DiffSortNet('bitonic', 
                                        self.input_size, 
                                        steepness=self.steepness)
        
        
        if self.perm_cols:
            self.colwise = nn.Conv1d(self.hid_channels * self.input_size,
                                    1, 1)
            
            self.col_sorter = DiffSortNet('bitonic',
                                        self.input_size,
                                        steepness=self.steepness)
        
    def to(self, device):
        new_self = super(BendingDiffSort_XY, self).to(device) 
        if self.perm_rows:
            new_self.row_sorter = DiffSortNet('bitonic', 
                                            self.input_size, 
                                            steepness=self.steepness,
                                            device=device)
        if self.perm_cols:
            new_self.col_sorter = DiffSortNet('bitonic',
                                            self.input_size,
                                            steepness=self.steepness,
                                            device=device)
        
        return new_self 
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        out = self.feat_extractor(x)
        
        if self.perm_rows:
        
            out_rows = out.permute(0, 2, 1, 3).reshape(batch_size, 
                                                    self.input_size, 
                                                    -1).permute(0, 2, 1)
            
            row_scores = self.rowwise(out_rows).flatten(1)
            _, sort_row_mat = self.row_sorter(row_scores)
            
            # First sort rows
            x = x.view(-1, self.input_size, self.input_size)
            sort_row_mat = sort_row_mat.unsqueeze(1).repeat(
                1, self.in_channels, 1, 1).view(
                    -1, self.input_size, self.input_size
                )
            x = torch.bmm(sort_row_mat, x).view(batch_size, 
                                                self.in_channels,
                                                self.input_size,
                                                self.input_size)
            
        if self.perm_cols:
            
            out_cols = out.permute(0, 3, 1, 2).reshape(batch_size,
                                                    self.input_size, 
                                                    -1).permute(0, 2, 1)
        
            col_scores = self.colwise(out_cols).flatten(1)
            
            
            _, sort_col_mat = self.col_sorter(col_scores)
            
            x = x.permute(0, 1, 3, 2).view(-1, 
                                        self.input_size, 
                                        self.input_size)
            sort_col_mat = sort_col_mat.unsqueeze(1).repeat(
                1, self.in_channels, 1, 1).view(
                    -1, self.input_size, self.input_size
                )
            x = torch.bmm(sort_col_mat, x).view(batch_size, 
                                                self.in_channels,
                                                self.input_size,
                                                self.input_size)
        
        return x
class BendingDiffSort(nn.Module):
    def __init__(self, n_channels, input_size, steepness=50):
        super(BendingDiffSort, self).__init__()
        self.in_channels = self.out_channels = n_channels
        self.input_size = input_size
        self.steepness = steepness
        
        self.feat_extractor = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Trick from https://datascience.stackexchange.com/questions/40906/determining-size-of-fc-layer-after-conv-layer-in-pytorch
        num_feats_before_fcnn = functools.reduce(
            operator.mul,
            list(self.feat_extractor(
                torch.rand(self.in_channels, input_size, input_size)
            ).shape)
        )
        
        self.fc1 = nn.Linear(num_feats_before_fcnn, 
                             self.in_channels)
        self.output_sorter = DiffSortNet('bitonic', 
                                         self.in_channels, 
                                         steepness=self.steepness)
        
    def to(self, device):
        new_self = super(BendingDiffSort, self).to(device) 
        new_self.output_sorter = DiffSortNet('bitonic', 
                                         self.in_channels, 
                                         steepness=self.steepness,
                                         device=device)
        
        return new_self 
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        out = self.feat_extractor(x)
        out = out.view(batch_size, -1)
        out = self.fc1(out)
        _, sort_mat = self.output_sorter(out)
        
        sorted_x = torch.bmm(x.reshape(batch_size, self.in_channels, -1).permute(0, 2, 1), 
                             sort_mat).permute(0, 2, 1).reshape(batch_size, 
                                               self.in_channels,
                                               self.input_size,
                                               self.input_size)
        
        return sorted_x

class BendingConvModule(nn.Module):
    def __init__(self, n_channels, act_fn='relu'):
        super(BendingConvModule, self).__init__()
        self.in_channels = self.out_channels = n_channels
        self.hid_channels = n_channels
        self.w1 = nn.Conv2d(self.in_channels, 
                            self.hid_channels, 3, 
                            padding='same')
        self.w2 = nn.Conv2d(self.hid_channels,
                            self.out_channels, 3,
                            padding='same')
        
        if act_fn == 'relu':    
            self.act_fn = F.relu
        elif act_fn == 'sin':
            self.act_fn = torch.sin
    
    def forward(self, x):
        x = self.w1(x)
        x = self.act_fn(x)
        return self.w2(x)
    
class BendingConvModule_XY(nn.Module):
    def __init__(self, n_channels, input_size, 
                 act_fn='relu', use_rad=False,
                 device='cuda'):
        super(BendingConvModule_XY, self).__init__()
        self.device = device
        self.input_size = input_size
        self.in_channels = n_channels
        self.act_fn = act_fn
        self.use_rad = use_rad
        self.out_channels = self.hid_channels = n_channels
        extra_chans = 3 if use_rad else 2
        self.w1 = nn.Conv2d(self.in_channels + extra_chans, 
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
        
        return new_self  
    
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
        
        inp = torch.cat((x, y, inp), dim=1)
        if self.use_rad:
            inp = torch.cat((r, inp), dim=1)
        
        inp = self.w1(inp)
        if self.act_fn == 'relu':
            inp = F.relu(inp)
        elif self.act_fn == 'sin':
            inp = torch.sin(inp)
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
    
if __name__ == "__main__":
    numchans = [1024, 1024, 512, 256, 128, 64, 6]
    inputsizes = [8, 16, 32, 64, 128, 256, 512]

    bending_idx = 1

    bendingmod_clip = BendingConvModule(numchans[bending_idx],
                                        act_fn='sin')
    bendsorting_clip = BendingDiffSort_XY(numchans[bending_idx],
                                    inputsizes[bending_idx])
    combined_bendmodule = nn.Sequential(bendsorting_clip, bendingmod_clip)
    
    x = torch.randn(1, 1024, 16, 16)
    
    y = combined_bendmodule(x)