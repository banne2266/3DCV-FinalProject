import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .CDC import CDC_Block
from .LGFI import LGFI_Block
from .DepthDecoder import depth_decoder


class Conv_BA(nn.Module):
    def __init__(self, nIn, nOut, kernel, stride, padding=0, BA=False) -> None:
        super().__init__()
        self.BA = BA
        self.conv = nn.Conv2d(nIn, nOut, kernel, stride, padding=padding)

        if BA:
            self.bn = nn.BatchNorm2d(nOut, eps = 1e-5)
            self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.conv(x)
        if self.BA:
            x = self.bn(x)
            x = self.gelu(x)
        return x


class LiteMono(nn.Module):
    def __init__(self, in_chans=3, h=192, w=640, drop_path_rate=0.2, expan_ratio=6,
            heads=[8,8,8], use_pos_embed_attn = [True, False, False]) -> None:
        super().__init__()

        self.num_ch_enc = np.array([48, 80, 128])
        self.depth = [4, 4, 10]
        self.dim = [48, 80, 128]
        if h == 192 and w == 640:
            self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
        elif h == 320 and w == 1024:
            self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]

        

        self.stem1 = nn.Sequential(
            Conv_BA(in_chans, self.dim[0], kernel=3, stride=2, padding=1, BA=True),
            Conv_BA(self.dim[0], self.dim[0], kernel=3, stride=1, padding=1, BA=True),
            Conv_BA(self.dim[0], self.dim[0], kernel=3, stride=1, padding=1, BA=True)
        )
        self.stem2 = nn.Sequential(
            Conv_BA(self.dim[0]+3, self.dim[0], kernel=3, stride=2, padding=1, BA=False)
        )

        self.down_sample_layers = nn.ModuleList()
        self.down_sample_layers.append(self.stem1)
        self.down_sample_layers.append(self.stem2)
        for i in range(2):
            temp = nn.Sequential(
                Conv_BA(self.dim[i]*2+3, self.dim[i+1], kernel=3, stride=2, padding=1, BA=False)
            )
            self.down_sample_layers.append(temp)
        

        self.down_sample_avgpool = nn.ModuleList()
        for i in range(4):
            self.down_sample_avgpool.append(nn.AvgPool2d(3, stride=2, padding=1))
        

        #TODO dp_rates
        self.stage = nn.ModuleList()
        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                if j < self.depth[i] - 1:
                    stage_blocks.append(CDC_Block(dim=self.dim[i], k=3, dilation=self.dilation[i][j]))
                else:
                    stage_blocks.append(LGFI_Block(dim=self.dim[i], drop_path=drop_path_rate, expan_rate=expan_ratio, use_pos_embed=use_pos_embed_attn,num_head=heads[i]))
            stage_blocks = nn.Sequential(*stage_blocks)
            self.stage.append(stage_blocks)

        
        self.depth_decoder = depth_decoder(num_ch_enc=self.dim)

    
    def forward(self, x):
        final_features = []
        x_avg_pool_down = [self.down_sample_avgpool[0](x), 0, 0, 0]#avgpool down sample
        for i in range(1, 4):
            x_avg_pool_down[i] = self.down_sample_avgpool[i](x_avg_pool_down[i-1])
        

        stage1 = self.down_sample_layers[0](x)#stage-1

        #stage-2
        stageN_pre = torch.cat([stage1, x_avg_pool_down[0]], dim=1) #concat
        stageN_down = self.down_sample_layers[1](stageN_pre) #downsample
        stageN_final = self.stage[0](stageN_down) #CDC+LGFI
        final_features.append(stageN_final)

        #stage-3~4
        for cur_stage in range(2, 4):
            stageN_pre = torch.cat([stageN_final, x_avg_pool_down[cur_stage-1], stageN_down], dim=1) #concat
            stageN_down = self.down_sample_layers[cur_stage](stageN_pre) #downsample
            stageN_final = self.stage[cur_stage-1](stageN_down) #CDC+LGFI
            final_features.append(stageN_final)
        
        output = self.depth_decoder(final_features)#output: [1/4 depth map, 1/2 depth map, full res depth map]
        return output