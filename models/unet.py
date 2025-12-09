import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128]):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        channels = in_channels
        for f in features:
            self.encoders.append(self._conv_block(channels, f))
            channels = f
        
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        
        for f in reversed(features):
            self.decoders.append(
                nn.ConvTranspose2d(
                    features[-1] * 2 if f == features[-1] else f * 2,
                    f, kernel_size=2, stride=2
                )
            )
            self.decoders.append(self._conv_block(f * 2, f))
        
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skips = skips[::-1]
        
        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)
            skip = skips[idx // 2]
            if x.size()[2:] != skip.size()[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((skip, x), dim=1)
            x = self.decoders[idx + 1](x)
        
        return torch.sigmoid(self.final(x))
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
