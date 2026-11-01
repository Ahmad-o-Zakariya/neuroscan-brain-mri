import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self,a,b):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class AttentionGate(nn.Module):
    def __init__(self,channels):
        super().__init__(); self.g=nn.Conv2d(channels,channels,1); self.x=nn.Conv2d(channels,channels,1); self.psi=nn.Sequential(nn.ReLU(inplace=True),nn.Conv2d(channels,1,1),nn.Sigmoid())
    def forward(self,skip,gate): return skip*self.psi(self.g(gate)+self.x(skip))

class UNet(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,base_channels=16,attention=False):
        super().__init__(); c=base_channels; self.attention=attention
        self.e1=ConvBlock(in_channels,c); self.e2=ConvBlock(c,2*c); self.e3=ConvBlock(2*c,4*c); self.e4=ConvBlock(4*c,8*c); self.mid=ConvBlock(8*c,16*c); self.pool=nn.MaxPool2d(2)
        self.u4=nn.ConvTranspose2d(16*c,8*c,2,2); self.d4=ConvBlock(16*c,8*c); self.u3=nn.ConvTranspose2d(8*c,4*c,2,2); self.d3=ConvBlock(8*c,4*c); self.u2=nn.ConvTranspose2d(4*c,2*c,2,2); self.d2=ConvBlock(4*c,2*c); self.u1=nn.ConvTranspose2d(2*c,c,2,2); self.d1=ConvBlock(2*c,c); self.att=(AttentionGate(8*c),AttentionGate(4*c),AttentionGate(2*c),AttentionGate(c)) if attention else None; self.out=nn.Conv2d(c,out_channels,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b)); d=self.e4(self.pool(c)); z=self.mid(self.pool(d))
        skips=(d,c,b,a); us=(self.u4,self.u3,self.u2,self.u1); ds=(self.d4,self.d3,self.d2,self.d1)
        for i,(u,dec,skip) in enumerate(zip(us,ds,skips)):
            z=u(z); s=self.att[i](skip,z) if self.attention else skip; z=dec(torch.cat((z,s),1))
        return self.out(z)

def build_classifier(num_classes=4,pretrained=True):
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None); model.classifier[1]=nn.Linear(model.classifier[1].in_features,num_classes); return model

def build_segmenter(architecture="attention_unet",base_channels=16):
    if architecture not in {"unet","attention_unet"}: raise ValueError(f"Unsupported segmenter: {architecture}")
    return UNet(base_channels=base_channels,attention=architecture=="attention_unet")
