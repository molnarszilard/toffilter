import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision.models.resnet import resnet101

def agg_node(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def smooth(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def predict(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid(),
    )

def upshuffle(in_planes, out_planes, upscale_factor):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes*upscale_factor**2, kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(upscale_factor),
        nn.ReLU()
    )

class DFILT(nn.Module):
    def __init__(self, pretrained=True, fixed_feature_weights=False):
        super(DFILT, self).__init__()

        resnet = resnet101(pretrained=pretrained)

        # Freeze those weights
        if fixed_feature_weights:
            for p in resnet.parameters():
                p.requires_grad = False
        self.layer01=nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2,2), padding=(3, 3), bias=False)
        self.layer02=nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer03=nn.ReLU(inplace=True)
        self.layer04=nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # self.layer0=nn.Sequential(self.layer01,self.layer02,self.layer03,self.layer04)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)        
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Aggregate layers
        self.agg1 = agg_node(256, 128)
        self.agg2 = agg_node(256, 128)
        self.agg3 = agg_node(256, 128)
        self.agg4 = agg_node(256, 128)
        
        # Upshuffle layers
        self.up1 = upshuffle(128,128,8)
        self.up2 = upshuffle(128,128,4)
        self.up3 = upshuffle(128,128,2)
        
        # Depth prediction
        self.predict1 = smooth(512, 128)
        self.predict2 = predict(128, 1)
        
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        _,_,H,W = x.size()
        
        # Bottom-up
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)
        
        # Top-down predict and refine
        a1,a2,a3,a4=self.agg1(p5),self.agg2(p4),self.agg3(p3),self.agg4(p2)
        d5, d4, d3, d2 = self.up1(a1), self.up2(a2), self.up3(a3), a4
        _,_,H,W = d2.size()
        vol = torch.cat( [ F.upsample(d, size=(H,W), mode='bilinear') for d in [d5,d4,d3,d2] ], dim=1 )
        
        pred1 = self.predict1(vol)
        pred2 = F.interpolate(self.predict2(pred1), size=(H*4,W*4), mode='bilinear')
        # pred2 = self.predict2(pred1)
        #return pred2
        # half_number = torch.tensor(0.5).to('cuda')
        # pred3 = x[:,1,:,:]+pred2-half_number
        # valid = x[0][1]!=0.0
        # pred4 = pred2.clone()
        # pred4[0][0] = pred2[0][0] * valid
        # pred2[pred2<3*torch.max(x)/10]=0
        # pred2[pred2>=7*torch.max(x)/10]=1
        return pred2