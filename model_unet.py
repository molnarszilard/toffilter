import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision.models.resnet import resnet101
class DFILTUNET(nn.Module):
    def __init__(self, pretrained=True, fixed_feature_weights=False):
        super(DFILTUNET, self).__init__()

        
        ngf = 16
        self.layer01=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2,2), padding=(3, 3), bias=False)
        self.layer02=nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer03=nn.ReLU(inplace=True)
        self.layer04=nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer0=nn.Sequential(self.layer01,self.layer02,self.layer03,self.layer04)
        
        # self.cnv1_d_an = nn.Conv2d(1, ngf, [7, 7], stride=2)
        self.cnv1_d = nn.Conv2d(1, ngf // 2, [7, 7], stride=2, padding=(3, 3), bias=False)
        self.cnv1_c = nn.Conv2d(1, ngf // 2, [1, 1], stride=1)
        # self.cnv1_a = torch.cat([cnv1_d, cnv1_c], dim=1)
        self.cnv1_an = nn.Sequential(nn.Conv2d(1, ngf, [7, 7], stride=2, padding=(3, 3), bias=False),
                        nn.BatchNorm2d(ngf, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
        self.cnv2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, [5,5], stride=2, padding=2, bias=False),
                        nn.BatchNorm2d(ngf*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
        self.cnv3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, [3, 3], stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(ngf*4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
        self.cnv4 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, [3, 3], stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(ngf*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
        self.cnv5 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, [3, 3], stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(ngf*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
        self.cnv6 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, [3, 3], stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(ngf*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)) 
        self.cnv7 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, [3, 3], stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(ngf*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))  # /128

        self.upcnv7 = nn.ConvTranspose2d(ngf * 8, ngf * 8, [3, 3], stride=2)  # /64
        # self.upcnv7.resize_(cnv6.size())
        # self.i7_in = torch.cat([upcnv7, cnv6], dim=1)
        self.icnv7 = nn.Sequential(nn.Conv2d(ngf * 16, ngf * 8, [3, 3], stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(ngf*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))

        self.upcnv6 = nn.ConvTranspose2d(ngf * 8, ngf * 8, [3, 3], stride=2)  # /32
        # self.upcnv6.resize_(cnv5.size())
        # self.i6_in = torch.cat([upcnv6, cnv5], dim=1)
        self.icnv6 = nn.Sequential(nn.Conv2d(ngf * 16, ngf * 8, [3, 3], stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(ngf*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))

        self.upcnv5 = nn.ConvTranspose2d(ngf * 8, ngf * 8, [3, 3], stride=2)  # /16
        # self.upcnv5.resize_(cnv4.size())
        # self.i5_in = torch.cat([upcnv5, cnv4], dim=1)
        self.icnv5 = nn.Sequential(nn.Conv2d(ngf * 16, ngf * 8, [3, 3], stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(ngf*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))

        self.upcnv4 = nn.ConvTranspose2d(ngf * 8, ngf * 4, [3, 3], stride=2)  # /8
        # self.i4_in = torch.cat([upcnv4, cnv3], dim=1)
        self.icnv4 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 4, [3, 3], stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(ngf*4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
        self.out4 = nn.Sequential(nn.Conv2d(ngf * 4, 1, [3, 3], stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
        # self.out4_up = F.interpolate(out4, size=(H/4,W/4), mode='bilinear')
        # out4_up = tf.image.resize_bilinear(out4, [int(H / 4), int(W / 4)])

        self.upcnv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, [3, 3], stride=2)  # /4
        # self.i3_in = torch.cat([upcnv3, cnv2, out4_up], dim=1)
        self.icnv3 = nn.Sequential(nn.Conv2d(ngf * 4+1, ngf * 2, [3, 3], stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(ngf*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
        self.out3 = nn.Sequential(nn.Conv2d(ngf * 2, 1, [3, 3], stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
        # self.out3_up = F.interpolate(out4, size=(H/2,W/2), mode='bilinear')

        self.upcnv2 = nn.ConvTranspose2d(ngf * 2, ngf, [3, 3], stride=2)  # /2
        # self.i2_in = torch.cat([upcnv2, cnv1, out3_up], dim=1)
        self.icnv2 = nn.Sequential(nn.Conv2d(ngf * 2+1, ngf, [3, 3], stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(ngf, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
        self.out2 = nn.Sequential(nn.Conv2d(ngf, 1, [3, 3], stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
        # out2_up = F.interpolate(out4, size=(H,W), mode='bilinear')

        self.upcnv1 = nn.ConvTranspose2d(ngf, ngf // 2, [3, 3], stride=2)
        # self.i1_in = torch.cat([upcnv1], dim=1)  # [upcnv1, out2_up]
        self.icnv1 = nn.Sequential(nn.Conv2d(ngf//2, ngf, [3, 3], stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(ngf, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
        self.out1 = nn.Sequential(nn.Conv2d(ngf, 1, [3, 3], stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False))
        
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
        Conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        B,C,H,W = x.size()
        
        # # Bottom-up
        # c1 = self.layer0(x)
        # c2 = self.layer1(c1)
        # c3 = self.layer2(c2)
        # c4 = self.layer3(c3)
        # c5 = self.layer4(c4)

        # # Top-down
        # p5 = self.toplayer(c5)
        # p4 = self._upsample_add(p5, self.latlayer1(c4))
        # p4 = self.smooth1(p4)
        # p3 = self._upsample_add(p4, self.latlayer2(c3))
        # p3 = self.smooth2(p3)
        # p2 = self._upsample_add(p3, self.latlayer3(c2))
        # p2 = self.smooth3(p2)
        
        # # Top-down predict and refine
        # a1,a2,a3,a4=self.agg1(p5),self.agg2(p4),self.agg3(p3),self.agg4(p2)
        # d5, d4, d3, d2 = self.up1(a1), self.up2(a2), self.up3(a3), a4
        # _,_,H,W = d2.size()
        # vol = torch.cat( [ F.upsample(d, size=(H,W), mode='bilinear') for d in [d5,d4,d3,d2] ], dim=1 )
        
        # pred1 = self.predict1(vol)
        # pred2 = F.interpolate(self.predict2(pred1), size=(H*4,W*4), mode='bilinear')
        # # pred2 = self.predict2(pred1)
        # #return pred2
        # # half_number = torch.tensor(0.5).to('cuda')
        # # pred3 = x[:,1,:,:]+pred2-half_number
        # # valid = x[0][1]!=0.0
        # # pred4 = pred2.clone()
        # # pred4[0][0] = pred2[0][0] * valid
        # pred2[pred2<3*torch.max(x)/10]=0
        # pred2[pred2>=7*torch.max(x)/10]=1
        # return pred2


        self.aux = None
        depth_in = x[:,1,:,:].unsqueeze(dim=1)
        if C>1:
            other_in = x[:,2,:,:].unsqueeze(dim=1)

        if self.aux is None:
            conv1 = self.cnv1_an(depth_in)
        else:
            conv1_d = self.cnv1_d(depth_in)
            conv1_c = self.cnv1_c(other_in) # here should be color image or other channel of x
            conv1 = torch.cat([conv1_d, conv1_c], dim=1)
        # conv1 = self.bn1(conv1)
        # conv1 = self.relu(conv1)
        # conv1 = self.mp(conv1)
        conv = self.layer0(depth_in)
        conv2 = self.cnv2(conv1)
        conv3 = self.cnv3(conv2)
        conv4 = self.cnv4(conv3)
        conv5 = self.cnv5(conv4)
        conv6 = self.cnv6(conv5)
        conv7 = self.cnv7(conv6) # /128

        upconv7 = self.upcnv7(conv7)  # /64
        # upconv7.reshape(conv6.shape)
        upconv7 = F.interpolate(upconv7, size=(conv6.shape[2],conv6.shape[3]), mode='bilinear')
        i7_in = torch.cat([upconv7, conv6], dim=1)
        iconv7 = self.icnv7(i7_in)

        upconv6 = self.upcnv6(iconv7) # /32
        # upconv6.resize_(conv5.size())
        upconv6 = F.interpolate(upconv6, size=(conv5.shape[2],conv5.shape[3]), mode='bilinear')
        i6_in = torch.cat([upconv6, conv5], dim=1)
        iconv6 = self.icnv6(i6_in)

        upconv5 = self.upcnv5(iconv6) # /16
        # upconv5.resize_(conv4.size())
        upconv5 = F.interpolate(upconv5, size=(conv4.shape[2],conv4.shape[3]), mode='bilinear')
        i5_in = torch.cat([upconv5, conv4], dim=1)
        iconv5 = self.icnv5(i5_in)

        upconv4 = self.upcnv4(iconv5) # / 8
        upconv4 = F.interpolate(upconv4, size=(conv3.shape[2],conv3.shape[3]), mode='bilinear')
        i4_in = torch.cat([upconv4, conv3], dim=1)
        iconv4 = self.icnv4(i4_in)
        oout4 = self.out4(iconv4)
        out4_up = F.interpolate(oout4, size=(H//4,W//4), mode='bilinear')

        upconv3 = self.upcnv3(iconv4) # /4
        upconv3 = F.interpolate(upconv3, size=(conv2.shape[2],conv2.shape[3]), mode='bilinear')
        # conv2 = F.interpolate(conv2, size=(out4_up.shape[2],out4_up.shape[3]), mode='bilinear')
        i3_in = torch.cat([upconv3, conv2, out4_up], dim=1)
        iconv3 = self.icnv3(i3_in)
        oout3 = self.out3(iconv3)
        out3_up = F.interpolate(oout3, size=(H//2,W//2), mode='bilinear')

        upconv2 = self.upcnv2(iconv3) # /2
        upconv2 = F.interpolate(upconv2, size=(conv1.shape[2],conv1.shape[3]), mode='bilinear')
        # conv2 = F.interpolate(conv2, size=(out4_up.shape[2],out4_up.shape[3]), mode='bilinear')
        i2_in = torch.cat([upconv2, conv1, out3_up], dim=1)
        iconv2 = self.icnv2(i2_in)
        oout2 = self.out2(iconv2)
        out2_up = F.interpolate(oout2, size=(H,W), mode='bilinear')

        upconv1 = self.upcnv1(iconv2)
        upconv1 = F.interpolate(upconv1, size=(depth_in.shape[2],depth_in.shape[3]), mode='bilinear')
        i1_in = torch.cat([upconv1], dim=1)
        # i1_in = torch.cat([upconv1, out2_up], dim=1)
        iconv1 = self.icnv1(i1_in)
        iconv1 = F.interpolate(iconv1, size=(depth_in.shape[2],depth_in.shape[3]), mode='bilinear')
        out = self.out1(iconv1)
        out = F.interpolate(out, size=(depth_in.shape[2],depth_in.shape[3]), mode='bilinear')
        # end_pts = nn.utils.convert_collection_to_dict(end_pts_collection)
        # weight_vars = tf.get_collection(weight_collection)
        valid = depth_in[0][0]!=0.0
        out[0][0] = out[0][0] * valid
        # print(out.max(),out.min())
        out=out/out.max()
        # print(out.max(),out.min())
        return out#, end_pts, weight_vars