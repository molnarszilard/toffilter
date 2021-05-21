import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelWiseOutlierLoss(nn.Module):
    def __init__(self):
        super(PixelWiseOutlierLoss, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        fake2=fake.clone()
        real2=real.clone()
        validloss = fake2-real2
        validloss[validloss<0]=0
        pwolloss=torch.log(torch.sum(validloss))
        return pwolloss