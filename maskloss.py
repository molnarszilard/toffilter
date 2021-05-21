import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        fake2=fake.clone()
        real2=real.clone()
        eps_number = torch.tensor(1e-7).to('cuda')
        # fake2[fake2<torch.max(fake2)*0.1]=0
        # real2[real2<torch.max(real2)*0.1]=0
        # validfake = fake<torch.max(fake)*0.1
        # validreal = real<torch.max(real)*0.1
        # validloss = torch.logical_xor(fake2,real2)
        real2[real2>0] = 1
        real2[real2==0] = eps_number
        fake2[fake2==0] = eps_number
        RMSE_log = torch.sqrt(torch.mean(torch.abs(torch.log(torch.abs(real2))-torch.log(torch.abs(fake2)))**2))
        loss = RMSE_log*10
        return loss