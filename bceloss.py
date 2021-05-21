import torch
import torch.nn as nn
import torch.nn.functional as F

class CEntropyLoss(nn.Module):
    def __init__(self):
        super(CEntropyLoss, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        fake2=fake.clone()
        real2=real.clone()
        eps_number = torch.tensor(1e-7).to('cuda')
        real2[real2>0] = 1
        celoss=-torch.mean(real2*torch.log(fake2)+(1-real2)*torch.log(1-fake2))
        
        return celoss