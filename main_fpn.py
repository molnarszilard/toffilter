# from dataset.dataloader import DepthDataset
from collections import Counter
from dataset.nyuv2_dataset import NYUv2Dataset
from model_fpn import DFILT
from model_unet import DFILTUNET
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from unet_model import UNet
from autoencode import Autoencoder
import argparse, time
import matplotlib, cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
matplotlib.use('Agg')

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

class DDDDepthDiff(nn.Module):
    def __init__(self):
        super(DDDDepthDiff, self).__init__()

    def point_cloud(self, depth1):
        """Transform a depth image into a point cloud with one point for each
        pixel in the image, using the camera transform for a camera
        centred at cx, cy with field of view fx, fy.

        depth is a 2-D ndarray with shape (rows, cols) containing
        depths from 1 to 254 inclusive. The result is a 3-D array with
        shape (rows, cols, 3). Pixels with invalid depth in the input have
        NaN for the z-coordinate in the result.

        """
        # depth is of shape (1,480,640)
        # K = [460.58518931365654, 0.0, 334.0805877590529, 0.0, 460.2679961517268, 169.80766383231037, 0.0, 0.0, 1.0] # pico zense
        # K = [460.585, 0.0, 334.081, 0.0, 460.268, 169.808, 0.0, 0.0, 1.0] # pico zense
        K = [582.62448167737955, 0.0, 313.04475870804731, 0.0, 582.69103270988637, 238.44389626620386, 0.0, 0.0, 1.0] # nyu_v2_dataset
        # K = [582.624, 0.0, 313.045, 0.0, 582.691, 238.444, 0.0, 0.0, 1.0] # nyu_v2_dataset
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]

        depth = depth1.clone()
        # open3d_img = o3d.t.geometry.Image(depth[0])#/1000.0)
        # intrinsics = o3d.camera.PinholeCameraIntrinsic(640,360,fx,fy,cx,cy)
        # pcd = o3d.geometry.create_point_cloud_from_depth_image(open3d_img,intrinsic=intrinsics)
        rows, cols = depth[0].shape
        # c, _ = torch.meshgrid(torch.arange(cols), torch.arange(cols))
        c = torch.meshgrid(torch.arange(cols))
        new_c = c[0].reshape([1,cols]).to('cuda')
        r = torch.meshgrid(torch.arange(rows))
        new_r = r[0].unsqueeze(-1).to('cuda')
        valid = (depth > 0) & (depth < 1)
        nan_number = torch.tensor(np.nan).to('cuda')
        eps_number = torch.tensor(1e-7).to('cuda')
        zero_number = torch.tensor(0.).to('cuda')
        z = torch.where(valid, depth/1000.0, nan_number)
        x = torch.where(valid, z * (new_c - cx) / fx, nan_number)
        y = torch.where(valid, z * (new_r - cy) / fy, nan_number)
        

        dimension = rows * cols
        z_ok = z.reshape(depth.shape[0],dimension)
        x_ok = x.reshape(depth.shape[0],dimension)
        y_ok = y.reshape(depth.shape[0],dimension)
    
        return torch.stack((x_ok,y_ok,z_ok),dim=1) 

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_ , H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        eps = 1e-7
        real1 = real.clone() #real[0].cpu().detach().numpy()
        fake1 = fake.clone() #fake[0].cpu().detach().numpy()
        
        # real_pcd = self.point_cloud(real1).clone()
        # fake_pcd = self.point_cloud(fake1).clone()

        all_real_pcd = self.point_cloud(real1[0]).clone() * 1000.0
        all_fake_pcd = self.point_cloud(fake1[0]).clone() * 1000.0
 
        for nr_img in range(1,args.bs):
            real_pcd = self.point_cloud(real1[nr_img]).clone() * 1000.0
            fake_pcd = self.point_cloud(fake1[nr_img]).clone() * 1000.0

            all_real_pcd = torch.cat(all_real_pcd,real_pcd)
            all_fake_pcd = torch.cat(all_fake_pcd,fake_pcd)

        all_real_pcd[all_real_pcd==0] = eps
        all_fake_pcd[all_fake_pcd==0] = eps

        # real_pcd[real_pcd==0] = eps
        # fake_pcd[fake_pcd==0] = eps

        #######################
        nan_z_real = all_real_pcd[:,2].clone()
        temp_z_real = nan_z_real[~torch.isnan(nan_z_real)]
       
        nan_z_fake = all_fake_pcd[:,2].clone()
        temp_z_fake = nan_z_fake[~torch.isnan(nan_z_real)]
        
        nan_x_real = all_real_pcd[:,0].clone()
        temp_x_real = nan_x_real[~torch.isnan(nan_x_real)]
        
        nan_x_fake = all_fake_pcd[:,0].clone()
        temp_x_fake = nan_x_fake[~torch.isnan(nan_x_real)]
        
        nan_y_real = all_real_pcd[:,1].clone()
        temp_y_real = nan_y_real[~torch.isnan(nan_y_real)]
        
        nan_y_fake = all_fake_pcd[:,1].clone()
        temp_y_fake = nan_y_fake[~torch.isnan(nan_y_real)]

        z_real = temp_z_real[~torch.isnan(temp_z_fake)]
        z_fake = temp_z_fake[~torch.isnan(temp_z_fake)]

        x_real = temp_x_real[~torch.isnan(temp_x_fake)]
        x_fake = temp_x_fake[~torch.isnan(temp_x_fake)]

        y_real = temp_y_real[~torch.isnan(temp_y_fake)]
        y_fake = temp_y_fake[~torch.isnan(temp_y_fake)]
        
        ####sixth try #####
        lossX = torch.sqrt(torch.mean(torch.abs(x_real-x_fake)**2))
        lossZ = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))
        lossY = torch.sqrt(torch.mean(torch.abs(y_real-y_fake)**2))

        # lossD = torch.mean(torch.abs(real1-fake1))
       
        # RMSE_log = torch.sqrt(torch.mean(torch.abs(torch.log(torch.abs(z_real))-torch.log(torch.abs(z_fake)))**2))
        RMSE_log = torch.sqrt(torch.mean(torch.abs(torch.log(torch.abs(z_real))-torch.log(torch.abs(z_fake)))**2))
        # delta = [RMSE_log, lossX, lossY, lossZ]
        # loss = 10 *RMSE_log* torch.abs(10*(3-torch.exp(1*lossX)-torch.exp(1*lossY)-torch.exp(1*lossZ)))
        loss = 10*(RMSE_log + torch.abs(10*(3-torch.exp(1*lossX)-torch.exp(1*lossY)-torch.exp(1*lossZ))))
        #v19 distance = 0.067836
        #v20-pico, distance = 0.054325
        #v26-pico distance = 0.038819
        # loss = 10 * torch.abs(10*(3-torch.exp(1*lossX)-torch.exp(1*lossY)-torch.exp(1*lossZ))) #v14 distance = 0.055079
        # v18 distance = 0.069045
        # loss = (lossX+lossY+lossZ)*100 #v15 distance = 0.105806
        # loss = lossD #v16 distance=0.078502
        # loss = RMSE_log*100 #v17 distance=0.123481
        return loss
# def get_acc(output, target):
#     # takes in two tensors to compute accuracy
#     pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#     correct = pred.eq(target.data.view_as(pred)).cpu().sum()
#     print("Target: ", Counter(target.data.cpu().numpy()))
#     print("Pred: ", Counter(pred.cpu().numpy().flatten().tolist()))
#     return float(correct)*100 / target.size(0) 


class PixelWiseOutlierLoss(nn.Module):
    def __init__(self):
        super(PixelWiseOutlierLoss, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        fake2=fake.clone()
        real2=real.clone()
        # fake2[fake2<torch.max(fake2)*0.1]=0
        # real2[real2<torch.max(real2)*0.1]=0
        # validfake = fake<torch.max(fake)*0.1
        # validreal = real<torch.max(real)*0.1
        # validloss = torch.logical_xor(fake2,real2)
        # real2[real2>0] = 1
        # fake2[fake2>0] = 1
        validloss = fake2-real2
        validloss[validloss<0]=0
        pwolloss=torch.log(torch.sum(validloss))
        return pwolloss

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

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Single image depth estimation')
    parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='nyuv2', type=str)
    parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=10, type=int)
    parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      default=True,
                      action='store_true')
    parser.add_argument('--bs', dest='bs',
                      help='batch_size',
                      default=1, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                      help='num_workers',
                      default=1, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                      help='display interval',
                      default=10, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                      help='output directory',
                      default='saved_models', type=str)

# config optimization
    parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="adam", type=str)
    parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=1e-3, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=3, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
    parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)
    parser.add_argument('--eval_epoch', dest='eval_epoch',
                      help='number of epoch to evaluate',
                      default=2, type=int)

# resume trained model
    parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
    parser.add_argument('--start_at', dest='start_epoch',
                      help='epoch to start with',
                      default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=8, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)

# training parameters
    parser.add_argument('--gamma_sup', dest='gamma_sup',
                      help='factor of supervised loss',
                      default=1., type=float)
    parser.add_argument('--gamma_unsup', dest='gamma_unsup',
                      help='factor of unsupervised loss',
                      default=1., type=float)
    parser.add_argument('--gamma_reg', dest='gamma_reg',
                      help='factor of regularization loss',
                      default=10., type=float)

    args = parser.parse_args()
    return args

def get_coords(b, h, w):
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(b,1,h,w))  # [B, 1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(b,1,h,w))  # [B, 1, H, W]
    coords = torch.cat((j_range, i_range), dim=1)
    norm = Variable(torch.Tensor([w,h]).view(1,2,1,1))
    coords = coords * 2. / norm - 1.
    coords = coords.permute(0, 2, 3, 1)
   
    return coords
        
def resize_tensor(img, coords):
    return nn.functional.grid_sample(img, coords, mode='bilinear', padding_mode='zeros')

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

#     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))
    
    return grad_y, grad_x

def imgrad_yx(img):
    N,C,_,_ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)

def reg_scalor(grad_yx):
    return torch.exp(-torch.abs(grad_yx)/255.)
    
    
class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

def collate_fn(data):
    imgs, depths = zip(*data)
    B = len(imgs)
    im_batch = torch.ones((B,3,376,1242))
    d_batch = torch.ones((B,1,376,1242))
    for ind in range(B):
        im, depth = imgs[ind], depths[ind]
        im_batch[ind, :, -im.shape[1]:, :im.shape[2]] = im
        d_batch[ind, :, -depth.shape[1]:, :depth.shape[2]] = depth
    return im_batch, d_batch

if __name__ == '__main__':

    args = parse_args()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You might want to run with --cuda")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    train_dataset = NYUv2Dataset()
    train_size = len(train_dataset)
    eval_dataset = NYUv2Dataset(train=False)
    eval_size = len(eval_dataset)
    print(train_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                            shuffle=True, num_workers=args.num_workers)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.bs,
                            shuffle=True, num_workers=args.num_workers)
        
    # network initialization
    print('Initializing model...')
    # dfilt = DFILT(fixed_feature_weights=False)
    # dfilt = DFILTUNET(fixed_feature_weights=False)
    dfilt = UNet(3,1)
    dfilt = Autoencoder()
    if args.cuda:
        dfilt = dfilt.cuda()
    
        
    print('Done!')

    # hyperparams
    lr = args.lr
    bs = args.bs
    lr_decay_step = args.lr_decay_step
    lr_decay_gamma = args.lr_decay_gamma

    # params
    params = []
    for key, value in dict(dfilt.named_parameters()).items():
      if value.requires_grad:
        if 'bias' in key:
            DOUBLE_BIAS=0
            WEIGHT_DECAY=4e-5
            params += [{'params':[value],'lr':lr*(DOUBLE_BIAS + 1), \
                  'weight_decay': 4e-5 and WEIGHT_DECAY or 0}]
        else:
            params += [{'params':[value],'lr':lr, 'weight_decay': 4e-5}]

    # optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    # rmse = RMSE()
    # depth_criterion = RMSE_log()
    # grad_criterion = GradLoss()
    # normal_criterion = NormalLoss()
    d_crit=DDDDepthDiff()
    # eval_metric = RMSE_log()
    pwol = PixelWiseOutlierLoss()
    maskloss = MaskLoss()
    criterion = nn.MSELoss()
    
    # resume
    if args.resume:
        load_name = os.path.join(args.output_dir,
          'dfilt_1_{}.pth'.format(args.checkepoch))
        print("loading checkpoint %s" % (load_name))
        state = dfilt.state_dict()
        checkpoint = torch.load(load_name)
        args.start_epoch = checkpoint['epoch']
        checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
        state.update(checkpoint)
        dfilt.load_state_dict(state)
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
        del checkpoint
        torch.cuda.empty_cache()

    # constants
    iters_per_epoch = int(train_size / args.bs)
    
    train_loss_arr = []
    val_loss_arr = []
    for epoch in range(args.start_epoch, args.max_epochs):
        train_loss = 0 
        eval_loss = 0
        # setting to train mode
        dfilt.train()
        start = time.time()
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        img = Variable(torch.FloatTensor(1))
        z = Variable(torch.FloatTensor(1))
        if args.cuda:
            img = img.cuda()
            z = z.cuda()
        
        train_data_iter = iter(train_dataloader)
        for step in range(iters_per_epoch):
            # print(min_value)
            # print(max_value)
            start = time.time()
            data = train_data_iter.next()
            
            img.resize_(data[0].size()).copy_(data[0])
            z.resize_(data[1].size()).copy_(data[1])
            optimizer.zero_grad()
            # height = img.size()[2]
            # width =img.size()[3]
            # subimg1=img[:,:,0:height*2/3,0:width*2/3]
            # subimg2=img[:,:,0:height*2/3,width/3:width]
            # subimg3=img[:,:,height/3:height,0:width*2/3]
            # subimg4=img[:,:,height/3:height,width/3:width]

            z_fake = dfilt(img)
            # zf1=dfilt(subimg1)
            # zf2=dfilt(subimg2)
            # zf3=dfilt(subimg3)
            # zf4=dfilt(subimg4)
            # zf=img.clone()
            # zf[:,:,0:height/3,0:width/3]=zf1[:,:,0:height/3,0:width/3]
            # zf[:,:,0:height/3,width*2/3:width]=zf2[:,:,0:height/3,width/3:width*2/3]
            # zf[:,:,height*2/3:height,0:width/3]=zf3[:,:,height/3:height*2/3,0:width/3]
            # zf[:,:,height*2/3:height,width*2/3:width]=zf4[:,:,height/3:height*2/3,width/3:width*2/3]
            # zf[:,:,he]

            dloss=d_crit(z_fake,z)
            loss = 1*dloss
            # loss = criterion(z_fake, z)

            # pwloss = pwol(z_fake,z)
            # loss = pwloss

            # loss=maskloss(z_fake,z)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            end = time.time()

            # info
            if step % args.disp_interval == 0:

                # print("[epoch %2d][iter %4d] loss: %.4f RMSElog: %.4f grad_loss: %.4f normal_loss: %.4f" \
                #                 % (epoch, step, loss, depth_loss, grad_loss, normal_loss))
                print("[epoch %2d][iter %4d] loss: %.4f " \
                                % (epoch, step, loss))
        # save model
        if epoch%1==0 or epoch==args.max_epochs-1:
            save_name = os.path.join(args.output_dir, 'dfilt_{}_{}.pth'.format(args.session, epoch))
            torch.save({'epoch': epoch+1,
                    'model': dfilt.state_dict(), 
#                     'optimizer': optimizer.state_dict(),
                   },
                   save_name)

            print('save model: {}'.format(save_name))
        print('time elapsed: %fs' % (end - start))
            
        with torch.no_grad():
            # setting to eval mode
            dfilt.eval()

            img = Variable(torch.FloatTensor(1), volatile=True)
            z = Variable(torch.FloatTensor(1), volatile=True)
            if args.cuda:
                img = img.cuda()
                z = z.cuda()

            print('evaluating...')

            eval_data_iter = iter(eval_dataloader)
            for i, data in enumerate(eval_data_iter):
                print(i,'/',len(eval_data_iter)-1)

                img.resize_(data[0].size()).copy_(data[0])
                z.resize_(data[1].size()).copy_(data[1])
                
                z_fake = dfilt(img)
                
                eloss=d_crit(z_fake,z)
                # eloss = pwol(z_fake,z)
                # eloss = maskloss(z_fake,z)
                # eloss = criterion(z_fake, z)
                eval_loss += eloss
                
                
            eval_loss = eval_loss/len(eval_dataloader)
            train_loss = train_loss/iters_per_epoch
            train_loss_arr.append(train_loss)
            val_loss_arr.append(eval_loss)
            print("[epoch %2d] loss: %.4f " \
                            % (epoch, torch.sqrt(eval_loss)))
            with open('val.txt', 'a') as f:
                f.write("[epoch %2d] loss: %.4f\n" \
                            % (epoch, torch.sqrt(eval_loss)))
    epochs = range(args.start_epoch, args.max_epochs)
    plt.plot(epochs, train_loss_arr, '-g', label='Training loss')
    plt.plot(epochs, val_loss_arr, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("losses.png")
    plt.close()
       

