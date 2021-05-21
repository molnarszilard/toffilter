# from dataset.dataloader import DepthDataset
from collections import Counter
from dataset.nyuv2_dataset import NYUv2Dataset
from model_fpn import DFILT
from model_unet import DFILTUNET
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from torch.autograd import Variable
from torchvision import transforms
from unet_model import UNet
from autoencoder import Autoencoder
from bceloss import CEntropyLoss
from depthdiffloss import DDDDepthDiff
from pixelwiseloss import PixelWiseOutlierLoss
from maskloss import MaskLoss
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
    parser.add_argument('--model', dest='model',
                      help='modeltype: dfilt, dfiltunet, unet, ae',
                      default="ae", type=str)

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
    parser.add_argument('--lt', dest='losst',
                      help='losstype: DDD, ownBCE, BCE, BCElogits, maskloss, pwloss',
                      default="mse", type=str)


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

if __name__ == '__main__':

    args = parse_args()

    if args.losst is 'DDD':
        criterion = DDDDepthDiff()
    if args.losst is 'ownBCE':
        criterion = CEntropyLoss()
    if args.losst is 'BCE':
        criterion = nn.BCELoss()
    if args.losst is 'BCElogits':
        criterion = nn.BCEWithLogitsLoss()
    if args.losst is 'maskloss':
        criterion = MaskLoss()
    if args.losst is 'pwloss':
        criterion = PixelWiseOutlierLoss()
    if args.losst is 'mse':
        criterion = nn.MSELoss()

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
    if args.model is 'dfilt':
        dfilt = DFILT(fixed_feature_weights=False)
    if args.model is 'dfiltunet':
        dfilt = DFILTUNET(fixed_feature_weights=False)
    if args.model is 'unet':
        dfilt = UNet(3,1)
    if args.model is 'ae':
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
            start = time.time()
            data = train_data_iter.next()
            
            img.resize_(data[0].size()).copy_(data[0])
            z.resize_(data[1].size()).copy_(data[1])
            optimizer.zero_grad()
          
            z_fake = dfilt(img)

            loss=criterion(z_fake,z)
            
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

                eloss = criterion(z_fake, z)
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
