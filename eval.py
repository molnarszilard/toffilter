from autoencoder import Autoencoder
from model_fpn import DFILT
from model_unet import DFILTUNET
from unet_model import UNet
from threading import Thread
from torch.autograd import Variable
from torchvision.utils import save_image
import argparse, time
import cv2
import numpy as np
import os, sys
import timeit
import torch, time
import imageio
import PIL
from torchvision import transforms

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Normal image estimation from ToF depth image')
    parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      default=True,
                      action='store_true')
    parser.add_argument('--num_workers', dest='num_workers',
                      help='num_workers',
                      default=1, type=int)  
    parser.add_argument('--input_image_path', dest='input_image_path',
                      help='path to a single input image for evaluation',
                      default='/media/rambo/ssd2/Szilard/nyu_v2_filter/comparison/depth3_n10/', type=str)
    parser.add_argument('--output_output_path', dest='output_image_path',
                      help='path to a single input image for evaluation',
                      default='/media/rambo/ssd2/Szilard/nyu_v2_filter/comparison/predictions/', type=str)
                    #   default='/media/rambo/ssd2/Szilard/pico_tofnest/4bag_unfiltered/depth3/', type=str)
    parser.add_argument('--eval_folder', dest='eval_folder',
                      help='evaluate only one image or the whole folder',
                      default=True, type=bool)
    parser.add_argument('--model_path', dest='model_path',
                      help='path to the model to use',
                      default='saved_models/dfilt_1_9_v40.pth', type=str)
    parser.add_argument('--model', dest='model',
                      help='modeltype: dfilt, dfiltunet, unet, ae',
                      default="dfilt", type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You might want to run with --cuda")
    
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
    
    
    load_name = os.path.join(args.model_path)
    print("loading checkpoint %s" % (load_name))
    state = dfilt.state_dict()
    checkpoint = torch.load(load_name)
    checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
    state.update(checkpoint)
    dfilt.load_state_dict(state)
    if 'pooling_mode' in checkpoint.keys():
        POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))
    del checkpoint
    torch.cuda.empty_cache()

    dfilt.train()

    img = Variable(torch.FloatTensor(1))

    print('evaluating...')
    with torch.no_grad():
        if args.eval_folder:
            dlist=os.listdir(args.input_image_path)
            dlist.sort()
            time_sum = 0
            counter = 0
            max_depth=10000.
            min_depth=300.
            nan_number = torch.tensor(np.nan).to('cuda')
            eps_number = torch.tensor(1e-7).to('cuda')
            zero_number = torch.tensor(0.).to('cuda')
            for filename in dlist:
                if filename.endswith(".png"):
                    inpath=args.input_image_path+filename
                    outpath=args.output_image_path+filename
                    print("Predicting for:"+filename)
                    depth = cv2.imread(inpath,cv2.IMREAD_UNCHANGED).astype(np.float32)
                    if len(depth.shape) < 3:
                        print("Got 1 channel depth images, creating 3 channel depth images")
                        combine_depth = np.empty((depth.shape[0],depth.shape[1], 3))
                        combine_depth[:,:,0] = depth
                        combine_depth[:,:,1] = depth
                        combine_depth[:,:,2] = depth
                        depth = combine_depth
                    depth2 = np.moveaxis(depth,-1,0)
                    img = torch.from_numpy(depth2).float().unsqueeze(0).cuda()
                    start = timeit.default_timer()
                    m_depth=torch.max(img)
                    img=img/m_depth                 
                    z_fake=dfilt(img)
                    stop = timeit.default_timer()
                    time_sum=time_sum+stop-start
                    counter=counter+1
                    # npimage=(z_fake[0]*255).squeeze(0).cpu().detach().numpy().astype(np.uint8)
                    npimage=(z_fake[0]*m_depth).squeeze(0).cpu().detach().numpy().astype(np.uint16)
                    cv2.imwrite(outpath, npimage)

                else:
                    continue
            print('Predicting '+str(counter)+' images took ', time_sum/counter)  
        else:
            print("Predicting for:"+input_image_path)
            depth = cv2.imread(input_image_path,cv2.IMREAD_UNCHANGED).astype(np.float32)
            if len(depth.shape) < 3:
                print("Got 1 channel depth images, creating 3 channel depth images")
                combine_depth = np.empty((depth.shape[0],depth.shape[1], 3))
                combine_depth[:,:,0] = depth
                combine_depth[:,:,1] = depth
                combine_depth[:,:,2] = depth
                depth = combine_depth
            depth2 = np.moveaxis(depth,-1,0)
            img = torch.from_numpy(depth2).float().unsqueeze(0).cuda()
            start = timeit.default_timer()
            m_depth=torch.max(img)
            img=img/m_depth                 
            z_fake=dfilt(img)
            stop = timeit.default_timer()
            time_sum=time_sum+stop-start
            counter=counter+1
            save_path=path[:-4]
            npimage=(z_fake[0]*m_depth).squeeze(0).cpu().detach().numpy().astype(np.uint16)
            cv2.imwrite(save_path +'_pred.png', npimage)
    
