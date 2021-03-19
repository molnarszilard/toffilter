from model_fpn import DFILT
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
                      default='/home/user/dataset/depth_images/depth.png', type=str)
    parser.add_argument('--eval_folder', dest='eval_folder',
                      help='evaluate only one image or the whole folder',
                      default=False, type=bool)
    parser.add_argument('--model_path', dest='model_path',
                      help='path to the model to use',
                      default='saved_models/dfilt_1_9_v12.pth', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You might want to run with --cuda")
    
    # network initialization
    print('Initializing model...')
    dfilt = DFILT(fixed_feature_weights=False)
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

    dfilt.eval()

    img = Variable(torch.FloatTensor(1))

    print('evaluating...')
    if args.eval_folder:
        dlist=os.listdir(args.input_image_path)
        dlist.sort()
        time_sum = 0
        counter = 0
        for filename in dlist:
            if filename.endswith(".png"):
                path=args.input_image_path+filename
                print("Predicting for:"+filename)
                depth = cv2.imread(path,cv2.IMREAD_UNCHANGED).astype(np.float32)
                if len(depth.shape) < 3:
                    print("Got 1 channel depth images, creating 3 channel depth images")
                    combine_depth = np.empty((depth.shape[0],depth.shape[1], 3))
                    combine_depth[:,:,0] = depth
                    combine_depth[:,:,1] = depth
                    combine_depth[:,:,2] = depth
                    depth = combine_depth
                depth2 = np.moveaxis(depth,-1,0)
                img = torch.from_numpy(depth2).float().unsqueeze(0).cuda()
                max_depth=10000.
                start = timeit.default_timer()
                img2=img.clone()
                img2[img2>max_depth] = max_depth            
                imgmask=img2.clone()
                imgmask=imgmask[:,0,:,:].unsqueeze(1)
                valid = (imgmask > 0) & (imgmask < max_depth+1)
                img2=img2/max_depth                
                zero_number = torch.tensor(0.).to('cuda')        
                z_fake = dfilt(img2)
                z_fake = torch.where(valid, z_fake*max_depth, zero_number)
                stop = timeit.default_timer()
                time_sum=time_sum+stop-start
                counter=counter+1
                save_path=path[:-4]
                npimage=(z_fake[0]).squeeze(0).cpu().detach().numpy().astype(np.uint16)
                cv2.imwrite(save_path +'_pred.png', npimage)

            else:
                continue
        print('Predicting '+str(counter)+' images took ', time_sum/counter)  
    else:
        depth = cv2.imread(args.input_image_path,cv2.IMREAD_UNCHANGED).astype(np.float32)
        if len(depth.shape) < 3:
            print("Got 1 channel depth images, creating 3 channel depth images")
            combine_depth = np.empty((depth.shape[0],depth.shape[1], 3))
            combine_depth[:,:,0] = depth
            combine_depth[:,:,1] = depth
            combine_depth[:,:,2] = depth
            depth = combine_depth
        depth2 = np.moveaxis(depth,-1,0)
        img = torch.from_numpy(depth2).float().unsqueeze(0)
        start = timeit.default_timer()
        z_fake = dfilt(img.cuda())
        stop = timeit.default_timer()
        zfv=z_fake*2-1
        z_fake_norm=zfv.pow(2).sum(dim=1).pow(0.5).unsqueeze(1)
        zfv=zfv/z_fake_norm
        z_fake=(zfv+1)/2
        save_path=args.input_image_path[:-4]
        save_image(z_fake[0], save_path +"_pred"+'.png')
        print('Predicting the image took ', stop-start)
    
