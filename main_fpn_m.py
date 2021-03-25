import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from constants import *
from model_fpn import I2D
import argparse
import time
# from utils.net_utils import adjust_learning_rate
from torch.autograd import Variable
# from dataset.dataloader import DepthDataset
# from dataset.nyuv2_dataset import NYUv2Dataset
from torchvision.utils import save_image
from dataset.nyuv2_dataset import MyCustomDataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from collections import Counter
import matplotlib
import cv2
import open3d as o3d
matplotlib.use('Agg')

class DDDDepthDiff(nn.Module):
    def __init__(self):
        super(DDDDepthDiff, self).__init__()

    def forward(self, fake, real,epoch,show_image):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        eps = 1e-7
        # eps = 2

        batch_size = real.shape[0]
        
        real1 = real.clone() #real[0].cpu().detach().numpy()
        fake1 = fake.clone() #fake[0].cpu().detach().numpy()
        ###### debug purposes ########
        # fake1[real1==0] = 1.0
        # a = np.asarray(real1.cpu().detach()*7000)[0]
        # # a[a!=0.0] = 10000
        # b = np.asarray(fake1.cpu().detach()*7000)[0]
        # plt.imshow(np.uint16(b), vmin=0, vmax=7000)
        # plt.colorbar()
        # plt.savefig(save_dir +'fake1_'+str(epoch)+'.png',bbox_inches='tight')
        # plt.close()

        # plt.imshow(np.uint16(a), vmin=0, vmax=7000)
        # plt.colorbar()
        # plt.savefig(save_dir +'real1_'+str(epoch)+'.png',bbox_inches='tight')
        # plt.close()
        # b[b!=0.0] = 10000
        # c = np.abs(a-b)
        # cv2.imwrite(save_dir+'gttest_'+str(epoch)+'.png',np.uint16(a))
        # cv2.imwrite(save_dir+'faketest_'+str(epoch)+'.png',np.uint16(b))
        # cv2.imwrite(save_dir+'diff_test_'+str(epoch)+'.png',np.uint16(c))
        ####################################

        # real1[real1==0] = eps
        # fake1[fake1==0] = eps

        # for calculating the loss on all the images in the batch size (Thanks Szilard for telling me about this!!!)
        all_real_pcd = self.point_cloud(real1[0]).clone() * 1000.0
        all_fake_pcd = self.point_cloud(fake1[0]).clone() * 1000.0
        
        for nr_img in range(1,batch_size):
            real_pcd = self.point_cloud(real1[nr_img]).clone() * 1000.0
            fake_pcd = self.point_cloud(fake1[nr_img]).clone() * 1000.0

            all_real_pcd = torch.cat(all_real_pcd,real_pcd)
            all_fake_pcd = torch.cat(all_fake_pcd,fake_pcd)

        all_real_pcd[all_real_pcd==0] = eps
        all_fake_pcd[all_fake_pcd==0] = eps

        # real_pcd = nan_real_pcd[~torch.isnan(nan_real_pcd)]
        # fake_pcd = nan_fake_pcd[~torch.isnan(nan_real_pcd)]

        
        # ### loss 22
        # # remove nans from z...
        # nan_z_real = real_pcd[:,2].clone()
        # z_real = nan_z_real[~torch.isnan(nan_z_real)]
       
        # nan_z_fake = fake_pcd[:,2].clone()
        # z_fake = nan_z_fake[~torch.isnan(nan_z_real)]
        
        # # and replace the nans from x and y with 0.0
        # x_real = real_pcd[:,0].clone()
        # x_real[torch.isnan(x_real)] = 0.0
        
        # x_fake = fake_pcd[:,0].clone()
        # x_fake[torch.isnan(x_real)] = 0.0 
        
        # y_real = real_pcd[:,1].clone()
        # y_real[torch.isnan(y_real)] = 0.0
        
        # y_fake = fake_pcd[:,1].clone()
        # y_fake[torch.isnan(y_real)] = 0.0
        
        # lossX = torch.mean(torch.abs(x_real-x_fake))
        # lossY = torch.mean(torch.abs(y_real-y_fake))
        # lossZ = torch.mean(torch.abs(z_real-z_fake))
        # RMSE = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))
        # delta = [RMSE, lossX, lossY, lossZ]
        # loss22 = RMSE * torch.abs(10*(3-torch.exp(1*lossX)+torch.exp(1*lossY)+torch.exp(1*lossZ)))

        # ### loss 21
        
        # z_real = real_pcd[:,2].clone()
        # z_real[torch.isnan(z_real)] = 10.0
       
        # z_fake = fake_pcd[:,2].clone()
        # z_fake[torch.isnan(z_real)] = 10.0
        
        # x_real = real_pcd[:,0].clone()
        # x_real[torch.isnan(x_real)] = 10.0
        
        # x_fake = fake_pcd[:,0].clone()
        # x_fake[torch.isnan(x_real)] = 10.0 
        
        # y_real = real_pcd[:,1].clone()
        # y_real[torch.isnan(y_real)] = 10.0
        
        # y_fake = fake_pcd[:,1].clone()
        # y_fake[torch.isnan(y_real)] = 10.0
        
        # lossX = torch.mean(torch.abs(x_real-x_fake))
        # lossY = torch.mean(torch.abs(y_real-y_fake))
        # lossZ = torch.mean(torch.abs(z_real-z_fake))
        # RMSE = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))
        # delta = [RMSE, lossX, lossY, lossZ]
        # loss21 = RMSE * torch.abs(1*(3-torch.exp(1*lossX)+torch.exp(1*lossY)+torch.exp(1*lossZ)))
        

        
        ### for the next losses you will need this section 

        # nan_z_real = real_pcd[:,2].clone()
        # z_real = nan_z_real[~torch.isnan(nan_z_real)]
       
        # nan_z_fake = fake_pcd[:,2].clone()
        # z_fake = nan_z_fake[~torch.isnan(nan_z_real)]
        
        # nan_x_real = real_pcd[:,0].clone()
        # x_real = nan_x_real[~torch.isnan(nan_x_real)]
        
        # nan_x_fake = fake_pcd[:,0].clone()
        # x_fake = nan_x_fake[~torch.isnan(nan_x_real)]
        
        # nan_y_real = real_pcd[:,1].clone()
        # y_real = nan_y_real[~torch.isnan(nan_y_real)]
        
        # nan_y_fake = fake_pcd[:,1].clone()
        # y_fake = nan_y_fake[~torch.isnan(nan_y_real)]
        
        # loss = np.sqrt(np.mean(np.abs(np.log(z_real)-np.log(z_fake))**2))
        # dist_real = torch.sqrt(torch.sum(real_pcd**2,dim=1))
        # dist_fake = torch.sqrt(torch.sum(fake_pcd**2,dim=1))
        
        # loss2 = torch.sqrt(torch.mean(torch.abs(torch.log(dist_real)-torch.log(dist_fake)) ** 2))
        
        # ### lossXX
        # x_real = real_pcd[:,0].clone()
        # x_fake = fake_pcd[:,0].clone()
        # y_real = real_pcd[:,1].clone()
        # y_fake = fake_pcd[:,1].clone()
        # lossX = torch.mean(torch.exp(torch.abs(x_real-x_fake)**2))
        # lossY = torch.mean(torch.exp(torch.abs(y_real-y_fake)**2))
        # lossZ = torch.mean(torch.exp(torch.abs(z_real-z_fake)**2))
        # RMSE_term = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))
        # delta = [RMSE_term, lossX, lossY, lossZ]
        # loss18 = RMSE_term*torch.abs(3-lossX+lossY+lossZ)

        # ### lossXX
        # x_real = real_pcd[:,0].clone()
        # x_fake = fake_pcd[:,0].clone()
        # y_real = real_pcd[:,1].clone()
        # y_fake = fake_pcd[:,1].clone()
        # lossX = torch.mean(torch.exp(10*torch.abs(x_real-x_fake)))
        # lossY = torch.mean(torch.exp(10*torch.abs(y_real-y_fake)))
        # lossZ = torch.mean(torch.exp(10*torch.abs(z_real-z_fake)))
        # RMSE_term = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))
        # delta = [RMSE_term, lossX, lossY, lossZ]
        # loss17 = 10*RMSE_term+torch.abs(3-lossX+lossY+lossZ)


        # ### loss20
        # lossX = torch.mean(torch.exp(torch.abs(x_real-x_fake))**2)
        # lossZ = torch.mean(torch.exp(torch.abs(z_real-z_fake))**2)
        # lossY = torch.mean(torch.exp(torch.abs(y_real-y_fake))**2)
        # RMSE = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))
        # delta = [RMSE, lossX, lossY, lossZ]
        # loss20 = RMSE * torch.abs(10*(3-lossX+lossY+lossZ)) 

        # ### loss19
        # lossX = torch.mean(torch.exp(10*torch.abs(x_real-x_fake)))
        # lossZ = torch.mean(torch.exp(10*torch.abs(z_real-z_fake)))
        # lossY = torch.mean(torch.exp(10*torch.abs(y_real-y_fake)))
        # RMSE = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))
        # delta = [RMSE, lossX, lossY, lossZ]
        # loss19 = RMSE * torch.abs(10*(3-lossX+lossY+lossZ))
 
        # ### loss18
                
        # lossX = torch.mean(torch.abs(x_real-x_fake))
        # lossY = torch.mean(torch.abs(y_real-y_fake))
        # lossZ = torch.mean(torch.abs(z_real-z_fake))
        # RMSE = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))
        # delta = [RMSE, lossX, lossY, lossZ]
        # loss18 = RMSE + torch.abs(10*(3-torch.exp(1*lossX)+torch.exp(1*lossY)+torch.exp(1*lossZ)))

        #######################
        # Take out nan points #
        # If this doesn't work replace the values with 2 or something
        ### loss17
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
        

        ######Original########
        # lossX = torch.mean(torch.abs(x_real-x_fake))
        # lossZ = torch.mean(torch.abs(z_real-z_fake))
        # lossY = torch.mean(torch.abs(y_real-y_fake))
        ####sixth try #####
        lossX = torch.sqrt(torch.mean(torch.abs(x_real-x_fake)**2))
        lossZ = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))
        lossY = torch.sqrt(torch.mean(torch.abs(y_real-y_fake)**2))
        
        #######second#############
        # lossX = torch.mean(torch.abs(torch.log(torch.abs(x_real))-torch.log(torch.abs(x_fake))))
        # lossY = torch.mean(torch.abs(torch.log(torch.abs(y_real))-torch.log(torch.abs(y_fake))))
        # lossZ = torch.mean(torch.abs(torch.log(torch.abs(z_real))-torch.log(torch.abs(z_fake))))
        
        ##### third ######
        # lossX = torch.mean(torch.log(torch.abs(1-torch.abs(x_real-x_fake))))
        # lossY = torch.mean(torch.log(torch.abs(1-torch.abs(y_real-y_fake))))
        # lossZ = torch.mean(torch.log(torch.abs(1-torch.abs(z_real-z_fake))))   
        ##### fourth ######
        # lossX = 15 - torch.abs(torch.log(torch.mean(torch.abs(x_real-x_fake)**2)))
        # lossY = 15 - torch.abs(torch.log(torch.mean(torch.abs(y_real-y_fake)**2)))
        # lossZ = 15 - torch.abs(torch.log(torch.mean(torch.abs(z_real-z_fake)**2)))
        # RMSE = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))
        ######fifth ################
        # lossX = torch.sqrt(torch.mean(torch.abs(torch.log(torch.abs(x_real))-torch.log(torch.abs(x_fake)))**2))
        # lossZ = torch.sqrt(torch.mean(torch.abs(torch.log(torch.abs(z_real))-torch.log(torch.abs(z_fake)))**2))
        # lossY = torch.sqrt(torch.mean(torch.abs(torch.log(torch.abs(y_real))-torch.log(torch.abs(y_fake)))**2))

        RMSE_log = torch.sqrt(torch.mean(torch.abs(torch.log(torch.abs(z_real))-torch.log(torch.abs(z_fake)))**2))
        # RMSE_log2 = torch.sqrt(torch.mean(torch.log(torch.abs(z_real-z_fake)**2)))
        # RMSE_log3 = torch.sqrt(torch.mean(torch.log(1-torch.abs(z_real-z_fake))))
        delta = [RMSE_log, lossX, lossY, lossZ]
        # loss13 = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))*torch.abs(10*(torch.exp(1*(lossX-0.01))+torch.exp(1*(lossY-0.01))+torch.exp(1*(lossZ-0.05))))
        # loss17 = 100*RMSE_log * torch.abs(10*(3-torch.exp(1*lossX)-torch.exp(1*lossY)-torch.exp(1*lossZ)))
        loss18 = 10*(RMSE_log + torch.abs(10*(3-torch.exp(1*lossX)-torch.exp(1*lossY)-torch.exp(1*lossZ))))
        # loss17p1 = 10*RMSE_log * (lossX+lossY+lossZ)
        # loss17p2 = 10*(RMSE_log + lossX + lossY)
        
        # if show_image:
        #     ### A loss pointcloud?... Probably...
        #     real_pcd = real_pcd 
        #     fake_pcd = fake_pcd  

        #     z_real = real_pcd[:,2].clone()
        #     z_real[torch.isnan(z_real)] = 0.0
        
        #     z_fake = fake_pcd[:,2].clone()
        #     z_fake[torch.isnan(z_real)] = 0.0
            
        #     x_real = real_pcd[:,0].clone()
        #     x_real[torch.isnan(x_real)] = 0.0
            
        #     x_fake = fake_pcd[:,0].clone()
        #     x_fake[torch.isnan(x_real)] = 0.0 
            
        #     y_real = real_pcd[:,1].clone()
        #     y_real[torch.isnan(y_real)] = 0.0
            
        #     y_fake = fake_pcd[:,1].clone()
        #     y_fake[torch.isnan(y_real)] = 0.0

        #     ################ - simple difference
        #     coord_X = torch.abs(x_real-x_fake)
        #     coord_Y = torch.abs(y_real-y_fake)
        #     coord_Z = torch.abs(z_real-z_fake) 

            
        #     max_depth = 7000/1000.0
        #     o3d_pcd = o3d.geometry.PointCloud()
            
        #     loss_pcd = torch.stack((coord_X,coord_Y,coord_Z),dim=1).cpu().detach().numpy()
        #     o3d_pcd.points = o3d.utility.Vector3dVector(loss_pcd*max_depth)
        #     o3d.io.write_point_cloud(save_dir+"loss_diff_cloud"+str(epoch)+".pcd", o3d_pcd)
        #     loss_pcd_img = self.image_from_cloud(real_pcd.cpu().detach().numpy())
        #     # o3d.io.write_image(save_dir+"loss_diff_cloud"+str(epoch)+".png",loss_pcd_img)
            
        #     plt.imshow(loss_pcd_img*max_depth*1000.0, vmin=0, vmax=max_depth*1000.0)
        #     plt.colorbar()
        #     plt.savefig(save_dir+"loss_diff_cloud"+str(epoch)+".png",bbox_inches='tight')
        #     plt.close()
        #     ################ - difference with exp
        #     coord_X = torch.exp(torch.abs(x_real-x_fake))
        #     coord_Y = torch.exp(torch.abs(y_real-y_fake))
        #     coord_Z = torch.exp(torch.abs(z_real-z_fake))

            
        #     max_depth = 7000
        #     o3d_pcd = o3d.geometry.PointCloud()
            
        #     loss_pcd = torch.stack((coord_X,coord_Y,coord_Z),dim=1).cpu().detach().numpy()
        #     o3d_pcd.points = o3d.utility.Vector3dVector(loss_pcd*max_depth)
        #     o3d.io.write_point_cloud(save_dir+"loss_exp_diff_cloud"+str(epoch)+".pcd", o3d_pcd)
            
        #     ################# - difference with exp and rmse multiplied
        #     coord_X = torch.exp(torch.abs(x_real-x_fake))
        #     coord_Y = torch.exp(torch.abs(y_real-y_fake))
        #     coord_Z = torch.exp(torch.abs(z_real-z_fake))

        #     rmse_term = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2)).cpu().detach().numpy()
        #     max_depth = 7000
        #     o3d_pcd = o3d.geometry.PointCloud()
            
        #     loss_pcd = torch.stack((coord_X,coord_Y,coord_Z),dim=1).cpu().detach().numpy()
        #     o3d_pcd.points = o3d.utility.Vector3dVector(rmse_term*loss_pcd*max_depth)
        #     o3d.io.write_point_cloud(save_dir+"loss_exp_diff_multip_rmse_cloud"+str(epoch)+".pcd", o3d_pcd)
        # ### loss16
        # x_real = real_pcd[:,0].clone()
        # x_fake = fake_pcd[:,0].clone()
        # y_real = real_pcd[:,1].clone()
        # y_fake = fake_pcd[:,1].clone()
        # lossX = torch.mean(torch.exp(10*torch.abs(x_real-x_fake)))
        # lossY = torch.mean(torch.exp(10*torch.abs(y_real-y_fake)))
        # lossZ = torch.mean(torch.exp(10*torch.abs(z_real-z_fake)))
        # RMSE_term = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))
        # delta = [RMSE_term, lossX, lossY, lossZ]
        # loss16 = RMSE_term*torch.abs(3-lossX+lossY+lossZ)
        
        
        # ### loss15
        # x_real = real_pcd[:,0].clone()
        # x_fake = fake_pcd[:,0].clone()
        # y_real = real_pcd[:,1].clone()
        # y_fake = fake_pcd[:,1].clone()
        # lossX = torch.mean(torch.abs(x_real-x_fake))
        # lossY = torch.mean(torch.abs(y_real-y_fake))
        # lossZ = torch.mean(torch.abs(z_real-z_fake))
        # delta = [lossX, lossY, lossZ]
        # loss15 = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))*torch.abs(10*(torch.exp(1*(lossX-0.05))+torch.exp(1*(lossY-0.01))+torch.exp(1*(lossZ-0.1))))
        

        # ### loss14
        # x_real = real_pcd[:,0].clone()
        # x_fake = fake_pcd[:,0].clone()
        # y_real = real_pcd[:,1].clone()
        # y_fake = fake_pcd[:,1].clone()
        # lossX = 1000*torch.mean(torch.abs(x_real-x_fake))
        # lossY = 1000*torch.mean(torch.abs(y_real-y_fake))
        # lossZ = 1000*torch.mean(torch.abs(z_real-z_fake))
        # delta = [lossX, lossY, lossZ]
        # loss14 = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))*torch.abs(10*(torch.exp(1*(lossX-0.01))+torch.exp(1*(lossY-0.01))+torch.exp(1*(lossZ-0.05))))
        
        # ### loss13
        # x_real = real_pcd[:,0].clone()
        # x_fake = fake_pcd[:,0].clone()
        # y_real = real_pcd[:,1].clone()
        # y_fake = fake_pcd[:,1].clone()
        # lossX = torch.mean(torch.abs(x_real-x_fake))
        # lossY = torch.mean(torch.abs(y_real-y_fake))
        # lossZ = torch.mean(torch.abs(z_real-z_fake))
        # RMSE = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))
        # delta = [RMSE, lossX, lossY, lossZ]
        # loss13 = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))*torch.abs(10*(torch.exp(1*(lossX-0.01))+torch.exp(1*(lossY-0.01))+torch.exp(1*(lossZ-0.05))))
        
        # ### loss12
        # x_real = real_pcd[:,0].clone()
        # x_fake = fake_pcd[:,0].clone()
        # y_real = real_pcd[:,1].clone()
        # y_fake = fake_pcd[:,1].clone()
        # lossX = torch.mean(torch.abs(x_real-x_fake))
        # lossY = torch.mean(torch.abs(y_real-y_fake))
        # lossZ = torch.mean(torch.abs(z_real-z_fake))
        # delta = [lossX, lossY, lossZ]
        # loss12 = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))*torch.abs(10*(1-torch.exp(torch.exp(1*(lossX-0.01))+torch.exp(1*(lossY-0.01))+torch.exp(1*(lossZ-0.05)))))
        

        # ### loss11
        # x_real = real_pcd[:,0].clone()
        # x_fake = fake_pcd[:,0].clone()
        # y_real = real_pcd[:,1].clone()
        # y_fake = fake_pcd[:,1].clone()
        # lossX = torch.mean(torch.abs(x_real-x_fake))
        # lossY = torch.mean(torch.abs(y_real-y_fake))
        # lossZ = torch.mean(torch.abs(z_real-z_fake))
        # delta = [lossX, lossY, lossZ]
        # loss11 = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))*torch.abs(1-torch.exp(10*(lossX-0.01)))*torch.abs(1-torch.exp(10*(lossY-0.01)))*torch.abs(1-torch.exp(10*(lossZ-0.05)))
        
        # ### loss10
        # delta = torch.mean(torch.abs(z_real-z_fake))
        # loss10 = torch.abs(1-torch.exp(10*(delta-0.005)))

        # ### loss9
        # delta = torch.abs(torch.mean(z_real)-torch.mean(z_fake))
        # loss9 = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2)) * torch.abs(1-torch.exp(10*(delta-0.005)))
        
        # ### loss8
        # delta = torch.mean(torch.abs(z_real-z_fake))
        # loss8 = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))*torch.abs(1-torch.exp(10*(delta-0.05)))

        # ### loss7
        # delta = torch.mean(torch.abs(z_real-z_fake))
        # loss7 = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2))*torch.abs(1-torch.exp(10*(delta-0.005)))
        
        ### loss6
        # delta = torch.mean(torch.abs(z_real-z_fake))
        # loss6 = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2)) * torch.exp(10*(delta-0.005))
        
        ### loss 5
        # delta = torch.mean(torch.abs(z_real-z_fake))
        # loss5 = torch.sqrt(torch.mean(torch.abs(z_real-z_fake)**2)) * torch.exp(10*(delta-0.1))
        
        # loss4 = torch.sqrt(torch.mean(torch.exp(torch.abs(z_real-z_fake))))
        
        # # 3. RMSE with condition
        # delta = torch.mean(torch.abs(z_real-z_fake))
        
        # loss3 = torch.sqrt(torch.mean(torch.abs(torch.exp(z_real)-torch.exp(z_fake)) ** 2))

        
        
        # # 2. RMSE on every  coordinate
        # x_real = real_pcd[:,0].clone()
        # x_fake = fake_pcd[:,0].clone()
        # y_real = real_pcd[:,1].clone()
        # y_fake = fake_pcd[:,1].clone()
        # loss2x = torch.sqrt(torch.mean((x_real-x_fake)**2))
        # loss2y = torch.sqrt(torch.mean((y_real-y_fake)**2))
        # loss2z = torch.sqrt(torch.mean((z_real-z_fake)**2))
        # loss2 = loss2x + loss2y + loss2z

        

        # # 1. without log
        # loss1 = torch.sqrt(torch.mean((z_real-z_fake) ** 2))

        # #0. simple loss function on z 
        # loss = torch.sqrt(torch.mean(torch.abs(torch.log(z_real)-torch.log(z_fake)) ** 2))
        
        # loss = torch.sqrt(torch.mean(
        #     torch.abs(torch.log(real2)-torch.log(fake2)) ** 2))
        return delta, loss18
    
    def l2_norm(self,v):
        norm_v = np.sqrt(np.sum(np.square(v), axis=1))
        return norm_v

    # def theta(v, w): return arccos(v.dot(w)/(norm(v)*norm(w)))

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
        cx = 334.081
        cy = 169.808
        fx = 460.585
        fy = 460.268

        depth = depth1.clone()
        # open3d_img = o3d.t.geometry.Image(depth[0])#/1000.0)
        # intrinsics = o3d.camera.PinholeCameraIntrinsic(640,360,fx,fy,cx,cy)
        # pcd = o3d.geometry.create_point_cloud_from_depth_image(open3d_img,intrinsic=intrinsics)
        
        rows, cols = depth[0].shape
        c, _ = torch.meshgrid(torch.arange(cols), torch.arange(cols))
        c = torch.meshgrid(torch.arange(cols))
        new_c = c[0].reshape([1,cols]).to('cuda')
        r = torch.meshgrid(torch.arange(rows))
        new_r = r[0].unsqueeze(-1).to('cuda')
        valid = (depth[0] > 0) & (depth[0] < 65535)
        nan_number = torch.tensor(np.nan).to('cuda')
        zero_number = torch.tensor(0.).to('cuda')
        z = torch.where(valid, depth[0]/1000.0, nan_number) # allways divide with 1000.0
        x = torch.where(valid, z * (new_c - cx) / fx, nan_number)
        y = torch.where(valid, z * (new_r - cy) / fy, nan_number)
        

        dimension = rows * cols
        z_ok = z.reshape(dimension)
        x_ok = x.reshape(dimension)
        y_ok = y.reshape(dimension)
    
        return torch.stack((x_ok,y_ok,z_ok),dim=1) 

    def image_from_cloud(self, point_cloud):
        
        cx = 334.081
        cy = 169.808
        fx = 460.585
        fy = 460.268
        
        # point_cloud = point_cloud/1000.0
        np_image = np.tile(0,(360,640))

        z = point_cloud[:,2] * 1000.0
        x = point_cloud[:,0]
        y = point_cloud[:,1]

        valid = ~(np.isnan(z) | (z==0))
        z = np.where(valid, z*1000.0, 0)
        # z[np.isnan(z) | (z==0)] = 1e-7
        # pos_x = (point_cloud[:,0] * 1000.0 * fx)/ z + cx
        valid_x = ~(np.isnan(x) | np.isnan(z) | (z==0))
        pos_x = np.where(valid_x,(x * 1000.0 * fx)/ z + cx, 0)
        pos_x = pos_x.astype(np.int32)
        # pos_y = (point_cloud[:,1] * 1000.0 * fy)/ z + cy
        valid_y = ~(np.isnan(y) | np.isnan(z) | (z==0))
        pos_y = np.where(valid_y,(y * 1000.0 * fy)/z + cy, 0) 
        pos_y = pos_y.astype(np.int32)
        
        pos_x[pos_x>639] = 639
        pos_x[pos_x<0] = 0
        pos_y[pos_y>359] =359
        pos_y[pos_y<0] = 0

        pos_x = pos_x.reshape(360,640)
        pos_y = pos_y.reshape(360,640)
        z = z.reshape(360,640)
        np_image[pos_y,pos_x] = z

        # depth = depth.cpu().detach().numpy()
        # rows, cols = depth[0].shape
        # c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        # valid = (depth[0] > 0) & (depth[0] < 65535)
        # z = np.where(valid, depth[0] / 1000.0, np.nan)
        # x = np.where(valid, z * (c - cx) / fx, 0)
        # y = np.where(valid, z * (r - cy) / fy, 0)
        return np_image

class NormalsDiff(nn.Module):
    def __init__(self):
        super(NormalsDiff, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        eps = 1e-7
        real = real[0].cpu().detach().numpy()
        fake = fake[0].cpu().detach().numpy()
        real[real==0] = eps
        fake[fake==0] = eps


        real_pcd = self.point_cloud(real)
        fake_pcd = self.point_cloud(fake)
        
        o3d.geometry.estimate_normals(real_pcd,search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03,max_nn=30))
        o3d.geometry.estimate_normals(fake_pcd,search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03,max_nn=30))

        real_normals = np.array(real_pcd.normals)
        fake_normals = np.array(fake_pcd.normals)
        
        
        normal_gt_norm = self.l2_norm(real_normals)
        normal_results_norm = self.l2_norm(fake_normals)

        normals_results = np.divide(fake_normals, np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))
        normals_gt = np.divide(real_normals, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))

        # Not oriented rms
        nn = np.sum(np.multiply(normals_gt, normals_results), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1

        angle = np.rad2deg(np.arccos(np.abs(nn))) 

        # inner_product = (fake_normals * real_normals).sum(1)
        # fake_norm = fake_normals.pow(2).sum(1).pow(0.5)
        # real_norm = real_normals.pow(2).sum(1).pow(0.5)
        # cos = inner_product / (2 * fake_norm * real_norm)
        # angle = torch.acos(cos)
        # eps=1e-7
        # # gt2 = gt.clone()
        # # pred2 = pred.clone()
        # gt2[gt2==0] = eps
        # pred2[pred2==0] = eps
        # if method == 0:
        # o3d.visualization.draw_geometries([real_pcd])
        # o3d.io.write_point_cloud("/home/marian/calibration_ws/monodepth-FPN/MonoDepth-FPN-PyTorch/dataset/training_data/training_data/training_process_debug/python_cloud_normals003.pcd", real_pcd)
        # o3d.geometry.estimate_normals(real_pcd,search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=60,max_nn=30))
        # o3d.io.write_point_cloud("/home/marian/calibration_ws/monodepth-FPN/MonoDepth-FPN-PyTorch/dataset/training_data/training_data/training_process_debug/python_cloud_normals60.pcd", real_pcd)
        # o3d.io.write_point_cloud("/home/marian/calibration_ws/monodepth-FPN/MonoDepth-FPN-PyTorch/dataset/training_data/training_data/training_process_debug/python_cloud_pred.pcd", fake_pcd)
        # print(real)
        loss = np.mean(angle)
        
        return loss
    
    def l2_norm(self,v):
        norm_v = np.sqrt(np.sum(np.square(v), axis=1))
        return norm_v

    # def theta(v, w): return arccos(v.dot(w)/(norm(v)*norm(w)))

    def point_cloud(self, depth):
        """Transform a depth image into a point cloud with one point for each
        pixel in the image, using the camera transform for a camera
        centred at cx, cy with field of view fx, fy.

        depth is a 2-D ndarray with shape (rows, cols) containing
        depths from 1 to 254 inclusive. The result is a 3-D array with
        shape (rows, cols, 3). Pixels with invalid depth in the input have
        NaN for the z-coordinate in the result.

        """
        # depth is of shape (1,480,640)
        cx = 334.081
        cy = 169.808
        fx = 460.585
        fy = 460.268

        open3d_img = o3d.geometry.Image(depth[0]/1000.0)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(640,360,fx,fy,cx,cy)
        pcd = o3d.geometry.create_point_cloud_from_depth_image(open3d_img,intrinsic=intrinsics)
        
        # rows, cols = depth[0].shape
        # c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        # valid = (depth[0] > 0) & (depth[0] < 65535)
        # z = np.where(valid, depth[0] / 1000.0, np.nan)
        # x = np.where(valid, z * (c - cx) / fx, 0)
        # y = np.where(valid, z * (r - cy) / fy, 0)
        return pcd #np.dstack((x, y, z))

    


class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        eps=1e-7
        real2 = real.clone()
        fake2 = fake.clone()
        real2[real2==0] = eps
        fake2[fake2==0] = eps
        loss = torch.sqrt(torch.mean(
            torch.abs(torch.log(real2)-torch.log(fake2)) ** 2))
        return loss


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        loss = torch.mean(torch.abs(10.*real-10.*fake))
        return loss


class L1_log(nn.Module):
    def __init__(self):
        super(L1_log, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        loss = torch.mean(torch.abs(torch.log(real)-torch.log(fake)))
        return loss


class BerHu(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold

    def forward(real, fake):
        mask = real > 0
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        fake = fake * mask
        diff = torch.abs(real-fake)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()[0]

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
        part2 = part2 / (2.*delta)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        loss = torch.sqrt(torch.mean(torch.abs(10.*real-10.*fake) ** 2))
        return loss


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    # L1 norm
    def forward(self, grad_fake, grad_real):

        return torch.sum(torch.mean(torch.abs(grad_real-grad_fake)))


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, grad_fake, grad_real):
        prod = (grad_fake[:, :, None, :] @
                grad_real[:, :, :, None]).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt(torch.sum(grad_fake**2, dim=-1))
        real_norm = torch.sqrt(torch.sum(grad_real**2, dim=-1))
        eps=1e-7
        real_norm2 = real_norm.clone()
        fake_norm2 = fake_norm.clone()
        real_norm2[real_norm2==0] = eps
        fake_norm2[fake_norm2==0] = eps

        return 1 - torch.mean(prod/(fake_norm2*real_norm2))

# def get_acc(output, target):
#     # takes in two tensors to compute accuracy
#     pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#     correct = pred.eq(target.data.view_as(pred)).cpu().sum()
#     print("Target: ", Counter(target.data.cpu().numpy()))
#     print("Pred: ", Counter(pred.cpu().numpy().flatten().tolist()))
#     return float(correct)*100 / target.size(0)


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Single image depth estimation')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='nyuv2', type=str)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=10, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
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
                        default=5, type=int)
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
                        default=1, type=int)
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
    i_range = Variable(torch.arange(0, h).view(
        1, h, 1).expand(b, 1, h, w))  # [B, 1, H, W]
    j_range = Variable(torch.arange(0, w).view(
        1, 1, w).expand(b, 1, h, w))  # [B, 1, H, W]
    coords = torch.cat((j_range, i_range), dim=1)
    norm = Variable(torch.Tensor([w, h]).view(1, 2, 1, 1))
    coords = coords * 2. / norm - 1.
    coords = coords.permute(0, 2, 3, 1)

    return coords


def resize_tensor(img, coords):
    return nn.functional.grid_sample(img, coords, mode='bilinear', padding_mode='zeros')


def imgrad(img):
    img = torch.mean(img, 1, True)
    # fx = np.array([[[1,0,-1],[1,0,-1],[1,0,-1]],[[2,0,-2],[2,0,-2],[2,0,-2]],[[1,0,-1],[1,0,-1],[1,0,-1]]])
    # conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=2, bias=False)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    # fy = np.array([[[1,2,1],[1,2,1],[1,2,1]],[[0,0,0],[0,0,0],[0,0,0]],[[-1,-2,-1],[-1,-2,-1],[-1,-2,-1]]])
    # conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

#     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))

    return grad_y, grad_x


def imgrad_yx(img):
    N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)


def reg_scalor(grad_yx):
    return torch.exp(-torch.abs(grad_yx)/255.)


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(
                self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(
            self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(
            self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat(
                (self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


def collate_fn(data):
    imgs, depths = zip(*data)
    B = len(imgs)
    im_batch = torch.ones((B, 3, 376, 1242))
    d_batch = torch.ones((B, 1, 376, 1242))
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

    train_dataset = MyCustomDataset()
    train_size = len(train_dataset)
    eval_dataset = MyCustomDataset(train=False)
    eval_size = len(eval_dataset)
    print(train_size)
    print(eval_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                                                   shuffle=True, num_workers=args.num_workers)
    # nr_of_pixels = len(train_dataset)*640*480*3
    # finding max depth and ir values
    # max_ir_value = 0
    # max_d_value = 0
    # min_ir_value = 100000
    # min_d_value = 100000
    # for batch in train_dataloader:
    #     # batch[0][0][0][batch[0][0][0]!=0].min() 
        
    #     # max depth value
    #     if batch[0][0][1].max() > max_d_value:
    #         max_d_value = batch[0][0][1].max()
    #     # max ir value
    #     if batch[0][0][0].max() > max_ir_value:
    #         max_ir_value = batch[0][0][0].max()

    #     # min depth value
    #     if batch[0][0][1][batch[0][0][1] != 0].min() < min_d_value:
    #         min_d_value = batch[0][0][1][batch[0][0][1] != 0].min()
    #     # min ir value
    #     if batch[0][0][0][batch[0][0][0] != 0].min() < min_ir_value:
    #         min_ir_value = batch[0][0][0][batch[0][0][0] != 0].min()

    # print("max_d_value: ", max_d_value, "min_d_value",min_d_value)
    # print("max_ir_value: ", max_ir_value, "min_ir_value",min_ir_value)
    
    # depth_test = cv2.imread("/home/marian/calibration_ws/monodepth-FPN/MonoDepth-FPN-PyTorch/dataset/training_data/training_data/combined_ir_d_d_debug/train/image.png",-1)
    # min_d_value = 100000
    # max_d_value = 0

    # mean = total_sum / nr_of_pixels

    # sum_of_squared_error = 0
    # for batch in train_dataloader: 
    #     sum_of_squared_error += ((batch[0] - mean).pow(2)).sum()
    # std = torch.sqrt(sum_of_squared_error / nr_of_pixels)
    
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.bs,
                                                  shuffle=True, num_workers=args.num_workers) #maybe trying with False for shuffle, here?
    # same as above but on validation set
    # max_ir_value = 0
    # max_d_value = 0
    # min_ir_value = 100000
    # min_d_value = 100000
    # for batch in eval_dataloader:
    #     # batch[0][0][0][batch[0][0][0]!=0].min() 
        
    #     # max depth value
    #     if batch[0][0][1].max() > max_d_value:
    #         max_d_value = batch[0][0][1].max()
    #     # max ir value
    #     if batch[0][0][0].max() > max_ir_value:
    #         max_ir_value = batch[0][0][0].max()

    #     # min depth value
    #     if batch[0][0][1][batch[0][0][1] != 0].min() < min_d_value:
    #         min_d_value = batch[0][0][1][batch[0][0][1] != 0].min()
    #     # min ir value
    #     if batch[0][0][0][batch[0][0][0] != 0].min() < min_ir_value:
    #         min_ir_value = batch[0][0][0][batch[0][0][0] != 0].min()

    # print("evaluation dataset")
    # print("max_d_value: ", max_d_value, "min_d_value",min_d_value)
    # print("max_ir_value: ", max_ir_value, "min_ir_value",min_ir_value)
    
    # network initialization
    print('Initializing model...')
    i2d = I2D(fixed_feature_weights=False)
    torch.cuda.empty_cache()
    if args.cuda:
        i2d = i2d.cuda()

    print('Done!')

    # hyperparams
    lr = args.lr
    bs = args.bs
    lr_decay_step = args.lr_decay_step
    lr_decay_gamma = args.lr_decay_gamma

    # params
    params = []
    for key, value in dict(i2d.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                DOUBLE_BIAS = 0
                WEIGHT_DECAY = 4e-5
                params += [{'params': [value], 'lr':lr*(DOUBLE_BIAS + 1),
                            'weight_decay': 4e-5 and WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr':lr, 'weight_decay': 4e-5}]

    # optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    rmse = RMSE()
    depth_criterion = RMSE_log()
    dddDepth_criterion = DDDDepthDiff()
    l1_crit = L1()
    normals_diff = NormalsDiff()
    grad_criterion = GradLoss()
    normal_criterion = NormalLoss()
    eval_metric = RMSE_log()

    # resume
    if args.resume:
        load_name = os.path.join(args.output_dir,
                                 'i2d_1_{}.pth'.format(args.checkepoch))
        print("loading checkpoint %s" % (load_name))
        state = i2d.state_dict()
        checkpoint = torch.load(load_name)
        args.start_epoch = checkpoint['epoch']
        checkpoint = {k: v for k,
                      v in checkpoint['model'].items() if k in state}
        state.update(checkpoint)
        i2d.load_state_dict(state)
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
        del checkpoint
        torch.cuda.empty_cache()

    # constants
    iters_per_epoch = int(train_size / args.bs)

    grad_factor = 10.
    normal_factor = 1.
    # max_depth = 6571
    max_depth = 8000
    
    #for visualizing the train and validation loss
    train_loss_arr = []
    val_loss_arr = []

    for epoch in range(args.start_epoch, args.max_epochs):

        train_loss = 0 
        val_loss = 0
        # setting to train mode
        i2d.train()
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
        show_image = True
        # saving results in a txt file
        save_dir = '/home/marian/calibration_ws/monodepth-FPN/MonoDepth-FPN-PyTorch/dataset/training_data/training_data/training_process/'
        
        
        for step in range(iters_per_epoch):
            start = time.time()
            data = train_data_iter.next()

            img.resize_(data[0].size()).copy_(data[0])#*max_depth)
            z.resize_(data[1].size()).copy_(data[1])#*max_depth)

            # max_depth = data[1].max()

            optimizer.zero_grad()
            z_fake = i2d(img)#*max_depth # * 6000 #z.max()
            
            # depth_loss = depth_criterion(z_fake, z)
            
            delta, dddDepth_loss = dddDepth_criterion(z_fake,z,epoch,show_image)#*max_depth,z*max_depth)
            # dddDepth_loss = dddDepth_criterion(z_fake,z)

            # grad_real, grad_fake = imgrad_yx(z), imgrad_yx(z_fake)
            
            # if epoch > 3:
            #     grad_loss = grad_criterion(grad_fake, grad_real) * grad_factor * (epoch > 3)
            # else: 
            #     grad_loss = 0
            
            # if epoch > 7:
            #     normals_diff_loss = normals_diff(z_fake*max_depth,z*max_depth) * (epoch > 7)
            #     # normal_loss = normal_criterion(grad_fake, grad_real) * normal_factor * (epoch > 7)
            # else:
            #     normals_diff_loss = 0
            #     # normal_loss = 0
            
 
            # loss = 10*(depth_loss + 0.01*grad_loss) + normals_diff_loss #+ normal_loss
            # loss = depth_loss + grad_loss + normal_loss
            # depth_loss_arr.append(depth_loss)
            # dddDepth_loss_arr.append(dddDepth_loss)
            
            torch.autograd.set_detect_anomaly(True)

            # if delta > 0.193 and epoch > 6:
            #     loss = (10*dddDepth_loss)**2
            # else:
            #     loss = 10*dddDepth_loss #depth_loss + 10*dddDepth_loss - depth_loss #+ normal_loss
            loss = 1*dddDepth_loss
            # loss *= 10
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            end = time.time()
            
            if show_image:
                # for i in range(img.shape[0]):
                # plt.imshow(np.transpose(imgs[i], (1, 2, 0)))
                # plt.show()
                # save_image(img[0], save_dir+'depthirPIL_'+str(epoch)+'.png')
                # plt.imshow(img[0].cpu().numpy().transpose((1,2,0)))
                # plt.savefig(save_dir +'depthir_'+str(epoch)+'.png',bbox_inches='tight')
                # plt.close()
                # rgbArray = np.zeros((len(img[0][1]),len(img[0][1][1]),3), 'uint16')
                o3d_pcd = o3d.geometry.PointCloud()

                ##############################
                #####save input cloud#########
                rgbArray = np.array(img[0].cpu()*max_depth,np.uint16).transpose((1,2,0))
                cv2.imwrite(save_dir+'depthirCV_'+str(epoch)+'.png',rgbArray)
                
                input_depth = img.clone() 
                input_pcd = dddDepth_criterion.point_cloud(input_depth[0]).cpu().detach().numpy()
                o3d_pcd.points = o3d.utility.Vector3dVector(input_pcd*max_depth)
                o3d.io.write_point_cloud(save_dir+"input_cloud"+str(epoch)+".pcd", o3d_pcd)
                # a = cv2.imread(save_dir+'depthirCV_'+str(epoch)+'.png', cv2.IMREAD_UNCHANGED)
                # vmin, vmax = 0, 10000/65536.

                ####################
                #depth ground truth#
                plt.imshow(z[0].cpu().numpy().transpose((1,2,0))*max_depth, vmin=0, vmax=max_depth)
                plt.colorbar()
                plt.savefig(save_dir +'gt_'+str(epoch)+'.png',bbox_inches='tight')
                plt.close()
                # plt.imshow(z[0].cpu().numpy().transpose((1,2,0)))#, vmin=vmin, vmax=vmax)
                # plt.colorbar()
                # plt.savefig(save_dir +'unscaled_gt_'+str(epoch)+'.png',bbox_inches='tight')
                # plt.close()
                z_pcd = dddDepth_criterion.point_cloud(z[0]).cpu().detach().numpy()
                o3d_pcd.points = o3d.utility.Vector3dVector(z_pcd*max_depth)
                o3d.io.write_point_cloud(save_dir+"gt_cloud"+str(epoch)+".pcd", o3d_pcd)
                
                ##################
                #depth prediction#
                plt.imshow(z_fake[0].cpu().detach().numpy().transpose((1,2,0))*max_depth, vmin=0, vmax=max_depth)
                plt.colorbar()
                plt.savefig(save_dir +'pred_'+str(epoch)+'.png',bbox_inches='tight')
                plt.close()
                # plt.imshow(z_fake[0].cpu().detach().numpy().transpose((1,2,0)))#, vmin=vmin, vmax=vmax)
                # plt.colorbar()
                # plt.savefig(save_dir +'unscaled_pred_'+str(epoch)+'.png',bbox_inches='tight')
                # plt.close()
                z_fake_pcd = dddDepth_criterion.point_cloud(z_fake[0]).cpu().detach().numpy()
                o3d_pcd.points = o3d.utility.Vector3dVector(z_fake_pcd*max_depth)
                o3d.io.write_point_cloud(save_dir+"pred_cloud"+str(epoch)+".pcd", o3d_pcd)

                ##############
                # txt images #
                # img_file = open(save_dir+'gt_'+str(epoch)+'.txt',"w")
                # for row in z[0].cpu().numpy():
                #     np.savetxt(img_file,row)
                # img_file.close()
                
                # # save_image(z_fake[0], save_dir+'predPIL_'+str(epoch)+'.png')
                # img_file = open(save_dir+'pred_'+str(epoch)+'.txt',"w")
                # for row in z_fake[0].cpu().detach().numpy():
                #     np.savetxt(img_file,row)
                # img_file.close()


                
                #### save difference ####
                plt.imshow(np.abs(z[0].cpu().numpy().transpose((1,2,0)) - z_fake[0].cpu().detach().numpy().transpose((1,2,0)))*max_depth)
                plt.colorbar()
                plt.savefig(save_dir+'diff_'+str(epoch)+'.png', bbox_inches='tight')
                plt.close()

                # z_diff = dddDepth_criterion.point_cloud(torch.abs(z[0]-z_fake[0])).cpu().detach().numpy()
                # o3d_pcd.points = o3d.utility.Vector3dVector(z_diff*max_depth)
                # o3d.io.write_point_cloud(save_dir+"diff_cloud"+str(epoch)+".pcd", o3d_pcd)
                

                show_image=False


            # info
            if step % args.disp_interval == 0:
                # file_object = open("/home/marian/calibration_ws/monodepth-FPN/MonoDepth-FPN-PyTorch/results.txt", 'a')
                print("[epoch %2d][iter %4d] loss: %.4f 3DDepthLoss: %.4f RMSE: %.4f lossX: %.4f lossY: %.4f lossZ: %.4f"#RMSElog: %.4f Grad: %.4f Normals diff: %.4f"
                      % (epoch, step, loss, dddDepth_loss, delta[0], delta[1], delta[2], delta[3]))#depth_loss, grad_loss, normals_diff_loss))
                # print("[epoch %2d][iter %4d] loss: %.4f RMSElog: %.4f Grad: %.4f Normals loss: %.4f"
                #       % (epoch, step, loss, depth_loss, grad_loss, normal_loss))

                # print("[epoch %2d][iter %4d] loss: %.4f RMSElog: %.4f"
                #       % (epoch, step, loss, depth_loss))
                # file_object.write("\n[epoch %2d][iter %4d] loss: %.4f RMSElog: %.4f" #grad_loss: %.4f" # normal_loss: %.4f" 
                #       % (epoch, step, loss, depth_loss))#, grad_loss))#, normal_loss))
                # file_object.close()
#                 print("[epoch %2d][iter %4d] loss: %.4f iRMSE: %.4f" \
#                                 % (epoch, step, loss, metric))
        # save model
        # plt.plot(depth_loss_arr,'g',dddDepth_loss_arr,'r')
        # plt.savefig(save_dir +'train_loss_'+str(epoch)+'.png',bbox_inches='tight')
        # plt.close()

        if epoch%4 == 0:
            save_name = os.path.join(args.output_dir, 'i2d_{}_{}.pth'.format(args.session, epoch))
            
            torch.save({'epoch': epoch+1,
                        'model': i2d.state_dict(),
                        #                     'optimizer': optimizer.state_dict(),
                        },
                    save_name)
            


        print('save model: {}'.format(save_name))
        print('time elapsed: %fs' % (end - start))

        # if epoch % 1 == 0:
        with torch.no_grad():
            # setting to eval mode
            i2d.eval()

            # img = Variable(torch.FloatTensor(1), volatile=True)
            # img = Variable(torch.FloatTensor(1),requires_grad=False)
            img = Variable(torch.FloatTensor(1))
            # z = Variable(torch.FloatTensor(1), volatile=True)
            # z = Variable(torch.FloatTensor(1), requires_grad=False)
            z = Variable(torch.FloatTensor(1))
            if args.cuda:
                img = img.cuda()
                z = z.cuda()

            print('evaluating...')

            rmse_accum = 0
            count = 0
            eval_data_iter = iter(eval_dataloader)
            for i, data_eval in enumerate(eval_data_iter):
                print(i, '/', len(eval_data_iter)-1)

                img.resize_(data_eval[0].size()).copy_(data_eval[0])
                z.resize_(data_eval[1].size()).copy_(data_eval[1])

                z_fake = i2d(img)

                depth_loss_eval = depth_criterion(z_fake,z)
                
                grad_real, grad_fake = imgrad_yx(z), imgrad_yx(z_fake)
                
                delta_val, dddDepth_loss_eval = dddDepth_criterion(z_fake,z,epoch,show_image)
                # dddDepth_loss_eval = dddDepth_criterion(z_fake,z)
                # if epoch > 3:
                #     grad_loss_eval = grad_criterion(grad_fake, grad_real) * grad_factor  #* (epoch > 3)
                # else:
                #     grad_loss_eval = 0
                
                # if epoch > 7:
                #     normals_diff_loss_eval = normals_diff(z_fake*max_depth,z*max_depth) #* (epoch > 7)
                #     # normal_loss_eval = normal_criterion(grad_fake, grad_real) * normal_factor * (epoch > 7)

                # else:
                #     normals_diff_loss_eval = 0
                #     normal_loss_eval = 0

                # loss_val = 10*(depth_loss_eval + 0.01*grad_loss_eval) + normals_diff_loss_eval
                # loss_val = depth_loss_eval + grad_loss_eval + normal_loss_eval
                # loss_val *= 10
                # if delta_val > 0.193 and epoch > 7:
                #     loss = (10*dddDepth_loss_eval)**2
                # else:
                #     loss_val = 10*dddDepth_loss_eval #depth_loss_eval + 10*dddDepth_loss_eval - depth_loss_eval
                loss_val = 1*dddDepth_loss_eval
                val_loss += loss_val.item()
                # print("Loss on test_data: ",loss_eval)
                if i==337:
                    plt.imshow(z[0].cpu().numpy().transpose((1,2,0))*max_depth)#, vmin=vmin, vmax=vmax)
                    plt.colorbar()
                    plt.savefig('/home/marian/calibration_ws/monodepth-FPN/MonoDepth-FPN-PyTorch/dataset/training_data/training_data/vis_images/gt_'+str(epoch)+'.png',bbox_inches='tight')
                    plt.close()

                    plt.imshow(z_fake[0].cpu().detach().numpy().transpose((1,2,0))*max_depth)#, vmin=vmin, vmax=vmax)
                    plt.colorbar()
                    plt.savefig('/home/marian/calibration_ws/monodepth-FPN/MonoDepth-FPN-PyTorch/dataset/training_data/training_data/vis_images/pred_'+str(epoch)+'.png',bbox_inches='tight')
                    plt.close()

                # save_image(z_fake[0],'/home/marian/calibration_ws/monodepth-FPN/MonoDepth-FPN-PyTorch/dataset/training_data/training_data/vis_images/depth_pred_'+str(epoch)+'_'+'.png')
                # depth_loss = float(img.size(0)) * rmse(z_fake, z)**2
                # eval_loss += depth_loss
                # rmse_accum += float(img.size(0)) * eval_metric(z_fake, z)**2
                # count += float(img.size(0))

            train_loss = train_loss/iters_per_epoch #len(train_dataloader)
            val_loss = val_loss/len(eval_dataloader)

            train_loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))

            file_object = open("/home/marian/calibration_ws/monodepth-FPN/MonoDepth-FPN-PyTorch/results.txt", 'a')
            # print("[epoch %2d][iter %4d] loss: %.4f RMSElog: %.4f"# grad_loss: %.4f"# normal_loss: %.4f"
            #         % (epoch, step, loss, depth_loss))#, grad_loss))#, normal_loss))
            # print("[epoch %2d][iter %4d] loss: %.4f RMSElog: %.4f"
            #       % (epoch, step, loss, depth_loss))
            file_object.write('\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss)) #grad_loss: %.4f" # normal_loss: %.4f" 
                    # % (epoch, step, loss, depth_loss))#, grad_loss))#, normal_loss))
            file_object.close()
            # print("[epoch %2d] RMSE_log: %.4f RMSE: %.4f"
            #       % (epoch, torch.sqrt(eval_loss/count), torch.sqrt(rmse_accum/count)))
            # with open('val.txt', 'a') as f:
            #     f.write("[epoch %2d] RMSE_log: %.4f RMSE: %.4f\n"
            #             % (epoch, torch.sqrt(eval_loss/count), torch.sqrt(rmse_accum/count)))
        
    # plt.plot(train_loss_arr,'g',val_loss_arr,'r')
    # plt.legend((train_loss_arr, val_loss_arr),('training loss', 'validation loss'))
    # plt.savefig(save_dir +'t75losses'+'.png',bbox_inches='tight')
    # plt.close()

    epochs = range(args.start_epoch, args.max_epochs)
    plt.plot(epochs, train_loss_arr, '-g', label='Training loss')
    plt.plot(epochs, val_loss_arr, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_dir+"losses.png")
    plt.close()