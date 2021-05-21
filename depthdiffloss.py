import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        rows, cols = depth[0].shape
        c = torch.meshgrid(torch.arange(cols))
        new_c = c[0].reshape([1,cols]).to('cuda')
        r = torch.meshgrid(torch.arange(rows))
        new_r = r[0].unsqueeze(-1).to('cuda')
        valid = (depth > 0) & (depth < 1.01)
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

        all_real_pcd = self.point_cloud(real1[0]).clone() * 1000.0
        all_fake_pcd = self.point_cloud(fake1[0]).clone() * 1000.0
 
        for nr_img in range(1,real.shape[0]):
            real_pcd = self.point_cloud(real1[nr_img]).clone() * 1000.0
            fake_pcd = self.point_cloud(fake1[nr_img]).clone() * 1000.0

            all_real_pcd = torch.cat(all_real_pcd,real_pcd)
            all_fake_pcd = torch.cat(all_fake_pcd,fake_pcd)

        all_real_pcd[all_real_pcd==0] = eps
        all_fake_pcd[all_fake_pcd==0] = eps

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

        RMSE_log = torch.sqrt(torch.mean(torch.abs(torch.log(torch.abs(z_real))-torch.log(torch.abs(z_fake)))**2))

        loss = 10*(RMSE_log + torch.abs(10*(3-torch.exp(1*lossX)-torch.exp(1*lossY)-torch.exp(1*lossZ))))

        return loss