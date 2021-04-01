import torch.utils.data as data
import numpy as np
from PIL import Image
# from scipy.misc import imread
from path import Path
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor, RandomHorizontalFlip, CenterCrop, ColorJitter
import torch, time, os
import torch.nn.functional as F
import random
import scipy.ndimage as ndimage
from scipy import misc
import cv2

    
class NYUv2Dataset(data.Dataset):
    def __init__(self, root='/media/rambo/ssd2/Szilard/nyu_v2_filter/dataset/', seed=None, train=True):
    # def __init__(self, root='/media/rambo/ssd2/Szilard/pico_tofnest/1bag_augmented/dataset_filter/', seed=None, train=True):
        
        np.random.seed(seed)
        self.root = Path(root)
        self.train = train
        if train:
            self.rgb_paths = [root+'depth3/train/'+d for d in os.listdir(root+'depth3/train')]
            # Randomly choose 50k images without replacement
            # self.rgb_paths = np.random.choice(self.rgb_paths, 4000, False)
        else:
            self.rgb_paths = [root+'depth3/test/'+d for d in os.listdir(root+'depth3/test/')]
            # self.rgb_paths = np.random.choice(self.rgb_paths, 1000, False)
        

        # self.augmentation = Compose([RandomHorizontalFlip()]) # , RandomCropRotate(10)
        # self.rgb_transform = Compose([ToPILImage(), Resize((360,640)), ToTensor()]) # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
        # self.depth_transform = Compose([ToPILImage(), Resize((360,640)), ToTensor()])
        
        if self.train:
            self.length = len(self.rgb_paths)
        else:
            self.length = len(self.rgb_paths)
            
    def __getitem__(self, index):
        path = self.rgb_paths[index]

        # rgb = Image.open(path)
        # depth = Image.open(path.replace('depth3', 'depthgt'))
        # depth = np.array(depth)
        # depth = Compose([ToPILImage(), Resize((480,640)), ToTensor()])(depth)
        # rgb, depth = np.array(rgb), np.array(depth)
        # return self.rgb_transform(rgb), self.depth_transform(depth).squeeze(-1)

        depth_input = cv2.imread(path,cv2.IMREAD_UNCHANGED).astype(np.float32)
        if len(depth_input.shape) < 3:
            print("Got 1 channel depth images, creating 3 channel depth images")
            combine_depth = np.empty((depth_input.shape[0],depth_input.shape[1], 3))
            combine_depth[:,:,0] = depth_input
            combine_depth[:,:,1] = depth_input
            combine_depth[:,:,2] = depth_input
            depth_input = combine_depth
        depthgt = cv2.imread(path.replace('depth3', 'depthgt'),cv2.IMREAD_UNCHANGED ).astype(np.float32)
        depth_input_mod = np.moveaxis(depth_input,-1,0)
        depthgt2=np.expand_dims(depthgt, axis=0)
        # max_depth=10000.0/

        return depth_input_mod/np.max(depth_input_mod), depthgt2/np.max(depth_input_mod)
        # return depth_input_mod, depth

    def __len__(self):
        return self.length

if __name__ == '__main__':
    # Testing
    dataset = NYUv2Dataset()
    print(len(dataset))
    for item in dataset[0]:
        print(item.size())
