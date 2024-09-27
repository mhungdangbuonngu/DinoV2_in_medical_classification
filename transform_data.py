import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from os.path import join, exists
import os
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    
print(f'using {device}')

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14 = dinov2_vits14.to(device)

patch_size = dinov2_vits14.patch_size

# original image of size 1024x1024
IMG_SIZE = 1024 
dataset_path = "/mnt/g/Code/Dataset/archive"

pre_process = transforms.Compose([
    transforms.Resize(IMG_SIZE), #just in case 
    transforms.CenterCrop(1008), #72 * patch size
    transforms.ToTensor(),
    transforms.Normalize(mean=0.52, std=0.23) # data set specific
])

def load_transform_save(img_path, npy_path):
    img = Image.open(img_path).convert('RGB')
    img_transformed = pre_process(img).to(device)
    features = dinov2_vits14(img_transformed.unsqueeze(0))
    features = dinov2_vits14.norm(features)
    numpy_feature = features.cpu().detach().numpy()
    np.save(npy_path, numpy_feature)
     
with torch.no_grad():
    
    img_folders = ['images_001', 'images_002', 'images_003', 'images_004', 'images_005', 'images_006', 'images_007', 'images_008', 'images_009', 'images_010', 'images_011', 'images_012']

    # transform img for train folder
    train_npy_folder = "/mnt/g/Code/Dataset/archive/train/img_feature"

    for folder in img_folders[:-1]:
        folder_path = join(join(dataset_path,folder),"images")
        
        for img_name in tqdm(os.listdir(folder_path)):    
            img_path = join(folder_path,img_name)
            load_transform_save(img_path=img_path, npy_path=join(train_npy_folder,img_name))
    
    #transform img for test folder
    test_npy_folder = "/mnt/g/Code/Dataset/archive/test/img_feature"
    folder_path = join(join(dataset_path,img_folders[-1]),"images")
    
    for img_name in tqdm(os.listdir(folder_path)):
        img_path = join(folder_path,img_name)  
        load_transform_save(img_path=img_path, npy_path=join(test_npy_folder,img_name))