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
IMG_SIZE = 224
img_data_path = r"img_data/"

pre_process = transforms.Compose([
    transforms.Resize(IMG_SIZE), #just in case 
    transforms.CenterCrop(224), #72 * patch size
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.2) # data set specific
])

def load_transform_save(img_path, npy_path):
    img = Image.open(img_path).convert('RGB')
    img_transformed = pre_process(img).to(device)
    features = dinov2_vits14(img_transformed.unsqueeze(0))
    features = dinov2_vits14.norm(features)
    numpy_feature = features.cpu().detach().numpy()
    np.save(npy_path, numpy_feature)
     
with torch.no_grad():
    #test 
    npy_data_path = r"npy_data/"
    img_data_path = r"img_data/"

    for train_test in os.listdir(img_data_path):
        train_test_path = os.path.join(img_data_path, train_test) 

        for yes_no in os.listdir(train_test_path):
            label = np.array([yes_no == "no", yes_no == "yes"]).astype(float)
            img_folder_path = os.path.join(train_test_path, yes_no)

            for img in os.listdir(img_folder_path):
                label_npy = f"{npy_data_path}/{train_test}/class_label/{img}.npy" 
                np.save(label_npy, label)

                img_file_path = os.path.join(img_folder_path, img)
                feature_npy = f"{npy_data_path}/{train_test}/img_feature/{img}.npy"
                
                load_transform_save(img_file_path, feature_npy)