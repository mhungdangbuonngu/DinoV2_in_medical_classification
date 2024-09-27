import torch
from sklearn.decomposition import PCA
import os
from PIL import Image
from torchvision import transforms
from model import get_available_device
import matplotlib.pyplot as plt

device = get_available_device()

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14 = dinov2_vits14.to(device)

patch_size = 14
IMG_SIZE = 1024

patch_h  = IMG_SIZE//patch_size
patch_w  = IMG_SIZE//patch_size
feat_dim = 384 # vitl14

pre_process = transforms.Compose([
    transforms.Resize(IMG_SIZE), #just in case 
    transforms.CenterCrop(1008), #72 * patch size
    transforms.ToTensor(),
    transforms.Normalize(mean=0.52, std=0.23) # data set specific
])

total_features  = []
ROOT_PATH = '/mnt/g/Code/Dataset/archive/images_001/images'

img_list = [os.path.join(ROOT_PATH, f'0000000{i}_000.png') for i in range(1,5)]

with torch.no_grad():
  for img_path in img_list:
    img = Image.open(img_path).convert('RGB')
    img_t = pre_process(img).to(device)
    
    features_dict = dinov2_vits14.forward_features(img_t.unsqueeze(0))
    features = features_dict['x_norm_patchtokens']
    total_features.append(features)

total_features = torch.cat(total_features, dim=0)

# First PCA to Seperate Background
# sklearn expects 2d array for traning
total_features = total_features.reshape(4 * patch_h * patch_w, feat_dim) #4(*H*w, 1024)
total_features = total_features.cpu()

pca = PCA(n_components=3)
pca.fit(total_features)
pca_features = pca.transform(total_features)

# visualize PCA components for finding a proper threshold
# 3 histograms for 3 components
plt.subplot(2, 2, 1)
plt.hist(pca_features[:, 0])
plt.subplot(2, 2, 2)
plt.hist(pca_features[:, 1])
plt.subplot(2, 2, 3)
plt.hist(pca_features[:, 2])
plt.savefig("pca_images/features_hist.png")
# plt.show()
# plt.close()