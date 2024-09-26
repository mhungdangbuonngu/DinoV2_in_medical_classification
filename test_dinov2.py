import torch
from PIL import Image
from torchvision import datasets, models, transforms
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    
print(f'using {device}')

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14 = dinov2_vits14.to(device)

pca = PCA(n_components=3)
scaler = MinMaxScaler(clip=True)
patch_size = dinov2_vits14.patch_size

# original image of size 1024x1024
IMG_SIZE = 1024 
#1024//14 (floor division)
patch_h  = IMG_SIZE//patch_size
patch_w  = IMG_SIZE//patch_size
feat_dim = 384 # vitl14

img_path = "/mnt/g/Code/Dataset/archive/images_001/images/00000001_000.png"

pre_process = transforms.Compose([
    transforms.Resize(IMG_SIZE), #just in case 
    transforms.CenterCrop(1008), #72 * batch size
    transforms.ToTensor(),
    transforms.Normalize(mean=0.52, std=0.23) # data set specific
])

with torch.no_grad():
    img = Image.open(img_path).convert('RGB')
    img_transformed = pre_process(img).to(device)
    features_dict = dinov2_vits14.forward_features(img_transformed.unsqueeze(0))
    features = features_dict['x_norm_patchtokens']
    # total_features.append(features)

# print(features)
# features.to(torch.device('cpu'))
numpy_feature = features.cpu().detach().numpy()
np.save("test.npy",numpy_feature)