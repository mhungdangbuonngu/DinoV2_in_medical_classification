from torchvision import datasets, transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_image_datasets = {
    x: datasets.ImageFolder(r'brain_tumor_dataset/Training', data_transforms[x]) 
    for x in ['train', 'test']
}

test_image_datasets = {
    x: datasets.ImageFolder(r'brain_tumor_dataset/Testing', data_transforms[x]) 
    for x in ['train', 'test']
}