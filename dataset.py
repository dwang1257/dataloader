import os
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

class customDataset(Dataset):
    
    def __init__(self, root_dir, folder_type = "training", transform = None, target_transform = None):
        self.transform = transform
        self.target_transform = target_transform
        self.folder_type = folder_type
        
        self.img_dir = os.path.join(root_dir, 'img', folder_type)
        self.mask_dir = os.path.join(root_dir, 'label', folder_type)
        
        self.img_files = sorted(os.listdir(self.img_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.img_files[idx])
        
    
        image = Image.open(img_path)
        mask = Image.open(mask_path)
    
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
    
        return image, mask, img_path, mask_path

img_transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

mask_transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.ToTensor() 
    ])


def visualize_sample(image, mask, path, mask_path):
    image = F.to_pil_image(image)
    mask = F.to_pil_image(mask)
    
    fig, ax = plt.subplots(1, 2, figsize = (10,5))
    plt.title(path)
    plt.title(mask_path)
    
    ax[0].imshow(image)
    
    ax[1].imshow(mask)
    plt.show()
    
if __name__ == '__main__':
    root_dir = '/Users/dylan/Desktop/masks'

    dataset = customDataset(root_dir, transform=img_transforms, target_transform=mask_transforms)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for idx, (images, masks, img_path, mask_path) in enumerate(dataloader):
        
        if idx < 5: 
            for i in range(images.size(0)):
                visualize_sample(images[i], masks[i], img_path[i], mask_path[i])
                break
        else:
            break
    
    #revert the normalization for the image
