from torchvision import datasets, transforms
import torchvision
import torch.utils.data as Data
import torch
import numpy as np
from PIL import Image
import os
import random


#whu
mean = [0.4352680, 0.445232, 0.413076]
std = [0.216836,0.203392,0.217332]

#inria
# mean = [0.31815762,0.32456695,0.29096074]
# std = [0.18410079,0.17732723,0.18069517]

class changeDatasets(Data.Dataset):
    def __init__(self, image1_dir, label1_dir, is_Transforms: True):
        print("PPSL")
        self.image1_dir = image1_dir
        self.label1_dir = label1_dir

        self.data_list = os.listdir(label1_dir)

        if is_Transforms:
            self.base_img = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(512, scale=(1, 1.2), ratio=(0.75, 1.33), interpolation=2),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomAffine(degrees=(-90, 90), translate=(0.1, 0.1)),          
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)
            ])

            self.base_label = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(512, scale=(1, 1.2), ratio=(0.75, 1.33), interpolation=0),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomAffine(degrees=(-90, 90), translate=(0.1, 0.1)),
                torchvision.transforms.ToTensor()
            ])

            self.colorjit = torchvision.transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2)
            self.affine = torchvision.transforms.RandomAffine(degrees=(-5, 5), scale=(1, 1.02),translate=(0.02, 0.02), shear=(-5, 5))

        else:
            self.tx1 = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512,512), interpolation=2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)
            ])


            self.lx = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512,512), interpolation=0),
                torchvision.transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image1_name = os.path.join(self.image1_dir, self.data_list[idx])
        image1 = Image.open(image1_name)

        label1_name = os.path.join(self.label1_dir, self.data_list[idx])
        label1 = Image.open(label1_name).convert("L")

        return image1, label1
    
    def be2bcd_sup(self, batchlist):
        
        image_t1 = []
        image_t2 = []
        label_t1 = []
        label_change = []
        img_patches = []
        lb_patches = []

        seed = np.random.randint(0, 2**16)

        for i in range(8):
            img_patches = self.cut_imgpatch(batchlist[i][0], img_patches)
            lb_patches = self.cut_lbpatch(batchlist[i][1], lb_patches)

        for batch in range(8, len(batchlist)):
            random.seed(seed+batch)
            torch.manual_seed(seed+batch)
            image1 = self.colorjit(batchlist[batch][0])

            random.seed(seed+batch)
            torch.manual_seed(seed+batch)
            image_t1.append(self.base_img(image1))

            image2 = self.img_pasted(self.affine(batchlist[batch][0]), img_patches[batch-8])
            
            random.seed(seed+batch)
            torch.manual_seed(seed+batch)
            image_t2.append(self.base_img(image2))

            random.seed(seed+batch)
            torch.manual_seed(seed+batch)
            label1 = self.base_label(batchlist[batch][1])
            label_t1.append(label1)

            label2 = self.lb_pasted(batchlist[batch][1], lb_patches[batch-8])

            random.seed(seed+batch)
            torch.manual_seed(seed+batch)
            label2 = self.base_label(label2)

            label_change.append(torch.logical_xor(label1, label2).type(torch.float32))


        return {'image1':torch.stack(image_t1), 'image2':torch.stack(image_t2), 'image1_label':torch.stack(label_t1), 'change_label':torch.stack(label_change)}
    

    def cut_imgpatch(self, image, patches):
        image = np.array(image)
        for i in range(2):
            patch = image[256*i: 256*(i+1), :, :]
            patches.append(patch)
        
        return patches
    
    def cut_lbpatch(self, image, patches):
        image = np.array(image)
        for i in range(2):
            patch = image[256*i: 256*(i+1),:]
            patches.append(patch)
        
        return patches
    
    def img_pasted(self, img, patch):
        img = np.array(img)
        img[256:512, :, :] = patch

        return Image.fromarray(img)
    
    def lb_pasted(self, img, patch):
        img = np.array(img)
        img[256:512, :] = patch
        
        return Image.fromarray(img)


class testDatasets(Data.Dataset):
    def __init__(self, image1_dir, image2_dir, label1_dir, is_Transforms: True):
        self.image1_dir = image1_dir
        self.image2_dir = image2_dir

        self.label1_dir = label1_dir

        self.data_list = os.listdir(label1_dir)

        self.tx1 = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512,512), interpolation=2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)
            ])

        self.tx2 = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512,512), interpolation=2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)
            ])

        self.lx = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512,512), interpolation=0),
                torchvision.transforms.ToTensor(),
            ])


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image1_name = os.path.join(self.image1_dir, self.data_list[idx])
        image1 = Image.open(image1_name)

        image2_name = os.path.join(self.image2_dir, self.data_list[idx])
        image2 = Image.open(image2_name)

        label1_name = os.path.join(self.label1_dir, self.data_list[idx])
        label1 = Image.open(label1_name).convert("L")

        image1 = self.tx1(image1)

        image2 = self.tx2(image2)

        label1 = self.lx(label1)

        return {'image1':image1, 'image2':image2, 'change':label1}

        
    

    

