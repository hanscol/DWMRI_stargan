from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import torch
import os
import random
import nibabel as nib
from DELIMIT.SphericalHarmonicTransformation import Signal2SH

class BLSA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, data_info, selected_attrs, crop_size, SH_convert, SH_order):
        """Initialize and preprocess the CelebA dataset."""

        self.selected_attrs = selected_attrs
        self.dataset = {}
        self.attr2idx = {}
        self.idx2attr = {}
        self.crop_size = crop_size
        self.parse_data(data_info)
        self.SH_convert = SH_convert
        self.SH_order = SH_order

        self.num_images = len(self.dataset)

    def randomCrop(self, img):
        assert img.shape[1] >= self.crop_size
        assert img.shape[2] >= self.crop_size
        assert img.shape[3] >= self.crop_size

        x = random.randint(0, img.shape[2] - self.crop_size)
        y = random.randint(0, img.shape[1] - self.crop_size)
        z = random.randint(0, img.shape[3] - self.crop_size)
        img = img[:, y:y + self.crop_size, x:x + self.crop_size, z:z + self.crop_size]

        return img

    def patch_copy(self, img):
        patched = np.zeros((img.shape[0], (img.shape[1]-2)*3, (img.shape[2]-2)*3, (img.shape[3]-2)*3))

        for x in range(1, img.shape[1]-1):
            for y in range(1, img.shape[2]-1):
                for z in range(1, img.shape[3]-1):
                    px = (x-1)*3+1
                    py = (y-1)*3+1
                    pz = (z-1)*3+1
                    patched[:, px-1:px+1, py-1:py+1, pz-1:pz+1] = img[:, x-1:x+1, y-1:y+1, z-1:z+1]

        return patched.astype(np.float32)

    def parse_data(self, data_info):
        with open(data_info) as f:
            lines = f.readlines()
            for l in lines:
                l = l.split(',')
                path = l[0].strip()
                scanner = int(l[1].strip())



                if scanner in self.dataset:
                    self.dataset[scanner].append(path)
                else:
                    self.dataset[scanner] = [path]

    
    def get_SH(self, image, order, gradients_path):
        b0 = np.expand_dims(image[:,:,:,0], axis=3)
        image = np.divide(image, b0)[:,:,:,1:].astype('float64')
        image[np.isnan(image)] = 0

        gradient_dirs = np.transpose(np.loadtxt(gradients_path))[1:,:].astype('float64')

        SH = Signal2SH(4, gradient_dirs)

        image = image.transpose((3, 0, 1, 2))
        image = SH.forward(torch.from_numpy(np.expand_dims(image, axis=0)))
        image = image.type(torch.FloatTensor).squeeze()

        return image


    def load_image(self, path, scanner):

        image = nib.load(path).get_fdata()
        
        if self.SH_convert:
            gradients_path = '/'.join(path.split('/')[0:-1]) + '/dwmri.bvec'
            image = self.get_SH(image, self.SH_order, gradients_path)
        else:
            image = image.transpose((3, 0, 1, 2)).astype('float32')

        # dim = max(image.shape)
        # for i in range(1,len(image.shape)):
        #     if image.shape[i] < dim:
        #         shape = []
        #         for j in range(0,len(image.shape)):
        #             if j != i:
        #                 shape.append(image.shape[j])
        #             else:
        #                 shape.append(dim - image.shape[j])
        #         image = np.concatenate((image, np.zeros(shape)), axis=i)



        label = np.zeros(len(self.selected_attrs))
        label[scanner] = 1
        label = label.astype('float32')

        return image, label

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if index % 20 == 0:
            self.scanner = random.randint(0,len(self.dataset.keys())-1)
            self.filename = self.dataset[self.scanner][random.randint(0,len(self.dataset[self.scanner])-1)]
            self.image, self.label = self.load_image(self.filename, self.scanner)
        patch = self.randomCrop(self.image)
        #copy_patch = self.patch_copy(patch)

        #return copy_patch, patch[:,1:patch.shape[1]-1, 1:patch.shape[2]-1, 1:patch.shape[3]-1], self.label, self.filename
        return patch, self.label, self.filename

    def __len__(self):
        """Return the number of images."""
        #return self.num_images
        return 1000000


def get_loader(data_info, selected_attrs, crop_size=48, image_size=128,
               batch_size=16, mode='train', num_workers=1, SH_convert=True, SH_order=4):
    """Build and return a data loader."""
    dataset = BLSA(data_info, selected_attrs, crop_size, SH_convert, SH_order)


    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=False)
    return data_loader
