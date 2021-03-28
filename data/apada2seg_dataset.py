import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
import data.random_pair as random_pair
import pdb


class TrainDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        self.dir_A = opt.raw_A_dir
        self.dir_B = opt.raw_B_dir
        self.dir_Seg = opt.raw_A_seg_dir

        self.A_paths = opt.imglist_A
        self.B_paths = opt.imglist_B

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        if not self.opt.isTrain:
            self.skipcrop = True
            self.skiprotate = True
        else:
            self.skipcrop = False
            self.skiprotate = False

        if self.skipcrop:
            osize = [opt.fineSize, opt.fineSize]
        else:
            osize = [opt.loadSize, opt.loadSize]

        if self.skiprotate:
            angle = 0
        else:
            angle = opt.angle

        transform_list = []
        transform_list.append(random_pair.randomrotate_pair(angle))    # scale the image
        self.transforms_rotate = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))    # scale the image
        self.transforms_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Scale(osize, Image.NEAREST))    # scale the segmentation
        self.transforms_seg_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(random_pair.randomcrop_pair(opt.fineSize))    # random crop image & segmentation
        self.transforms_crop = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Normalize([0.5], [0.5]))
        self.transforms_normalize = transforms.Compose(transform_list)

    def __getitem__(self, index):
        index_A = index % self.A_size
        A_path = os.path.join(self.dir_A, self.A_paths[index_A])
        Seg_path = os.path.join(self.dir_A, self.A_paths[index_A])
        Seg_path = Seg_path.replace('.png', '_mask.png')

        index_B = random.randint(0, self.B_size - 1)
        B_path = os.path.join(self.dir_B, self.B_paths[index_B])

        A_img = Image.open(A_path).convert('L')
        Seg_img = Image.open(Seg_path).convert('I')
        B_img = Image.open(B_path).convert('L')

        A_img = self.transforms_scale(A_img)
        B_img = self.transforms_scale(B_img)
        Seg_img = self.transforms_seg_scale(Seg_img)

        if not self.skiprotate:
            [A_img, Seg_img] = self.transforms_rotate([A_img, Seg_img])
            [B_img] = self.transforms_rotate([B_img])

        if not self.skipcrop:
            [A_img, Seg_img] = self.transforms_crop([A_img, Seg_img])
            [B_img] = self.transforms_crop([B_img])

        A_img = self.transforms_toTensor(A_img)
        B_img = self.transforms_toTensor(B_img)
        Seg_img = self.transforms_toTensor(Seg_img)

        A_img = self.transforms_normalize(A_img)
        B_img = self.transforms_normalize(B_img)

        Seg_img[Seg_img > 0] = 1

        Seg_imgs = torch.Tensor(self.opt.output_nc_seg, self.opt.fineSize, self.opt.fineSize)
        Seg_imgs[0, :, :] = Seg_img == 0
        Seg_imgs[1, :, :] = Seg_img == 1

        return {'A': A_img, 'B': B_img, 'Seg': Seg_imgs, 'Seg_one': Seg_img,
                'A_paths': A_path, 'B_paths': B_path, 'Seg_paths': Seg_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
