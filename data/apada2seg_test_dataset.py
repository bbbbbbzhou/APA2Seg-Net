import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
import data.random_pair as random_pair


class TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        self.dir_B = opt.test_B_dir

        self.B_filenames = opt.imglist_testB
        self.B_size = len(self.B_filenames)

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
        B_filename = self.B_filenames[index % self.B_size]
        B_path = os.path.join(self.dir_B, B_filename)
        B_img = Image.open(B_path).convert('L')
        B_img = self.transforms_scale(B_img)
        B_img = self.transforms_toTensor(B_img)
        B_img = self.transforms_normalize(B_img)

        Seg_filename = self.B_filenames[index % self.B_size]
        Seg_path = os.path.join(self.dir_B, Seg_filename)
        Seg_path = Seg_path.replace('.png', '_mask.png')
        Seg_img = Image.open(Seg_path).convert('I')
        Seg_img = self.transforms_toTensor(Seg_img)
        Seg_img[Seg_img > 0] = 1

        Seg_imgs = torch.Tensor(self.opt.output_nc_seg, self.opt.fineSize, self.opt.fineSize)
        Seg_imgs[0, :, :] = Seg_img == 0
        Seg_imgs[1, :, :] = Seg_img == 1

        return {'B': B_img, 'Seg': Seg_imgs,
                'B_paths': B_path, 'Seg_paths': Seg_path}

    def __len__(self):
        return self.B_size

    def name(self):
        return 'TestCTDataset'
