from pathlib import Path

import imageio
import numpy as np
import math
from collections import OrderedDict

import torch
from torch.utils.data import Dataset

from utils import make_tuple
from PIL import Image


root_dir = Path(__file__).parents[1]
data_dir = root_dir /'edcstfn_data/DX/data'    #存放数据的目录
pretrained = Path(__file__).parents[0] /'pretrained.pth'

REF_PREFIX_1 = '00'
PRE_PREFIX = '01'
REF_PREFIX_2 = '02'
COARSE_PREFIX = 'M'
FINE_PREFIX = 'L'
SCALE_FACTOR = 1
pixel_value_scale = 1


def get_pair_path(im_dir, n_refs):
    # 将一组数据集按照规定的顺序组织好
    paths = []
    order = OrderedDict()   #集合  时间_数据，例如，00_MOD09A1、01_LC08。ordervalues：odict_values(['00_MOD09A1', '00_LC08', '01_MOD09A1', '01_LC08'])
    order[0] = REF_PREFIX_1 + '_' + COARSE_PREFIX
    order[1] = REF_PREFIX_1 + '_' + FINE_PREFIX
    order[2] = PRE_PREFIX + '_' + COARSE_PREFIX
    order[3] = PRE_PREFIX + '_' + FINE_PREFIX

    if n_refs == 2:
        order[2] = REF_PREFIX_2 + '_' + COARSE_PREFIX
        order[3] = REF_PREFIX_2 + '_' + FINE_PREFIX
        order[4] = PRE_PREFIX + '_' + COARSE_PREFIX
        order[5] = PRE_PREFIX + '_' + FINE_PREFIX

    for prefix in order.values():
        for path in Path(im_dir).glob('*.tif'):   #
            if path.name.startswith(prefix):     #检测字符串是否以指定的前缀开始。
                paths.append(path.expanduser().resolve())
                # 这个break非常有必要，他防止一个文件夹中有第二个代表同一时间同一个卫星的图像出现。此时我也联想到存放数据的文件夹也许可以用时间命名。
                break
    if n_refs == 2:
        assert len(paths) == 6 or len(paths) == 5
    else:
        assert len(paths) == 3 or len(paths) == 4
    return paths

def get_image_pair(patches,patch_size,n_refs):
    LR, LR_sr, HR, Ref_1, Ref_sr_1, Ref_2, Ref_sr_2 = None, None, None, None, None, None, None
    if n_refs == 1:
        HR = patches[3]
        HR = np.array(HR)
        h, w = HR.shape[:2]

        ### LR and LR_sr,LR就是HR的降采样，LR_sr就是降采样之后的上采样
        LR_sr = patches[6]

        ###把LR传进去
        LR = patches[2]
        LR = np.array(LR)

        ### Ref and Ref_sr
        Ref_sub_1 = patches[1]
        Ref_sub_1 = np.array(Ref_sub_1)

        h2, w2 = Ref_sub_1.shape[:2]
        Ref_sr_sub_1 = np.array(Image.fromarray(Ref_sub_1).resize((w2 // 1, h2 // 1), Image.BICUBIC))
        Ref_sr_sub_1 = np.array(Image.fromarray(Ref_sr_sub_1).resize((w2, h2), Image.BICUBIC))
        ### complete ref and ref_sr to the same size, to use batch_size > 1
        Ref_1 = np.zeros((patch_size, patch_size, 4))
        Ref_sr_1 = np.zeros((patch_size, patch_size, 4))
        Ref_1[:h2, :w2, :] = Ref_sub_1
        Ref_sr_1[:h2, :w2, :] = Ref_sr_sub_1
        Ref_2 = np.zeros((patch_size, patch_size, 4))
        Ref_sr_2 = np.zeros((patch_size, patch_size, 4))
    elif n_refs == 2:
        HR = patches[5]
        HR = np.array(HR)
        h, w = HR.shape[:2]

        ### LR and LR_sr,LR就是HR的降采样，LR_sr就是降采样之后的上采样
        LR = np.array(Image.fromarray(HR).resize((w // 4, h // 4), Image.BICUBIC))
        LR_sr = patches[6]


        ###把LR传进去
        LR = patches[4]
        LR = np.array(LR)

        ### Ref and Ref_sr
        Ref_sub_1 = patches[1]
        Ref_sub_1 = np.array(Ref_sub_1)

        h2, w2 = Ref_sub_1.shape[:2]
        Ref_sr_sub_1 = np.array(Image.fromarray(Ref_sub_1).resize((w2 // 1, h2 // 1), Image.BICUBIC))
        Ref_sr_sub_1 = np.array(Image.fromarray(Ref_sr_sub_1).resize((w2, h2), Image.BICUBIC))
        ### complete ref and ref_sr to the same size, to use batch_size > 1
        Ref_1 = np.zeros((patch_size, patch_size, 4))
        Ref_sr_1 = np.zeros((patch_size, patch_size, 4))
        Ref_1[:h2, :w2, :] = Ref_sub_1
        Ref_sr_1[:h2, :w2, :] = Ref_sr_sub_1

        ### Ref and Ref_sr
        Ref_sub_2 = patches[3]
        Ref_sub_2 = np.array(Ref_sub_2)

        h2, w2 = Ref_sub_2.shape[:2]
        Ref_sr_sub_2 = np.array(Image.fromarray(Ref_sub_2).resize((w2 // 1, h2 // 1), Image.BICUBIC))
        Ref_sr_sub_2 = np.array(Image.fromarray(Ref_sr_sub_2).resize((w2, h2), Image.BICUBIC))
        ### complete ref and ref_sr to the same size, to use batch_size > 1
        Ref_2 = np.zeros((patch_size, patch_size, 4))
        Ref_sr_2 = np.zeros((patch_size, patch_size, 4))
        Ref_2[:h2, :w2, :] = Ref_sub_2
        Ref_sr_2[:h2, :w2, :] = Ref_sr_sub_2

    ### change type
    LR = LR.astype(np.float32)
    LR_sr = LR_sr.astype(np.float32)
    HR = HR.astype(np.float32)
    Ref_1 = Ref_1.astype(np.float32)
    Ref_sr_1 = Ref_sr_1.astype(np.float32)
    Ref_2 = Ref_2.astype(np.float32)
    Ref_sr_2 = Ref_sr_2.astype(np.float32)
    sample = {'LR': LR,
              'LR_sr': LR_sr,
              'HR': HR,
              'Ref_1': Ref_1,
              'Ref_sr_1': Ref_sr_1,
              'Ref_2': Ref_2,
              'Ref_sr_2': Ref_sr_2}

    return sample


def load_image_pair(im_dir,patch_padding, n_refs):
    # 按照一定顺序获取给定文件夹下的一组数据
    paths = get_pair_path(im_dir, n_refs)
    refs = [p for p in (data_dir / 'refs').glob('*.tif')]
    ref = refs[np.random.randint(0, len(refs))]
    paths.append(ref)
    images = []
    for p in paths:
        image = imageio.imread(p)
        # im = image.astype(np.float32)  # H*W*C (numpy.ndarray)
        im = image.astype(np.uint8)  # H*W*C (numpy.ndarray)
        image = np.zeros((im.shape[0] + patch_padding[0] * 2, im.shape[1] + patch_padding[1] * 2, im.shape[2])).astype(
            np.uint8)
        image[patch_padding[0]:im.shape[0] + patch_padding[0], patch_padding[1]:im.shape[1] + patch_padding[1], :] = im
        images.append(image[:,:,0:4])
    # 对数据的尺寸进行验证,暂时取消
    assert images[0].shape[0] * SCALE_FACTOR == images[1].shape[0]
    assert images[0].shape[1] * SCALE_FACTOR == images[1].shape[1]
    return images

def im2tensor(sample):
    LR, LR_sr, HR, Ref_1, Ref_sr_1, Ref_2, Ref_sr_2 = sample['LR'], sample['LR_sr'], sample['HR'], sample['Ref_1'], \
                                                      sample['Ref_sr_1'], sample['Ref_2'], sample['Ref_sr_2']
    LR = LR.transpose((2, 0, 1))
    LR_sr = LR_sr.transpose((2, 0, 1))
    HR = HR.transpose((2, 0, 1))
    Ref_1 = Ref_1.transpose((2, 0, 1))
    Ref_sr_1 = Ref_sr_1.transpose((2, 0, 1))
    Ref_2 = Ref_2.transpose((2, 0, 1))
    Ref_sr_2 = Ref_sr_2.transpose((2, 0, 1))

    LR = LR / pixel_value_scale
    LR_sr = LR_sr / pixel_value_scale
    HR = HR / pixel_value_scale
    Ref_1 = Ref_1 / pixel_value_scale
    Ref_sr_1 = Ref_sr_1 / pixel_value_scale
    Ref_2 = Ref_2 / pixel_value_scale
    Ref_sr_2 = Ref_sr_2 / pixel_value_scale



    return {'LR': torch.from_numpy(LR).float(),
            'LR_sr': torch.from_numpy(LR_sr).float(),
            'HR': torch.from_numpy(HR).float(),
            'Ref_1': torch.from_numpy(Ref_1).float(),
            'Ref_sr_1': torch.from_numpy(Ref_sr_1).float(),
            'Ref_2': torch.from_numpy(Ref_2).float(),
            'Ref_sr_2': torch.from_numpy(Ref_sr_2).float()
            }


class PatchSet(Dataset):
    """
    每张图片分割成小块进行加载
    Pillow中的Image是列优先，而Numpy中的ndarray是行优先
    """
    def __init__(self, image_dir, image_size, patch_size, patch_stride=None,patch_padding=None, n_refs=1):
        super(PatchSet, self).__init__()
        patch_size = make_tuple(patch_size)
        patch_padding = make_tuple(patch_padding)

        if not patch_stride:
            patch_stride = patch_size
        else:
            patch_stride = make_tuple(patch_stride)
        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding

        self.refs = n_refs  #需要几对参考图像

        self.image_dirs = [p for p in self.root_dir.glob('*') if p.is_dir()]    #train下有多少个文件夹

        self.num_im_pairs = len(self.image_dirs)

        # 计算出图像进行分块以后的patches的数目，ceil向上取整。这个过程就像卷积层与卷积核一样。
        self.num_patches_x = math.ceil((image_size[0] + patch_padding[0] - patch_size[0] + 1) / patch_stride[0])
        self.num_patches_y = math.ceil((image_size[1] + patch_padding[0] - patch_size[1] + 1) / patch_stride[1])
        self.num_patches = self.num_im_pairs * self.num_patches_x * self.num_patches_y

        self.transform = im2tensor

    def map_index(self, index):
        id_n = index // (self.num_patches_x * self.num_patches_y)   #整数除法返回向下取整后的结果，代表的是第几个数据组。可以有很多组数据。
        residual = index % (self.num_patches_x * self.num_patches_y)    #返回除法的余数，代表的是数据组的第几个块。
        id_x = self.patch_stride[0] * (residual % self.num_patches_x)   #代表移动了多少像素，每多一个块就要移动5个像素。
        id_y = self.patch_stride[1] * (residual // self.num_patches_x)
        return id_n, id_x, id_y

    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)
        images = load_image_pair(self.image_dirs[id_n],self.patch_padding, self.refs)
        patches = [None] * len(images)

        # 运行完下列代码，不是缩小了30倍，而是只取一张图像里面的一部分，而这一部分的大小刚好是原来的三十分之一。
        scales = [1, SCALE_FACTOR]
        for i in range(len(patches)):
            scale = scales[i % 2]
            im = images[i][
                 id_x * scale:(id_x + self.patch_size[0]) * scale + self.patch_padding[0]*2,
                 id_y * scale:(id_y + self.patch_size[1]) * scale + self.patch_padding[1]*2,:]     #a:b代表从第a个元素到第b个元素的意思。所以b-a等于10（30）。
            patches[i] = im
        del images[:]
        del images

        sample = get_image_pair(patches,self.patch_size[0]+self.patch_padding[0]*2,self.refs)
        sample = self.transform(sample)

        return sample

    def __len__(self):      #train_loader的长度等于这个长度除以train_loader的batch_size
        return self.num_patches
