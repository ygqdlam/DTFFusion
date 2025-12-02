import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from model.Loss import *
from data import PatchSet, get_pair_path, SCALE_FACTOR
import utils

from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import pandas as pd
import shutil
import sys
from tensorboardX import SummaryWriter

import metrics
from model.TTSR import *
from ttsr_utils import calc_psnr_and_ssim


class Experiment(object):
    def __init__(self, option):
        self.args = option
        self.device = torch.device('cuda:0' if option.cuda else 'cpu')
        self.resolution_scale = SCALE_FACTOR       #细图像与粗图像之间的比例
        self.image_size = option.image_size
        self.epochs = option.epochs

        self.save_dir = option.save_dir     #保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.save_dir / 'train'       #项目根目录生成的train
        self.train_dir.mkdir(exist_ok=True)
        self.history = self.train_dir / 'history.csv'   #项目根目录里train文件夹下
        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)
        self.checkpoint = self.train_dir / 'last.pth'
        self.epoch_checkpoint = self.train_dir
        self.best = self.train_dir / 'best.pth'

        self.logger = utils.get_logger("train")    #Console日志定义,只要有这个类就行，不需要管这个方法是干什么的
        self.logger.info('Model initialization')
        self.logger.info(f'lr: {option.lr} ,epochs:{option.epochs},batch_size:{option.batch_size},patch_size:{option.patch_size},patch_stride:{option.patch_stride},test_patch:{option.test_patch},ref:{option.refs}')
        self.model = TTSR(option).to(self.device)    #网络定义
        self.pretrained = Pretrained().to(self.device)      #特征损失？
        # utils.load_pretrained(self.pretrained, option.pretrained)
        # if option.cuda and option.ngpu > 1:
        #     device_ids = [i for i in range(option.ngpu)]
        #     self.model = nn.DataParallel(self.model, device_ids=device_ids)
        #     self.pretrained = nn.DataParallel(self.pretrained, device_ids=device_ids)

        self.criterion = CompoundLoss(self.device,self.pretrained)      #损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=option.lr, weight_decay=1e-6)      #优化器

        self.logger.info(str(self.model))
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)   #n_params参数总数，requires_grad判断输入是否需要保存梯度
        self.logger.info(f'There are {n_params} trainable parameters.')

        self.writer = SummaryWriter("logs/train")   #可视化

        self.advloss = AdversarialLoss(self.device)      #生成式对抗网络的损失函数

        self.metrics = {
            'psnr': metrics.psnr,
            'ssim': metrics.ssim,
            'sam': metrics.sam,
            'ergas': metrics.ergas,
            'cc': metrics.compare_corr,
            'rmse': metrics.RMSE,
        }
        #打印pdf记录
        self.train_metric = {
            'loss': [],
            'mse': [],
            'psnr': [],
            'ssim': [],
            'sam': [],
            'ergas': [],
            'cc': [],
            'rmse': [],
        }
        self.val_metric = {
            'loss': [],
            'mse': [],
            'psnr': [],
            'ssim': [],
            'sam': [],
            'ergas': [],
            'cc': [],
            'rmse': [],

        }
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched


    def train_on_epoch(self, n_epoch, data_loader,patch_size =None, patch_padding=None):
        self.model.train()
        epoch_loss = utils.AverageMeter()
        epoch_error = utils.AverageMeter()

        batches = len(data_loader)      #数据集里有59*59个块，data_loader的batch_size是32.
        self.logger.info(f'Epoch[{n_epoch}/{self.epochs}] - {datetime.now()}')
        for idx, data in enumerate(data_loader):    #每个数据里有6个图片的张量
            self.optimizer.zero_grad()

            t_start = timer()
            sample_batched = self.prepare(data)  # 加载到GPU设备上。
            lr = sample_batched['LR']
            lr_sr = sample_batched['LR_sr']
            hr = sample_batched['HR']
            ref_1 = sample_batched['Ref_1']
            ref_sr_1 = sample_batched['Ref_sr_1']
            ref_2 = sample_batched['Ref_2']
            ref_sr_2 = sample_batched['Ref_sr_2']
            sr_1, sr_2, S, T_lv3, T_lv2, T_lv1 = self.model(lr=lr, lrsr=lr_sr, ref_1=ref_1, refsr_1=ref_sr_1,
                                                            ref_2=ref_2, refsr_2=ref_sr_2)
            hr = hr[:, :, patch_padding[0]:patch_size[0] + patch_padding[0],
                         patch_padding[1]:patch_size[1] + patch_padding[1]]
            for i in range(len(sr_1)):
                sr_1[i] = sr_1[i][:, :, patch_padding[0]:patch_size[0] + patch_padding[0],
                     patch_padding[1]:patch_size[1] + patch_padding[1]]
            for i in range(len(sr_2)):
                sr_2[i] = sr_2[i][:, :, patch_padding[0]:patch_size[0] + patch_padding[0],
                     patch_padding[1]:patch_size[1] + patch_padding[1]]

            rec_loss = 0
            if (self.args.refs == 1):
                for i in range(len(sr_1)):
                    loss1 = self.criterion(sr_1[i], hr)
                    rec_loss = loss1 + rec_loss
            elif (self.args.refs == 2):
                for i in range(len(sr_1)):
                    rec_1_loss = self.criterion(sr_1[i], hr)
                    rec_2_loss = self.criterion(sr_2[i], hr)
                    rec_loss = rec_loss + rec_1_loss + rec_2_loss
            loss = rec_loss

            epoch_loss.update(loss.item())
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                rec_score = 0
                if (self.args.refs == 1):
                    for i in range(len(sr_1)):
                        mse = F.mse_loss(sr_1[i], hr)
                        rec_score = mse + rec_score
                elif (self.args.refs == 2):
                    for i in range(len(sr_1)):
                        rec_1_mse = F.mse_loss(sr_1[i], hr)
                        rec_2_mse = F.mse_loss(sr_2[i], hr)
                        rec_score = rec_score + rec_1_mse + rec_2_mse
                score = rec_score
            epoch_error.update(score.item())

            t_end = timer()

            # nn.utils.clip_grad_value_(self.model.parameters(),1)
            nn.utils.clip_grad_norm(self.model.parameters(), 1, norm_type=2)

            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         if param.grad is not None:
            #             print("{}, gradient: {}".format(name, param.grad.mean()))
            #         else:
            #             print("{} has not gradient".format(name))

            self.logger.info(f'Epoch[{n_epoch}/{self.epochs} {idx}/{batches}] - '
                             f'Loss: {loss.item():.10f} - '
                             f'MSE: {score.item():.5f} - '
                             f'batch_Time: {t_end - t_start}s')
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        return epoch_loss.avg, epoch_error.avg

    @torch.no_grad()
    def test_on_epoch(self, data_loader,patch_size =None, patch_padding=None):
        self.model.eval()
        epoch_loss = utils.AverageMeter()
        epoch_error = utils.AverageMeter()
        epoch_psnr = utils.AverageMeter()
        epoch_ssim = utils.AverageMeter()


        for data in data_loader:
            sample_batched = self.prepare(data)  # 加载到GPU设备上。
            lr = sample_batched['LR']
            lr_sr = sample_batched['LR_sr']
            hr = sample_batched['HR']
            ref_1 = sample_batched['Ref_1']
            ref_sr_1 = sample_batched['Ref_sr_1']
            ref_2 = sample_batched['Ref_2']
            ref_sr_2 = sample_batched['Ref_sr_2']
            sr, _, _, _, _ = self.model(lr=lr, lrsr=lr_sr, ref_1=ref_1, refsr_1=ref_sr_1, ref_2=ref_2, refsr_2=ref_sr_2,
                                        val=True)
            sr = sr[:, :, patch_padding[0]:patch_size[0] + patch_padding[0],
                         patch_padding[1]:patch_size[1] + patch_padding[1]]
            hr = hr[:, :, patch_padding[0]:patch_size[0] + patch_padding[0],
                     patch_padding[1]:patch_size[1] + patch_padding[1]]

            loss = self.criterion(sr, hr)
            epoch_loss.update(loss.item())
            score = F.mse_loss(sr, hr)
            epoch_error.update(score.item())
            for i in range(sr.shape[0]):
                psnr, ssim = calc_psnr_and_ssim(sr[i].detach(),hr[i].detach())
                epoch_psnr.update(psnr)
                epoch_ssim.update(ssim)

        utils.save_checkpoint(self.model, self.optimizer, self.checkpoint)      #保存参数模型
        return epoch_loss.avg, epoch_error.avg,epoch_psnr.avg,epoch_ssim.avg


    def train(self, train_dir, val_dir, patch_size, patch_stride, batch_size,
              train_refs,patch_padding=0, num_workers=0, epochs=30, resume=True):
        self.logger.info('Loading data...')
        patch_padding = utils.make_tuple(patch_padding)
        patch_size = utils.make_tuple(patch_size)


        train_set = PatchSet(train_dir, self.image_size, patch_size, patch_stride, patch_padding=patch_padding,
                             n_refs=train_refs)
        val_set = PatchSet(val_dir, self.image_size, patch_size,  patch_padding=patch_padding,n_refs=train_refs)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

        least_error = sys.maxsize
        start_epoch = 0
        if resume and self.checkpoint.exists(): #有预训练模型时会用到的代码
            utils.load_checkpoint(self.checkpoint, self.model, self.optimizer)
            if self.history.exists():
                df = pd.read_csv(self.history)
                least_error = df['val_error'].min()
                start_epoch = int(df.iloc[-1]['epoch']) + 1

        self.logger.info('Training...')
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=6)   #对学习率进行动态的调整（动态的下降）
        #scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=1)
        # for epoch in range(start_epoch, epochs + start_epoch):
        for epoch in range(start_epoch, epochs):

            for param_group in self.optimizer.param_groups:     #这一部分只是想打印下来看看当前的学习率降到什么程度了
                self.logger.info(f"Current learning rate: {param_group['lr']}")
            epoch_start = timer()
            train_loss, train_error = self.train_on_epoch(epoch, train_loader,patch_size = patch_size,patch_padding=patch_padding)  #训练

            val_loss, val_error,val_psnr,val_ssim = self.test_on_epoch(val_loader,patch_size = patch_size,patch_padding=patch_padding) #验证
            ### 保留每一个epoch的参数
            checkpoint_name = str(epoch)+'.pth'
            utils.save_checkpoint(self.model, self.optimizer, self.epoch_checkpoint / checkpoint_name)  # 保存参数模型

            epoch_end = timer()
            csv_header = ['epoch', 'train_loss', 'train_error', 'val_loss', 'val_error','val_psnr','val_ssim','epoch_time']
            csv_values = [epoch, train_loss, train_error, val_loss, val_error,val_psnr,val_ssim,epoch_end-epoch_start]
            utils.log_csv(self.history, csv_values, header=csv_header)
            self.logger.info(f'Epoch[{epoch}- '
                             f'train_loss: {train_loss:.10f} - '
                             f'train_error: {train_error:.10f} - '
                             f'val_loss: {val_loss:.10f} - '
                             f'val_error: {val_error:.10f} - '
                             f'val_psnr: {val_psnr:.10f} - '
                             f'val_ssim: {val_ssim:.10f} - '
                             f'epoch_Time: {epoch_end - epoch_start}s')

            self.writer.add_scalars('data/loss', {'train_loss':train_loss,
                                                  'val_loss':val_loss}, epoch)
            self.writer.add_scalars('data/error', {'train_error': train_error,
                                                  'val_error': val_error}, epoch)

            # #把指标输出到pdf
            # self.train_metric['loss'].append(train_loss)
            # self.train_metric['mse'].append(train_error)
            # self.val_metric['loss'].append(val_loss)
            # self.val_metric['mse'].append(val_error)
            # #此处加1是因为epoch是从0开始的
            # utils.plot_loss(self.train_metric['loss'],'trainloss',epoch+1)
            # utils.plot_loss(self.train_metric['mse'],'trainmse',epoch+1)
            # utils.plot_loss(self.val_metric['loss'],'valloss',epoch+1)
            # utils.plot_loss(self.val_metric['mse'],'valmse',epoch+1)
            utils.CSV_plot()    #把打印指标jpg的任务封装到该方法中，只需要csv文件就可。



            scheduler.step(val_loss)
            if val_error < least_error:     #如果错误率小于最小的错误率
                self.logger.info(f'Save Model in Epoch:{epoch}')
                shutil.copy(str(self.checkpoint), str(self.best))   #shutil.copyfile(src, dst)复制文件内容src到dst。如果没有预训练模型，那这段代码也执行，只不过没什么用。
                least_error = val_error




    @torch.no_grad()
    def test(self, test_dir, patch_size, test_refs,patch_padding = 0, num_workers=0):
        self.model.eval()
        patch_size = utils.make_tuple(patch_size)
        patch_padding = utils.make_tuple(patch_padding)


        utils.load_checkpoint(self.best, model=self.model)
        self.logger.info('Testing...')
        # 记录测试文件夹中的文件路径，用于最后投影信息的匹配
        image_dirs = [p for p in test_dir.glob('*') if p.is_dir()]
        image_paths = [get_pair_path(d, test_refs) for d in image_dirs]

        # 在预测阶段，对图像进行切块的时候必须刚好裁切完全，这样才能在预测结束后进行完整的拼接
        assert self.image_size[0] % patch_size[0] == 0
        assert self.image_size[1] % patch_size[1] == 0
        rows = int(self.image_size[1] / patch_size[1])
        cols = int(self.image_size[0] / patch_size[0])
        n_blocks = rows * cols
        test_set = PatchSet(test_dir, self.image_size, patch_size,patch_padding=patch_padding, n_refs=test_refs)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers)

        scaled_patch_size = tuple(i * self.resolution_scale for i in patch_size)
        scaled_image_size = tuple(i * self.resolution_scale for i in self.image_size)
        pixel_value_scale = 1
        im_count = 0
        patches = []
        t_start = datetime.now()

        for inputs in test_loader:
            # 如果包含了target数据，则去掉最后的target
            if len(inputs) % 2 == 0:
                del inputs[-1]
            name = image_paths[im_count][-1].name
            if len(patches) == 0:
                t_start = timer()
                self.logger.info(f'Predict on image {name}')

            # 分块进行预测（每次进入深度网络的都是影像中的一块）
            sample_batched = self.prepare(inputs)  # 加载到GPU设备上。
            lr = sample_batched['LR']
            lr_sr = sample_batched['LR_sr']
            hr = sample_batched['HR']
            ref_1 = sample_batched['Ref_1']
            ref_sr_1 = sample_batched['Ref_sr_1']
            ref_2 = sample_batched['Ref_2']
            ref_sr_2 = sample_batched['Ref_sr_2']
            sr, _, _, _, _ = self.model(lr=lr, lrsr=lr_sr, ref_1=ref_1, refsr_1=ref_sr_1, ref_2=ref_2, refsr_2=ref_sr_2,
                                        val=True)

            prediction = sr.cpu().numpy()
            patches.append(prediction * pixel_value_scale)

            # 完成一张影像以后进行拼接
            if len(patches) == n_blocks:
                result = np.empty((NUM_BANDS, *scaled_image_size), dtype=np.float32)
                block_count = 0
                for i in range(rows):
                    row_start = i * scaled_patch_size[1]
                    for j in range(cols):
                        col_start = j * scaled_patch_size[0]
                        result[:,
                        col_start: col_start + scaled_patch_size[0],
                        row_start: row_start + scaled_patch_size[1]
                        ] = patches[block_count][0, :, patch_padding[0]:patch_size[0] + patch_padding[0],
                            patch_padding[1]:patch_size[1] + patch_padding[1]]
                        block_count += 1
                patches.clear() #预测完了，此时result的结果是什么呢？
                # 存储预测影像结果
                # result = result.astype(np.int16)
                result = result.astype(np.uint8)
                prototype = str(image_paths[im_count][1])
                utils.save_array_as_tif(result, self.test_dir / name, prototype=prototype)
                im_count += 1
                t_end = timer()
                self.logger.info(f'Prediction Time cost: {t_end - t_start}s')




