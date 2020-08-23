from fmri.models.supervised.cnn3d import ConvResnet3D
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from scipy.stats import norm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from fmri.models.utils.distributions import log_gaussian
from fmri.utils.activations import Swish, Mish
from fmri.utils.CycleAnnealScheduler import CycleScheduler
from fmri.utils.dataset import load_checkpoint, save_checkpoint, MRIDatasetClassifier, CTDataset
from fmri.utils.transform_3d import Normalize, RandomRotation3D, ColorJitter3D, Flip90, Flip180, Flip270, XFlip, YFlip, \
    ZFlip, RandomAffine3D
from fmri.models.supervised.resnetcnn3d import ConvResnet3D
from fmri.utils.plot_performance import plot_performance
import torchvision
from torchvision import transforms
from ax.service.managed_loop import optimize
import random

import os

import nibabel as nib
from fmri.utils.utils import validation_spliter

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = "cpu"


class Predict:
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 strides,
                 dilatations,
                 padding,
                 path,
                 n_classes,
                 init_func=torch.nn.init.kaiming_uniform_,
                 activation=torch.nn.GELU,
                 batch_size=8,
                 epochs=1000,
                 fp16_run=False,
                 checkpoint_path=None,
                 epochs_per_checkpoint=1,
                 epochs_per_print=1,
                 gated=True,
                 has_dense=True,
                 batchnorm=False,
                 resblocks=False,
                 maxpool=3,
                 verbose=2,
                 size=32,
                 mean=0.5,
                 std=0.5,
                 plot_perform=True,
                 val_share=0.1,
                 cross_validation=5,
                 is_bayesian=True,
                 random_node='output'
                 ):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.dilatations = dilatations
        self.padding = padding
        self.batch_size = batch_size
        self.epochs = epochs
        self.fp16_run = fp16_run
        self.checkpoint_path = checkpoint_path
        self.epochs_per_checkpoint = epochs_per_checkpoint
        self.epochs_per_print = epochs_per_print
        self.gated = gated
        self.has_dense = has_dense
        self.batchnorm = batchnorm
        self.resblocks = resblocks
        self.maxpool = maxpool
        self.verbose = verbose
        self.path = path
        self.size = size
        self.std = std
        self.mean = mean
        self.activation = activation
        self.init_func = init_func
        self.val_share = val_share
        self.plot_perform = plot_perform
        self.cross_validation = cross_validation
        self.is_bayesian = is_bayesian

    def predict(self, params):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        mom_range = params['mom_range']
        n_res = params['n_res']
        niter = params['niter']
        scheduler = params['scheduler']
        optimizer_type = params['optimizer']
        momentum = params['momentum']
        learning_rate = params['learning_rate'].__format__('e')
        weight_decay = params['weight_decay'].__format__('e')

        weight_decay = float(str(weight_decay)[:1] + str(weight_decay)[-4:])
        learning_rate = float(str(learning_rate)[:1] + str(learning_rate)[-4:])
        if self.verbose > 1:
            print("Parameters: \n\t",
                  'zdim: ' + str(self.n_classes) + "\n\t",
                  'mom_range: ' + str(mom_range) + "\n\t",
                  'niter: ' + str(niter) + "\n\t",
                  'nres: ' + str(n_res) + "\n\t",
                  'learning_rate: ' + learning_rate.__format__('e') + "\n\t",
                  'momentum: ' + str(momentum) + "\n\t",
                  'weight_decay: ' + weight_decay.__format__('e') + "\n\t",
                  'optimizer_type: ' + optimizer_type + "\n\t",
                  )

        self.modelname = "classif_3dcnn_" \
                         + '_bn' + str(self.batchnorm) \
                         + '_niter' + str(niter) \
                         + '_nres' + str(n_res) \
                         + '_momrange' + str(mom_range) \
                         + '_momentum' + str(momentum) \
                         + '_' + str(optimizer_type) \
                         + "_nclasses" + str(self.n_classes) \
                         + '_gated' + str(self.gated) \
                         + '_resblocks' + str(self.resblocks) \
                         + '_initlr' + learning_rate.__format__('e') \
                         + '_wd' + weight_decay.__format__('e') \
                         + '_size' + str(self.size)
        model = ConvResnet3D(self.maxpool,
                             self.in_channels,
                             self.out_channels,
                             self.kernel_sizes,
                             self.strides,
                             self.dilatations,
                             self.padding,
                             self.batchnorm,
                             self.n_classes,
                             is_bayesian=self.is_bayesian,
                             activation=torch.nn.ReLU,
                             n_res=n_res,
                             gated=self.gated,
                             has_dense=self.has_dense,
                             resblocks=self.resblocks,
                             ).to(device)
        l1 = nn.L1Loss()
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(params=model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay,
                                          amsgrad=True)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(params=model.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay,
                                        momentum=momentum)
        elif optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(params=model.parameters(),
                                            lr=learning_rate,
                                            weight_decay=weight_decay,
                                            momentum=momentum)
        else:
            exit('error: no such optimizer type available')
        # if self.fp16_run:
        #     from apex import amp
        #    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

        # Load checkpoint if one exists
        epoch = 0
        best_loss = -1
        model, optimizer, \
        epoch, losses, \
        kl_divs, losses_recon, \
        best_loss = load_checkpoint(checkpoint_path,
                                    model,
                                    self.maxpool,
                                    save=False,
                                    padding=self.padding,
                                    has_dense=self.has_dense,
                                    batchnorm=self.batchnorm,
                                    flow_type=None,
                                    padding_deconv=None,
                                    optimizer=optimizer,
                                    z_dim=self.n_classes,
                                    gated=self.gated,
                                    in_channels=self.in_channels,
                                    out_channels=self.out_channels,
                                    kernel_sizes=self.kernel_sizes,
                                    kernel_sizes_deconv=None,
                                    strides=self.strides,
                                    strides_deconv=None,
                                    dilatations=self.dilatations,
                                    dilatations_deconv=None,
                                    name=self.modelname,
                                    n_res=n_res,
                                    resblocks=resblocks,
                                    h_last=None,
                                    n_elements=None,
                                    n_flows=None
                                    )
        model = model.to(device)


        test_set = CTDataset(self.path, transform=None, size=self.size)
        test_loader = DataLoader(test_set,
                                 num_workers=0,
                                 shuffle=True,
                                     batch_size=1,
                                     pin_memory=False,
                                     drop_last=True)

        # pbar = tqdm(total=len(train_loader))
        f = open("demofile2.txt", "a")
        f.write("Patient_Week,FVC,Confidence")
        for i, batch in enumerate(test_loader):
            #    pbar.update(1)
            model.zero_grad()
            patient, images, targets = batch

            images = images.to(device)
            targets = targets.to(device)

            _, mu, log_var = model(images)
            rv = norm(mu.detach().cpu().numpy(), np.exp(log_var.detach().cpu().numpy()))
            confidence = rv.pdf(mu.detach().cpu().numpy())

            l1_loss = l1(mu, targets.cuda())

            fvc = l1_loss.item() * test_set.max_fvc
            confidence = confidence * test_set.max_fvc
            f.write(",".join([patient, fvc, confidence]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.manual_seed(11)

    random.seed(10)

    size = 32
    in_channels = [1, 256, 256, 256, 256]
    out_channels = [256, 256, 256, 256, 256]
    kernel_sizes = [3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1]
    dilatations = [1, 1, 1, 1, 1]
    paddings = [1, 1, 1, 1, 1]
    bs = 16
    maxpool = 2
    has_dense = True
    batchnorm = True
    gated = False
    resblocks = False
    checkpoint_path = "../train/checkpoints"
    path = '/run/media/simon/DATA&STUFF/data/test_32x32/'

    params = {
        'mom_range': 0,
        'n_res': 0,
        'niter': 1000,
        'scheduler': 'CycleScheduler',
        'optimizer': 'adamw',
        'momentum': 0.9525840232148767,
        'learning_rate': 2.000000e-04,
        'weight_decay': 5.000000e-03,

    }
    predict = Predict(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_sizes=kernel_sizes,
                      strides=strides,
                      dilatations=dilatations,
                      path=path,
                      padding=paddings,
                      batch_size=bs,
                      checkpoint_path=checkpoint_path,
                      epochs_per_checkpoint=1,
                      gated=gated,
                      resblocks=resblocks,
                      batchnorm=batchnorm,
                      maxpool=maxpool,
                      activation=torch.nn.ReLU,
                      init_func=torch.nn.init.xavier_uniform_,
                      n_classes=1,
                      epochs_per_print=10,
                      size=size
                      )
    from matplotlib import pyplot as plt

    predict.predict(params)
