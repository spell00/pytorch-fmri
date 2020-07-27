import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from fmri.utils.CycleAnnealScheduler import CycleScheduler
from fmri.utils.dataset import load_checkpoint, save_checkpoint, MRIDataset, _resize_data
from fmri.utils.transform_3d import Normalize, Flip90, Flip180, Flip270, XFlip, YFlip, ZFlip
from fmri.models.unsupervised.VAE_3DCNN import Autoencoder3DCNN
from fmri.models.unsupervised.SylvesterVAE3DCNN import SylvesterVAE
from fmri.utils.plot_performance import plot_performance
import torchvision
from torchvision import transforms
# from ax.plot.contour import plot_contour
# from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
# from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate, CNN
import random

import os

output_directory = "checkpoints"
from torch.utils.data import Dataset
import nilearn as nl
import h5py
import nibabel as nib
from fmri.utils.utils import validation_split

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def load_subject(filename, mask_img):
    subject_data = None
    with h5py.File(filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
    subject_img = nl.image.new_img_like(mask_img, subject_data, affine=mask_img.affine, copy_header=True)

    return subject_img


class Train:
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 kernel_sizes_deconv,
                 strides,
                 strides_deconv,
                 dilatations,
                 dilatations_deconv,
                 save,
                 padding,
                 padding_deconv,
                 path,
                 num_elements=0,
                 batch_size=8,
                 epochs=1000,
                 fp16_run=False,
                 checkpoint_path=None,
                 epochs_per_checkpoint=-1,
                 epochs_per_print=10,
                 gated=True,
                 has_dense=True,
                 batchnorm=False,
                 resblocks=False,
                 flow_type='vanilla',
                 maxpool=3,
                 verbose=2,
                 size=32,
                 mean=0.5,
                 std=0.5,
                 plot_perform=True,
                 val_share=0.1,
                 activation=torch.nn.ReLU
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.kernel_sizes_deconv = kernel_sizes_deconv
        self.strides = strides
        self.strides_deconv = strides_deconv
        self.dilatations = dilatations
        self.dilatations_deconv = dilatations_deconv
        self.padding = padding
        self.padding_deconv = padding_deconv
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
        self.flow_type = flow_type
        self.maxpool = maxpool
        self.num_elements = num_elements
        self.save = save
        self.verbose = verbose
        self.path = path
        self.size = size
        self.std = std
        self.mean = mean
        self.val_share = val_share
        self.plot_perform = plot_perform
        self.activation = activation

    def train(self, params):
        num_elements = params['num_elements']
        mom_range = params['mom_range']
        n_res = params['n_res']
        niter = params['niter']
        scheduler = params['scheduler']
        optimizer_type = params['optimizer']
        momentum = params['momentum']
        z_dim = params['z_dim']
        learning_rate = params['learning_rate'].__format__('e')
        n_flows = params['n_flows']
        weight_decay = params['weight_decay'].__format__('e')
        warmup = params['warmup']
        l1 = params['l1'].__format__('e')
        l2 = params['l2'].__format__('e')

        weight_decay = float(str(weight_decay)[:1] + str(weight_decay)[-4:])
        learning_rate = float(str(learning_rate)[:1] + str(learning_rate)[-4:])
        l1 = float(str(l1)[:1] + str(l1)[-4:])
        l2 = float(str(l2)[:1] + str(l2)[-4:])
        if self.verbose > 1:
            print("Parameters: \n\t",
                  'zdim: ' + str(z_dim) + "\n\t",
                  'mom_range: ' + str(mom_range) + "\n\t",
                  'num_elements: ' + str(num_elements) + "\n\t",
                  'niter: ' + str(niter) + "\n\t",
                  'nres: ' + str(n_res) + "\n\t",
                  'learning_rate: ' + learning_rate.__format__('e') + "\n\t",
                  'momentum: ' + str(momentum) + "\n\t",
                  'n_flows: ' + str(n_flows) + "\n\t",
                  'weight_decay: ' + weight_decay.__format__('e') + "\n\t",
                  'warmup: ' + str(warmup) + "\n\t",
                  'l1: ' + l1.__format__('e') + "\n\t",
                  'l2: ' + l2.__format__('e') + "\n\t",
                  'optimizer_type: ' + optimizer_type + "\n\t",
                  )

        self.modelname = "vae_3dcnn_" \
                         + '_flows' + self.flow_type + str(n_flows) \
                         + '_bn' + str(self.batchnorm) \
                         + '_niter' + str(niter) \
                         + '_nres' + str(n_res) \
                         + '_momrange' + str(mom_range) \
                         + '_momentum' + str(momentum) \
                         + '_' + str(optimizer_type) \
                         + "_zdim" + str(z_dim) \
                         + '_gated' + str(self.gated) \
                         + '_resblocks' + str(self.resblocks) \
                         + '_initlr' + learning_rate.__format__('e') \
                         + '_warmup' + str(warmup) \
                         + '_wd' + weight_decay.__format__('e') \
                         + '_l1' + l1.__format__('e') \
                         + '_l2' + l2.__format__('e') \
                         + '_size' + str(self.size)
        if self.flow_type != 'o-sylvester':
            model = Autoencoder3DCNN(z_dim,
                                     self.maxpool,
                                     self.in_channels,
                                     self.out_channels,
                                     self.kernel_sizes,
                                     self.kernel_sizes_deconv,
                                     self.strides,
                                     self.strides_deconv,
                                     self.dilatations,
                                     self.dilatations_deconv,
                                     self.padding,
                                     self.padding_deconv,
                                     has_dense=self.has_dense,
                                     batchnorm=self.batchnorm,
                                     flow_type=self.flow_type,
                                     n_flows=n_flows,
                                     n_res=n_res,
                                     gated=self.gated,
                                     resblocks=self.resblocks,
                                     activation=self.activation
                                     ).to(device)
        else:
            model = SylvesterVAE(z_dim=z_dim,
                                 maxpool=self.maxpool,
                                 in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 kernel_sizes=self.kernel_sizes,
                                 kernel_sizes_deconv=self.kernel_sizes_deconv,
                                 strides=self.strides,
                                 strides_deconv=self.strides_deconv,
                                 dilatations=self.dilatations,
                                 dilatations_deconv=self.dilatations_deconv,
                                 padding=self.padding,
                                 padding_deconv=self.padding_deconv,
                                 batchnorm=self.batchnorm,
                                 flow_type=self.flow_type,
                                 n_res=n_res,
                                 gated=self.gated,
                                 has_dense=self.has_dense,
                                 resblocks=self.resblocks,
                                 h_last=z_dim,
                                 n_flows=n_flows,
                                 num_elements=3,
                                 auxiliary=False,
                                 a_dim=0,
                                 ).to(device)
        model.random_init()
        criterion = nn.MSELoss(reduction="none")
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
        if self.checkpoint_path is not None and self.save:
            model, optimizer, \
            epoch, losses, \
            kl_divs, losses_recon, \
            best_loss = load_checkpoint(checkpoint_path,
                                        model,
                                        self.maxpool,
                                        save=self.save,
                                        padding=self.padding,
                                        has_dense=self.has_dense,
                                        batchnorm=self.batchnorm,
                                        flow_type=self.flow_type,
                                        padding_deconv=self.padding_deconv,
                                        optimizer=optimizer,
                                        z_dim=z_dim,
                                        gated=self.gated,
                                        in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_sizes=self.kernel_sizes,
                                        kernel_sizes_deconv=self.kernel_sizes_deconv,
                                        strides=self.strides,
                                        strides_deconv=self.strides_deconv,
                                        dilatations=self.dilatations,
                                        dilatations_deconv=self.dilatations_deconv,
                                        name=self.modelname,
                                        n_flows=n_flows,
                                        n_res=n_res,
                                        resblocks=resblocks,
                                        h_last=self.out_channels[-1],
                                        )
            model = model.to(device)
        model.flow = model.flow.to(device)
        # t1 = torch.Tensor(np.load('/run/media/simon/DATA&STUFF/data/biology/arrays/t1.npy'))
        # targets = torch.Tensor([0 for _ in t1])

        train_transform = transforms.Compose([
            XFlip(),
            YFlip(),
            ZFlip(),
            Flip90(),
            Flip180(),
            Flip270(),
            torchvision.transforms.Normalize(mean=(self.mean), std=(self.std)),
            Normalize()
        ])
        all_set = MRIDataset(self.path, transform=train_transform)
        train_set, valid_set = validation_split(all_set, val_share=self.val_share)

        train_loader = DataLoader(train_set,
                                  num_workers=0,
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  pin_memory=False,
                                  drop_last=True)
        valid_loader = DataLoader(valid_set,
                                  num_workers=0,
                                  shuffle=True,
                                  batch_size=2,
                                  pin_memory=False,
                                  drop_last=True)

        # Get shared output_directory ready
        logger = SummaryWriter('logs')
        epoch_offset = max(1, epoch)

        if scheduler == 'ReduceLROnPlateau':
            lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                     factor=0.1,
                                                                     cooldown=50,
                                                                     patience=200,
                                                                     verbose=True,
                                                                     min_lr=1e-15)
        elif scheduler == 'CycleScheduler':
            lr_schedule = CycleScheduler(optimizer,
                                         learning_rate,
                                         n_iter=niter * len(train_loader),
                                         momentum=[
                                             max(0.0, momentum - mom_range),
                                             min(1.0, momentum + mom_range),
                                         ])

        losses = {
            "train": [],
            "valid": [],
        }
        kl_divs = {
            "train": [],
            "valid": [],
        }
        losses_recon = {
            "train": [],
            "valid": [],
        }
        shapes = {
            "train": len(train_set),
            "valid": len(valid_set),
        }
        early_stop_counter = 0

        for epoch in range(epoch_offset, self.epochs):
            if early_stop_counter == 500:
                if self.verbose > 0:
                    print('EARLY STOPPING.')
                break
            best_epoch = False
            model.train()
            train_losses = []
            train_abs_error = []
            train_kld = []
            train_recons = []
            model.train()

            # pbar = tqdm(total=len(train_loader))
            for i, batch in enumerate(train_loader):
                #    pbar.update(1)
                model.zero_grad()
                images = batch
                images = images.to(device)
                images = images.unsqueeze(1)
                reconstruct, kl = model(images)
                loss_recon = criterion(
                    reconstruct,
                    images
                ).sum() / self.batch_size
                kl_div = torch.mean(kl)
                loss = loss_recon + kl_div
                # l2_reg = torch.Tensor([0])
                # l1_reg = torch.Tensor([0])
                # for name, param in model.named_parameters():
                #     if 'weight' in name:
                #         l1_reg = l1 + torch.norm(param, 1)
                # for name, param in model.named_parameters():
                #     if 'weight' in name:
                #         l2_reg = l2 + torch.norm(param, 1)
                # loss += l1 * l1_reg
                # loss += l2 * l2_reg
                loss.backward()
                # lr_schedule.step()

                try:
                    train_losses += [loss.item()]
                except:
                    return best_loss
                train_kld += [kl_div.item()]
                train_recons += [loss_recon.item()]

                logger.add_scalar('training_loss', loss.item(), i + len(train_loader) * epoch)
                del kl, loss_recon, kl_div, loss, images, reconstruct,  # , l1_reg, l2_reg, name, param

            # img = nib.Nifti1Image(images.detach().cpu().numpy()[0], np.eye(4))
            # recon = nib.Nifti1Image(reconstruct.detach().cpu().numpy()[0], np.eye(4))
            if 'views' not in os.listdir():
                os.mkdir('views')
            # img.to_filename(filename='views/image_train_' + str(epoch) + '.nii.gz')
            # recon.to_filename(filename='views/reconstruct_train_' + str(epoch) + '.nii.gz')
            losses["train"] += [np.mean(train_losses)]
            kl_divs["train"] += [np.mean(train_kld)]
            losses_recon["train"] += [np.mean(train_recons)]
            del train_losses, train_kld, train_recons, train_abs_error  # , img, recon

            if epoch % self.epochs_per_print == 0:
                if self.verbose > 1:
                    print("Epoch: {}:\t"
                          "Train Loss: {:.5f} , "
                          "kld: {:.3f} , "
                          "recon: {:.3f}"
                          .format(epoch,
                                  losses["train"][-1],
                                  kl_divs["train"][-1],
                                  losses_recon["train"][-1])
                          )

            if np.isnan(losses["train"][-1]):
                if self.verbose > 0:
                    print('PREMATURE RETURN...')
                return best_loss
            model.eval()
            valid_losses = []
            valid_kld = []
            valid_recons = []
            valid_abs_error = []
            # pbar = tqdm(total=len(valid_loader))
            for i, batch in enumerate(valid_loader):
                #    pbar.update(1)
                images = batch
                images = images.to(device)
                images = images.unsqueeze(1)
                reconstruct, kl = model(images)
                loss_recon = criterion(
                    reconstruct,
                    images
                ).sum()
                kl_div = torch.mean(kl)
                if epoch < warmup:
                    kl_div = kl_div * (epoch / warmup)
                loss = loss_recon + kl_div
                try:
                    valid_losses += [loss.item()]
                except:
                    return best_loss
                valid_kld += [kl_div.item()]
                valid_recons += [loss_recon.item()]
                logger.add_scalar('training loss', np.log2(loss.item()), i + len(train_loader) * epoch)
                del kl, loss_recon, kl_div, loss, images, reconstruct

            losses["valid"] += [np.mean(valid_losses)]
            kl_divs["valid"] += [np.mean(valid_kld)]
            losses_recon["valid"] += [np.mean(valid_recons)]
            if epoch - epoch_offset > 5:
                lr_schedule.step(losses["valid"][-1])
            # should be valid, but train is ok to test if it can be done without caring about
            # generalisation
            mode = 'valid'
            if (losses[mode][-1] < best_loss or best_loss == -1) and not np.isnan(losses[mode][-1]):
                if self.verbose > 1:
                    print('BEST EPOCH!', losses[mode][-1])
                early_stop_counter = 0
                best_loss = losses[mode][-1]
                best_epoch = True
            else:
                early_stop_counter += 1

            if epoch % self.epochs_per_checkpoint == 0 and self.save:
                # img = nib.Nifti1Image(images.detach().cpu().numpy()[0], np.eye(4))
                # recon = nib.Nifti1Image(reconstruct.detach().cpu().numpy()[0], np.eye(4))
                if 'views' not in os.listdir():
                    os.mkdir('views')
                # img.to_filename(filename='views/image_' + str(epoch) + '.nii.gz')
                # recon.to_filename(filename='views/reconstruct_' + str(epoch) + '.nii.gz')
                if best_epoch:
                    if self.verbose > 1:
                        print('Saving model...')
                    save_checkpoint(model=model,
                                    optimizer=optimizer,
                                    maxpool=maxpool,
                                    padding=self.padding,
                                    padding_deconv=self.padding_deconv,
                                    learning_rate=learning_rate,
                                    epoch=epoch,
                                    checkpoint_path=output_directory,
                                    z_dim=z_dim,
                                    gated=self.gated,
                                    batchnorm=self.batchnorm,
                                    losses=losses,
                                    kl_divs=kl_divs,
                                    losses_recon=losses_recon,
                                    in_channels=self.in_channels,
                                    out_channels=self.out_channels,
                                    kernel_sizes=self.kernel_sizes,
                                    kernel_sizes_deconv=self.kernel_sizes_deconv,
                                    strides=self.strides,
                                    strides_deconv=self.strides_deconv,
                                    dilatations=self.dilatations,
                                    dilatations_deconv=self.dilatations_deconv,
                                    best_loss=best_loss,
                                    save=self.save,
                                    name=self.modelname,
                                    n_flows=n_flows,
                                    flow_type=self.flow_type,
                                    n_res=n_res,
                                    resblocks=resblocks,
                                    h_last=z_dim
                                    )
                # del img, recon
            del valid_losses, valid_kld, valid_recons, valid_abs_error
            if epoch % self.epochs_per_print == 0:
                if self.verbose > 0:
                    print("Epoch: {}:\t"
                          "Valid Loss: {:.5f} , "
                          "kld: {:.3f} , "
                          "recon: {:.3f}"
                          .format(epoch,
                                  losses["valid"][-1],
                                  kl_divs["valid"][-1],
                                  losses_recon["valid"][-1]
                                  )
                          )
                if self.verbose > 1:
                    print("Current LR:", optimizer.param_groups[0]['lr'])
                if 'momentum' in optimizer.param_groups[0].keys():
                    print("Current Momentum:", optimizer.param_groups[0]['momentum'])
            if self.plot_perform:
                plot_performance(loss_total=losses, losses_recon=losses_recon, kl_divs=kl_divs, shapes=shapes,
                                 results_path="../figures",
                                 filename="training_loss_trace_"
                                          + self.modelname + '.jpg')
        if self.verbose > 0:
            print('BEST LOSS :', best_loss)
        return best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.manual_seed(11)

    random.seed(10)

    size = 32
    z_dim = 50
    in_channels = [1, 64, 128, 128, 128]
    out_channels = [64, 128, 128, 128, 128]
    kernel_sizes = [3, 3, 3, 3, 3]
    kernel_sizes_deconv = [3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1]
    strides_deconv = [1, 1, 1, 1, 1]
    dilatations = [1, 1, 1, 1, 1]
    dilatations_Deconv = [1, 1, 1, 1, 1, 1]
    paddings = [1, 1, 1, 1, 1]
    paddings_deconv = [1, 1, 1, 1, 1]
    dilatations_deconv = [1, 1, 1, 1, 1]
    n_flows = 10
    bs = 8
    maxpool = 2
    flow_type = 'hf'
    epochs_per_checkpoint = 1
    has_dense = True
    batchnorm = True
    gated = False
    resblocks = True
    checkpoint_path = "checkpoints"
    basedir = '/Users/simonpelletier/Downloads/images3d/t1/'
    path = basedir + str(size) + 'x' + str(size) + '/'

    n_epochs = 10000
    save = False
    training = Train(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_sizes=kernel_sizes,
                     kernel_sizes_deconv=kernel_sizes_deconv,
                     strides=strides,
                     strides_deconv=strides_deconv,
                     dilatations=dilatations,
                     dilatations_deconv=dilatations_deconv,
                     path=path,
                     padding=paddings,
                     padding_deconv=paddings_deconv,
                     batch_size=bs,
                     epochs=n_epochs,
                     checkpoint_path=checkpoint_path,
                     epochs_per_checkpoint=1,
                     gated=gated,
                     resblocks=resblocks,
                     fp16_run=False,
                     batchnorm=batchnorm,
                     flow_type=flow_type,
                     save=save,
                     maxpool=maxpool,
                     )
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "warmup", "type": "choice", "values": [0, 0]},
            {"name": "mom_range", "type": "choice", "values": [0, 0]},
            {"name": "num_elements", "type": "range", "bounds": [1, 5]},
            {"name": "niter", "type": "choice", "values": [10, 10]},
            {"name": "n_res", "type": "range", "bounds": [0, 10]},
            {"name": "z_dim", "type": "range", "bounds": [50, 256]},
            {"name": "n_flows", "type": "range", "bounds": [2, 20]},
            {"name": "scheduler", "type": "choice", "values":
                ['ReduceLROnPlateau', 'ReduceLROnPlateau']},
            {"name": "optimizer", "type": "choice", "values": ['adamw', 'adamw']},
            {"name": "l1", "type": "range", "bounds": [1e-14, 1e-1], "log_scale": True},
            {"name": "l2", "type": "range", "bounds": [1e-14, 1e-1], "log_scale": True},
            {"name": "weight_decay", "type": "range", "bounds": [1e-14, 1e-1], "log_scale": True},
            {"name": "momentum", "type": "range", "bounds": [0.9, 1.]},
            {"name": "learning_rate", "type": "range", "bounds": [1e-4, 1e-3], "log_scale": True},
        ],
        evaluation_function=training.train,
        objective_name='loss',
        minimize=True,
        total_trials=100
    )
    from matplotlib import pyplot as plt

    fig = plt.figure()
    # render(plot_contour(model=model, param_x="learning_rate", param_y="weight_decay", metric_name='Loss'))
    # fig.savefig('test.jpg')
    print('Best Loss:', values[0]['loss'])
    print('Best Parameters:')
    print(json.dumps(best_parameters, indent=4))

    # cv_results = cross_validate(model)
    # render(interact_cross_validation(cv_results))
