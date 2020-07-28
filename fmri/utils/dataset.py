import os
import torch
import itertools
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from fmri.models.unsupervised.VAE_3DCNN import Autoencoder3DCNN
from fmri.models.unsupervised.SylvesterVAE3DCNN import SylvesterVAE


def _resize_data(data, new_size=(160, 160, 160)):
    initial_size_x = data.shape[0]
    initial_size_y = data.shape[1]
    initial_size_z = data.shape[2]

    new_size_x = new_size[0]
    new_size_y = new_size[1]
    new_size_z = new_size[2]

    delta_x = initial_size_x / new_size_x
    delta_y = initial_size_y / new_size_y
    delta_z = initial_size_z / new_size_z

    new_data = np.zeros((new_size_x, new_size_y, new_size_z))

    for x, y, z in itertools.product(range(new_size_x),
                                     range(new_size_y),
                                     range(new_size_z)):
        new_data[x][y][z] = data[int(x * delta_x)][int(y * delta_y)][int(z * delta_z)]

    return new_data


class MRIDataset(Dataset):
    def __init__(self, path, transform=None, size=32, device='cuda'):
        self.path = path
        self.device = device
        self.size = size
        self.samples = os.listdir(path)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        x = nib.load(self.path + x).dataobj
        x = np.array(x)
        # x = _resize_data(x, (self.size, self.size, self.size))
        x = torch.Tensor(x) # .to(self.device)
        # x.requires_grad = False
        if self.transform:
            x = self.transform(x)
        return x.unsqueeze(0)


def load_checkpoint(checkpoint_path,
                    model,
                    maxpool,
                    padding,
                    padding_deconv,
                    optimizer,
                    z_dim,
                    gated,
                    in_channels,
                    out_channels,
                    kernel_sizes,
                    kernel_sizes_deconv,
                    strides,
                    has_dense,
                    strides_deconv,
                    dilatations,
                    dilatations_deconv,
                    batchnorm,
                    flow_type,
                    save,
                    n_flows,
                    n_res,
                    resblocks,
                    h_last,
                    name="vae_1dcnn"):
    # if checkpoint_path
    losses_recon = {
        "train": [],
        "valid": [],
    }
    kl_divs = {
        "train": [],
        "valid": [],
    }
    losses = {
        "train": [],
        "valid": [],
    }
    if checkpoint_path not in os.listdir():
        os.mkdir(checkpoint_path)
    if name not in os.listdir(checkpoint_path):
        print("Creating checkpoint...")
        if save:
            save_checkpoint(model=model,
                            optimizer=optimizer,
                            maxpool=maxpool,
                            padding=padding,
                            padding_deconv=padding_deconv,
                            flow_type=flow_type,
                            save=save,
                            learning_rate=None,
                            has_dense=has_dense,
                            epoch=0,
                            checkpoint_path=checkpoint_path,
                            z_dim=z_dim,
                            gated=gated,
                            losses=losses,
                            kl_divs=kl_divs,
                            losses_recon=losses_recon,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_sizes=kernel_sizes,
                            kernel_sizes_deconv=kernel_sizes_deconv,
                            strides=strides,
                            strides_deconv=strides_deconv,
                            dilatations=dilatations,
                            dilatations_deconv=dilatations_deconv,
                            batchnorm=batchnorm,
                            name=name,
                            n_flows=n_flows,
                            n_res=n_res,
                            resblocks=resblocks,
                            h_last=z_dim,
                            )
    checkpoint_dict = torch.load(checkpoint_path + '/' + name, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    best_loss = checkpoint_dict['best_loss']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    losses_recon = checkpoint_dict['losses_recon']
    kl_divs = checkpoint_dict['kl_divs']
    losses = checkpoint_dict['losses']
    print("Loaded checkpoint '{}' (epoch {})".format(
        checkpoint_path, epoch))
    return model, optimizer, epoch, losses, kl_divs, losses_recon, best_loss


def save_checkpoint(model,
                    optimizer,
                    maxpool,
                    padding,
                    padding_deconv,
                    learning_rate,
                    epoch,
                    checkpoint_path,
                    z_dim,
                    gated,
                    losses,
                    kl_divs,
                    losses_recon,
                    in_channels,
                    out_channels,
                    kernel_sizes,
                    kernel_sizes_deconv,
                    strides,
                    strides_deconv,
                    dilatations,
                    dilatations_deconv,
                    batchnorm,
                    save,
                    n_flows,
                    n_res,
                    resblocks,
                    h_last,
                    flow_type='vanilla',
                    best_loss=-1,
                    has_dense=True,
                    name="vae_1dcnn"):
    if not save:
        return
    if flow_type != 'o-sylvester':
        model_for_saving = Autoencoder3DCNN(maxpool=maxpool,
                                            padding=padding,
                                            batchnorm=batchnorm,
                                            padding_deconv=padding_deconv,
                                            flow_type=flow_type,
                                            has_dense=has_dense,
                                            n_flows=n_flows,
                                            z_dim=z_dim,
                                            n_res=n_res,
                                            resblocks=resblocks,
                                            h_last=h_last,
                                            gated=gated,
                                            in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_sizes=kernel_sizes,
                                            kernel_sizes_deconv=kernel_sizes_deconv,
                                            strides=strides,
                                            strides_deconv=strides_deconv,
                                            dilatations=dilatations,
                                            dilatations_deconv=dilatations_deconv,
                                            ).cuda()
    else:
        model_for_saving = SylvesterVAE(z_dim=z_dim,
                                        maxpool=maxpool,
                                        in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_sizes=kernel_sizes,
                                        kernel_sizes_deconv=kernel_sizes_deconv,
                                        strides=strides,
                                        strides_deconv=strides_deconv,
                                        dilatations=dilatations,
                                        dilatations_deconv=dilatations_deconv,
                                        padding=padding,
                                        padding_deconv=padding_deconv,
                                        batchnorm=batchnorm,
                                        flow_type=flow_type,
                                        n_res=n_res,
                                        gated=gated,
                                        has_dense=has_dense,
                                        resblocks=resblocks,
                                        h_last=h_last,
                                        n_flows=n_flows,
                                        num_elements=3,
                                        auxiliary=False,
                                        a_dim=0
                                        ).cuda()

    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'losses': losses,
                'best_loss': best_loss,
                'kl_divs': kl_divs,
                'losses_recon': losses_recon,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path + '/' + name)
