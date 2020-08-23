import os
import torch
import itertools
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from fmri.models.supervised.resnetcnn3d import ConvResnet3D
from fmri.models.unsupervised.VAE_3DCNN import Autoencoder3DCNN
from fmri.models.unsupervised.SylvesterVAE3DCNN import SylvesterVAE
import random
import pydicom
import pandas as pd

random.seed(42)


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


class EEGDataset(Dataset):
    def __init__(self, path, transform=None, crop_size=100000, device='cuda'):
        self.path = path
        self.device = device
        self.crop_size = crop_size
        self.samples = os.listdir(path)
        self.transform = transform
        self.crop_size = crop_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        x = np.load(self.path + x)
        max_start_crop = x.shape[1] - self.crop_size
        ran = np.random.randint(0, max_start_crop)
        x = torch.Tensor(x)[:, ran:ran + self.crop_size]  # .to(self.device)
        # x.requires_grad = False
        if self.transform:
            x = self.transform(x)
        return x


class MRIDataset(Dataset):
    def __init__(self, path, transform=None, size=32, device='cuda'):
        self.path = path
        self.device = device
        self.size = size
        self.samples = os.listdir(path)
        random.shuffle(self.samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        x = nib.load(self.path + x).dataobj
        x = np.array(x)
        # x = _resize_data(x, (self.size, self.size, self.size))
        x = torch.Tensor(x)  # .to(self.device)
        # x.requires_grad = False
        if self.transform:
            x = self.transform(x)
        return x.unsqueeze(0)


def load_scan(path):
    """
    Loads scans from a folder and into a list.

    Parameters: path (Folder path)

    Returns: slices (List of slices)
    """
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(scans):
    """
    Converts raw images to Hounsfield Units (HU).

    Parameters: scans (Raw images)

    Returns: image (NumPy array)
    """

    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Since the scanning equipment is cylindrical in nature and image output is square,
    # we set the out-of-scan pixels to 0
    image[image == -2000] = 0

    # HU = m*P + b
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


class CTDataset(Dataset):
    def __init__(self, path, transform=None, size=32, device='cuda'):
        self.path = path
        self.device = device
        self.size = size
        self.samples = os.listdir(path)
        random.shuffle(self.samples)
        self.transform = transform
        self.labels = pd.read_csv('/run/media/simon/DATA&STUFF/data/train.csv')
        self.max_fvc = max(self.labels['FVC'].to_list())
        self.labels['FVC'] /= self.max_fvc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.load(self.path + '/' + self.samples[idx])
        # x.requires_grad = False
        if self.transform:
            x = self.transform(x)
        patient = self.labels['Patient'].to_list().index(self.samples[idx])
        return patient, x.unsqueeze(0), torch.Tensor(
            [float(self.labels['FVC'][patient])])


class MRIDatasetClassifier(Dataset):
    def __init__(self, path, transform=None, size=32, device='cuda'):
        self.path = path
        self.device = device
        self.size = size
        self.names = os.listdir(path)
        random.shuffle(self.names)
        self.samples = []
        self.targets = []
        for i, name in enumerate(self.names):
            samples = os.listdir(path + '/' + name + '/' + str(size) + 'x' + str(size))
            # self.samples.extend(samples)
            self.targets.extend([i for _ in range(len(samples))])
            self.samples.extend([path + '/' + name + '/' + str(size) + 'x' + str(size) + "/" + s for s in samples])
        indices = [i for i in range(len(self.targets))]
        random.shuffle(indices)
        self.samples = [self.samples[ind] for ind in indices]
        self.targets = [self.targets[ind] for ind in indices]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # x = self.samples[idx]
        target = self.targets[idx]
        x = nib.load(self.samples[idx]).dataobj
        x = np.array(x)
        x = torch.Tensor(x)
        if self.transform:
            x = self.transform(x)
        return x.unsqueeze(0), target


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
                    n_elements,
                    name="vae_1dcnn",
                    model_name=Autoencoder3DCNN

                    ):
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
                            n_elements=n_elements,
                            model_name=Autoencoder3DCNN
                            )
    checkpoint_dict = torch.load(checkpoint_path + '/' + name, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    best_loss = checkpoint_dict['best_loss']
    # optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    losses_recon = checkpoint_dict['losses_recon']
    kl_divs = checkpoint_dict['kl_divs']
    losses = checkpoint_dict['losses']
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
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
                    n_elements,
                    flow_type='vanilla',
                    best_loss=-1,
                    has_dense=True,
                    name="vae_3dcnn",
                    model_name=Autoencoder3DCNN,
                    is_bayesian=True,
                    n_classes=1
                    ):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if not save:
        return
    if name == 'classifier':
        model_for_saving = model_name(maxpool=maxpool,
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
                                      model_name=Autoencoder3DCNN
                                      )
    model_type = name.split("_")[0]
    if model_type == 'vae':
        if flow_type != 'o-sylvester':
            model_for_saving = model_name(maxpool=maxpool,
                                          padding=padding,
                                          batchnorm=batchnorm,
                                          padding_deconv=padding_deconv,
                                          flow_type=flow_type,
                                          has_dense=has_dense,
                                          n_flows=n_flows,
                                          z_dim=z_dim,
                                          n_res=n_res,
                                          resblocks=resblocks,
                                          # h_last=h_last,
                                          gated=gated,
                                          in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_sizes=kernel_sizes,
                                          kernel_sizes_deconv=kernel_sizes_deconv,
                                          strides=strides,
                                          strides_deconv=strides_deconv,
                                          dilatations=dilatations,
                                          dilatations_deconv=dilatations_deconv,
                                          # model_name=Autoencoder3DCNN
                                          )
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
                                            h_last=z_dim,
                                            n_flows=n_flows,
                                            num_elements=n_elements,
                                            auxiliary=False,
                                            a_dim=0,

                                            )
    elif model_type == "classif":
        model_for_saving = ConvResnet3D(maxpool,
                             in_channels,
                             out_channels,
                             kernel_sizes,
                             strides,
                             dilatations,
                             padding,
                             batchnorm,
                             n_classes,
                             is_bayesian=is_bayesian,
                             activation=torch.nn.ReLU,
                             n_res=n_res,
                             gated=gated,
                             has_dense=has_dense,
                             resblocks=resblocks,
                             )
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'losses': losses,
                'best_loss': best_loss,
                # 'kl_divs': kl_divs,
                # 'losses_recon': losses_recon,
                'epoch': epoch,
                # 'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path + '/' + name)
