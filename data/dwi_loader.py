from torch.utils.data.dataset import Dataset
import numpy as np
import h5py
from data.augmentation import DWIAugment
import torch
import torch.utils.data


class h5Loader(Dataset):
    '''
    Dataloader for the proposed framework with q-space augmentation
    '''
    def __init__(self, folder, is_train=True, dir_group=20, pad_size=128):
        self.pad_size = pad_size
        self.is_train = is_train
        if self.is_train:
            self.key = 'train'
        else:
            self.key = 'val'
        self.folder = folder

        self.dir_group = dir_group  # load multiple directions each time

        self.dataset = 'hcp_wuminn'

        hf = h5py.File(self.folder, 'r')
        self.bval_max = (hf['{}_bval_vec'.format(self.key)][:,3]).max()
        hf.close()
        print(self.dataset, 'bval max: {}'.format(self.bval_max))

    def __getitem__(self, item):
        idx = torch.randint(0, self.__len__()-self.dir_group, ()).numpy()
        hf = h5py.File(self.folder, 'r')
        if self.dir_group > 0:
            idx = np.arange(idx, idx+self.dir_group)
        slice_b0 = hf['{}_b0'.format(self.key)][idx].transpose(0,2,1)
        slice_dwis = hf['{}_dwi'.format(self.key)][idx].transpose(0,2,1)
        t1 =  hf['{}_t1'.format(self.key)][idx].transpose(0,2,1)
        t2 =  hf['{}_t2'.format(self.key)][idx].transpose(0,2,1)
        t2[t2 < 0.] = 0.
        t1[t1 < 0.] = 0.
        t1 /= np.max(t1, (1,2))[:,None, None]
        t2 /= np.max(t2, (1,2))[:,None, None]
        t1[slice_b0 == 0.] = 0.
        t2[slice_b0 == 0.] = 0.
        slice_b0 = np.clip(slice_b0, 0, 1)
        cond = hf['{}_bval_vec'.format(self.key)][idx]
        bvals_ = cond[:, 3, None, None]

        cond[:,3]  /= self.bval_max
        if self.is_train:
            slice_b0, cond, slice_dwis = DWIAugment(slice_b0, cond, slice_dwis, p_reverse=0.1, p_rec=0.1)

        assert self.dir_group == slice_dwis.shape[0]

        b0_list = [np.expand_dims(slice_b0[i], -1).astype(np.float32) for i in range(self.dir_group)]
        t2_list = [np.expand_dims(t2[i], -1).astype(np.float32) for i in range(self.dir_group)]
        t1_list = [np.expand_dims(t1[i], -1).astype(np.float32) for i in range(self.dir_group)]
        dwi_list = [np.expand_dims(slice_dwis[i], -1).astype(np.float32) for i in range(self.dir_group)]
        cond_list = [cond[i].astype(np.float32) for i in range(self.dir_group)]

        return_dict = dict()
        pad_size = self.pad_size
        w, h, _ = b0_list[0].shape
        for i in range(self.dir_group):
            pad_xs = np.zeros((pad_size, pad_size, 1))
            pad_xs[:w, :h] = b0_list[i]
            b0_list[i] = pad_xs

            pad_ys = np.zeros_like(pad_xs)
            pad_ys[:w,:h] = dwi_list[i]
            dwi_list[i] = pad_ys

            pad_t2 = np.zeros_like(pad_xs)
            pad_t2[:w,:h] = t2_list[i]
            t2_list[i] = pad_t2

            pad_t1 = np.zeros_like(pad_xs)
            pad_t1[:w,:h] = t1_list[i]
            t1_list[i] = pad_t1
        del pad_xs, pad_ys, pad_t2

        for i in range(self.dir_group):
            return_dict['b0_%d'%(i+1)] = b0_list[i].transpose(2, 0, 1)
            return_dict['t2_%d'%(i+1)] = t2_list[i].transpose(2, 0, 1)
            return_dict['t1_%d'%(i+1)] = t1_list[i].transpose(2, 0, 1)
            return_dict['dwi_%d'%(i+1)] = dwi_list[i].transpose(2, 0, 1)
            return_dict['cond_%d'%(i+1)] = cond_list[i]
        return return_dict

    def __len__(self):
        hf = h5py.File(self.folder, 'r')
        return hf['{}_b0'.format(self.key)].shape[0]
