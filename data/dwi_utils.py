import os, sys

sys.path.append('..')
sys.path.append('.')

import numpy as np
import h5py
import time


def read_h5(h5_file, subject, is_train=True, slice_idx=None, dir_idx=None, return_dwi=False,
            group_dir=0, load_noddi=False, load_dki=False, slice_range=None, load_t12=False):
    '''
    :param h5_file: the h5 file containing data [string or
    :param subject:
    :param is_train: specify to load corresponding dwis
    :param slice_idx: slice wise index
    :param dir_idx: direction wise index,
            if None, load full volume;
            if scalar, load corresponding directional volume,
            if -1, load partial volumes randomly with the number specified by group dir
    :param return_dwi: whether to return the dwi or not
    :param group_dir: number of directions loaded
    :param load_dti: whether to load dti related derivatives
    :return:
    '''
    if isinstance(h5_file, h5py._hl.files.File):
        hf = h5_file
    else:
        print(h5_file)
        hf = h5py.File(h5_file, 'r')
    if str(subject) not in hf.keys():
        gp = hf[str(id)]
    else:
        gp = hf[str(subject)]
    data_id = 'train' if is_train else 'val'
    ndir = gp['%s_bvec' % data_id].shape[0]
    print(gp['%s_bvec' % data_id].shape)
    if slice_idx is None:  # for testing purpose, load full volume at once
        print(gp.keys())
        affine = gp['affine'][()]
        norm_b0 = gp['b0'][()]
        b0_max = gp['b0_max'][()]
        idx_ = gp['%s_idx' % data_id][()]
        bvec_ = gp['%s_bvec' % data_id][()]
        bval_ = gp['%s_bval' % data_id][()]
        if return_dwi:
            print('loading dwi {}'.format(gp['%s_dwi' % data_id].shape))
            s = time.time()
            dwi_ = gp['%s_dwi' % data_id][()]
            e = time.time()
            print('%.4f sec' % (e - s), dwi_.shape)
        else:
            dwi_ = None
        return_data = {'b0': norm_b0,
                       'b0_max': b0_max,
                       'dir_idx': idx_,
                       'bvec': bvec_,
                       'bval': bval_,
                       'dwi': dwi_,
                       'affine': affine}
        if load_t12:
            t1_ = gp['t1'][()]
            t2_ = gp['t2'][()]
            return_data['t1'] = t1_
            return_data['t2'] = t2_
        if load_noddi:
            nd = gp['ndi'][()]
            od = gp['odi'][()]
            return_data['nd'] = nd
            return_data['od'] = od
        if load_dki:
            ak = gp['ak'][()]
            mk = gp['mk'][()]
            rk = gp['rk'][()]
            return_data['ak'] = ak
            return_data['mk'] = mk
            return_data['rk'] = rk
        return return_data
    else:  # for training, load 2D slice
        slice_b0 = gp['b0'][:, :, slice_idx].transpose()
        num_slice = gp['b0'].shape[-1]
        while np.sum(slice_b0) == 0.:
            low, high = slice_range if slice_range is not None else (0, num_slice)
            slice_idx = np.random.randint(low, high)
            slice_b0 = gp['b0'][:, :, slice_idx].transpose()
        num_bvec = gp['%s_bvec' % data_id].shape[0]
        if group_dir > 0 or dir_idx is None:
            dir_idx = np.random.choice(num_bvec, group_dir, replace=False)
            dir_idx = np.sort(dir_idx)
            print(dir_idx)
        bvec_ = gp['%s_bvec' % data_id][dir_idx]
        bval_ = gp['%s_bval' % data_id][dir_idx]
        slice_dwi = gp['%s_dwi' % data_id][dir_idx, :, :, slice_idx]  # n_dwi, w, h
        print(slice_dwi.shape)
        slice_dwi = slice_dwi.transpose(0, 2, 1)
        b0_max = gp['b0_max'][()]

        return_data = {'b0': slice_b0,
                       'b0_max': b0_max,
                       'dir_idx': dir_idx,
                       'bvec': bvec_,
                       'bval': bval_,
                       'dwi': slice_dwi}

        if load_t12:
            slice_t1 = gp['t1'][:, :, slice_idx].transpose()
            slice_t2 = gp['t2'][:, :, slice_idx].transpose()
            print(slice_t1.shape, slice_t2.shape)
            return_data['t1'] = slice_t1
            return_data['t2'] = slice_t2
        hf.close()
        return return_data



