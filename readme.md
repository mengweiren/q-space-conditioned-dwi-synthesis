# Multi-modal q-space Conditioned DWI Synthesis
Reference Implementation of paper "Q-space conditioned Translation Networks forDirectional Synthesis of Diffusion WeightedImages from Multi-modal Structural MRI" of Mengwei Ren*, Heejong Kim*, Neel Dey and Guido Gerig (* equal contribution), to appear in MICCAI 2021.

[Project Page](https://heejongkim.com/dwi-synthesis) | [Paper](https://arxiv.org/abs/2106.13188) 

![Synthetic results on interoplated q-space](demo.gif)

## Dependencies 
```shell
conda env create -f environment.yml
conda activate smri2dwi
```

## Data preparation
We recommend using h5 files for fast data loading during training. We assume the training h5 file includes the following datasets & shape, where N is the number of (w,h) 2D slices.  
```shell
- 'train_b0': N, h, w 
- 'train_bval_vec': N, 4
- 'train_dwi': N, h, w
- 'train_t1': N, h, w
- 'train_t2': N, h, w
```

## Training
All network and training related parameters will be specified in a configuration file. Simply run the following command to start training.
```shell script
python train.py --config ../config/smri2scalar.yaml
```

## Testing
We provide a notebook under ./mains/ that shows how to synthesize a 2D slice give structural inputs and b-vector/b-values as conditions.

## Citation
If you use this code, please consider citing our work:
```
@misc{ren2021qspace,
      title={Q-space Conditioned Translation Networks for Directional Synthesis of Diffusion Weighted Images from Multi-modal Structural MRI}, 
      author={Mengwei Ren and Heejong Kim and Neel Dey and Guido Gerig},
      year={2021},
      eprint={2106.13188},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
