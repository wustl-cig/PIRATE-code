# PIRATE

This is the code repo for the paper ['*A Plug-and-Play Image Registration Network*'](https://arxiv.org/pdf/2310.04297.pdf).
### [Project Page](https://wustl-cig.github.io/pirate/)  | [Preprint](https://arxiv.org/abs/2310.04297) | [Camera ready](https://openreview.net/forum?id=DGez4B2a6Y)

## Download datasets
Available datasets:
- [OASIS-1](https://www.oasis-brains.org/)
- [CANDI](https://www.nitrc.org/projects/candi_share/)
- [EMPIRE10-lung](https://empire10.grand-challenge.org/)

## Setup the environment
Prerequisites
```
pytorch 2.2.1
numpy 1.23.1
SimpleITK 2.2.1
tqdm 4.64.1
h5py 3.7.0
```

Setup the environment
```
conda env create --file PIRATE.yml
```
To activate this environment, use
```
conda activate PIRATE-env
```
To deactivate an active environment, use
```
conda deactivate
```
## Run the code
Run inference PIRATE:
```
python inference_PIRATE.py
```

Run inference PIRATE+:
```
python inference_PIRATEplus.py
```

**NOTE**: We already provide the pre-trained models in the folder ```pretrained_model/AWGN_denoiser/``` and ```pretrained_model/PIRATEplus/```

Run training PIRATE(AWGN denoiser):
```
python train_denoiser.py
```

Run training PIRATE+:
```
python train_PIRATEplus.py
```

## Expected outputs
After inference, the results will be saved in the folder ```output```, including
```
the warped image (.nii.gz)
```

## File structure
```
PIRATE
  |-data: example data
    |-fixed.nii.gz
	  |-moving.nii.gz
		|-field.h5py
  |-model: PIRATE and PIRATE+ model
    |-base.py: basic functions.
	  |-loss.py: loss functions used in training and inference.
		|-PIRATE.py: PIRATE model.
		|-PIRATEplus.py: PIRATE+ model.
  |-output: store output images.
  |-pretrained_model:
    |-AWGN_denoiser: pretrained PIRATE on OASIS-1 dataset
    |-PIRATEplus: pretrained PIRATE+ on OASIS-1 dataset
  |-inference_PIRATE.py : inference function of PIRATE.
  |-inference_PIRATEplus.py: inference function of PIRATE+.
  |-train_denoiser.py : training function of PIRATE.
  |-train_PIRATEplus.py: training function of PIRATE+.
```
