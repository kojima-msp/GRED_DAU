GRED_DAU
====

[![paper-info](https://img.shields.io/badge/APSIPA_TSIP-Open_Access-gray?labelColor=00629B)](https://www.emerald.com/atsip/article/15/1/332/1367519/Graph-signal-denoising-using-regularization-by)
[![doi](https://img.shields.io/badge/DOI-10.1108%2FATSIP--12--2025--0110-gray?labelColor=FCB61F)](https://doi.org/10.1108/ATSIP-12-2025-0110)
[![arXiv](https://img.shields.io/badge/arXiv-2512.14213-gray?labelColor=b31b1b)](https://arxiv.org/abs/2512.14213)
[![Python](https://custom-icon-badges.herokuapp.com/badge/Python-3572A5?logo=Python&logoColor=white)]()
[![our-page](https://img.shields.io/badge/Our_Homepage-green)](https://www.sip.comm.eng.osaka-u.ac.jp/)

Official Pytorch implementation of the paper "[Graph Signal Denoising Using
Regularization by Denoising and Its Parameter Estimation]()" (accepted to APSIPA TSIP).

## Abstract
> In this paper, we propose an interpretable denoising method for graph signals using regularization by denoising (RED). RED is a technique developed for image restoration that uses an efficient (and sometimes black-box) denoiser in the regularization term of the optimization problem. By using RED, optimization problems can be designed with the explicit use of the denoiser, and the gradient of the regularization term can be easily computed under mild conditions. We adapt RED for denoising of graph signals beyond image processing. We show that many graph signal denoisers, including graph neural networks, theoretically or practically satisfy the conditions for RED. We also study the effectiveness of RED from a graph filter perspective. Furthermore, we propose supervised and unsupervised parameter estimation methods based on deep algorithm unrolling. These methods aim to enhance the algorithm applicability, particularly in the unsupervised setting. Denoising experiments for synthetic and real-world datasets show that our proposed method improves signal denoising accuracy in mean squared error compared to existing graph signal denoising methods.

## Install
We use [uv](https://docs.astral.sh/uv/) to manage the Python environment. 

```bash
git clone https://github.com/kojima-msp/GRED_DAU.git
cd GRED_DAU
uv sync
uv tool install gdown
source .venv/bin/activate

# install dataset and pretrained weights
gdown "https://drive.google.com/drive/folders/10q2CwEwEiOQ7veMjyG5-D8froEd2Zjsf?usp=drive_link" --folder -O datasets/
gdown "https://drive.google.com/drive/folders/1KgOV3VC1l18PwdarzKh7B5299FUuwaHR?usp=drive_link" --folder -O src/exp/train/
```

## Usage

### Make datasets (Optional)
Datasets are also available via the `gdown` in the Install section. If you want to regenerate them from scratch, run:
```
# you need to download the modelnet dataset from https://modelnet.cs.princeton.edu/ and place it in the `datasets/ModelNet10` folder
python datasets/make_bandlimited.py
python datasets/make_modelnet.py
```

### Train models
```
# train LR/PnP with optuna optimization
#     --dataset {bandlimited, modelnet}
python src/exp/train/train_lr.py --dataset bandlimited
python src/exp/train/train_pnp.py --dataset bandlimited

# train proposed method with optuna optimization
#     --denoiser {lr, pnp}: the denoiser used in the proposed method
#     --dataset {bandlimited, modelnet}
python src/exp/train/train_proposed.py --denoiser lr --dataset bandlimited

# train proposed method with DAU
#     --denoiser {lr, pnp}: the denoiser used in the proposed method
#     --dataset {bandlimited, modelnet}
python src/exp/train/train_proposed_dau.py --denoiser lr --dataset bandlimited
```

### Evaluate models
```
# --dataset {bandlimited, modelnet}
python src/exp/eval/eval.py --dataset bandlimited
```


### Plot RMSE latex table (Table 2 and 3)
```
#     --dataset {bandlimited, modelnet}
python src/exp/eval/print_scores.py --dataset bandlimited
```


### Visualization Results (Figure 3 and 4)
```
#     --dataset {bandlimited, modelnet}
#     --noise_level {10, 15, 20, 25, 30}
#     --mode {original, diff}: plot the denoised signal itself ("original")
#                              or its absolute error against the ground truth ("diff")
python src/exp/eval/visualization.py --dataset bandlimited --noise_level 20 --mode diff
```

### Show Computation Time (Table 4)
```
python src/exp/eval/calc_computation_time.py --dataset bandlimited
python src/exp/eval/calc_computation_time.py --dataset modelnet 
```

## Citation
```
```
