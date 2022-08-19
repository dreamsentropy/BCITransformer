# Hierarchical Transformer for Brain Computer Interface

**Update (2.8.2022)**: Initial code release

**Update (9.8.2022)**: Update `generate_dataset.py` in handling dataset code and method definition.


## Introduction
TBA

## Requirement
### Environment
Make sure you have `Python==3.9` installed on the computer.

### Installation
1. [PyTorch](pytorch.org)
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
```

2. [MOABB](http://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014001.html)
```bash
pip install moabb==0.4.5
```

3. [SciPy](scikit-learn.org)
```bash
pip install scikit-learn==1.8.1
```

4. [Einops](https://pypi.org/project/einops/)
```bash
pip install einops==0.4.1
```

3. [tqdm](https://pypi.org/project/tqdm/)
```bash
pip install tqdm
```

## Usage
Please follow these following steps to run the code.
### Download Dataset
Open [`generate_dataset.py`](https://github.com/skepsl/BCITransformer/blob/main/generate_dataset.py) code through the IDE.
This code aims to download and generate the corresponding MI dataset for each subject. First, it will download raw datasets from MOABB and save it in the local directory. We suggest that the computer has at least 5GB free capacity to store all original and preprocessed datasets.

The argument for `dataset` is either `BCIC`, `PhysioNet`, `Cho`, `Lee`.

Example to generate **Dataset I**, use:
```bash
Dataset(dataset='BCIC').get_dataset()
```

### Training and Evaluation
The code to train and evaluate the model is inside [`main.py`](https://github.com/skepsl/BCITransformer/blob/main/main.py). 
The argument for `dataset` is either `BCIC`, `PhysioNet`, `Cho`, `Lee`. The fold is a number between 1-10. Hence, the subject is a number. 

Example to  train **Dataset I**, **subject 1**, **fold 1**, use:
```bash
train(dataset='BCIC', subject=1, fold=1) 
```

## Citation

```
@article{tba2023,
  title={To Be Announced},
  author={TBA},
  journal={TBA},
  year={TBA}
}
```


