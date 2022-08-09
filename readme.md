# Hierarchical Transformer for Brain Computer Interface

**Update (2.8.2022)**: Initial code release

**Update (9.8.2022)**: Update `generate_dataset.py` in handling dataset code and method definition.


## Introduction
This repository is the implementation of our paper entitled "Hierarchical Transformer Learning for Motor Imagery Classification Tasks". Below image is an illustration of the model.

![Alt-Text](/images/model_with_bg.png)

## Requirement

Make sure you have `Python>=3.9` installed on the computer.

Install MOABB library
```bash
pip install moabb
```

The environment to support the code is listed below:
1. MOABB
2. PyTorch
3. Scipy
4. Einops
5. TQDM

## Usage
Please follow these following steps to run the code.
### Download Dataset
Open [`generate_dataset.py`](https://github.com/skepsl/BCITransformer/blob/main/generate_dataset.py) code through the IDE.
This code aims to download and generate the corresponding MI dataset for each subject. First, it will download raw datasets from MOABB and save it in the local directory. We suggest that the computer has at least 5GB free capacity to store all the dataset.

To generate **Dataset I**, use:
```bash
Dataset(dataset='BCIC').get_dataset()
```

To generate **Dataset II**, use:
```bash
Dataset(dataset='PhysioNet').get_dataset()
```

To generate **Dataset III**, use:
```bash
Dataset(dataset='Cho').get_dataset()
```

To generate **Dataset IV**, use:
```bash
Dataset(dataset='Lee').get_dataset()
```

### Training and Evaluation
The code to train and evaluate the model is inside [`main.py`](https://github.com/skepsl/BCITransformer/blob/main/main.py). 


## Citation

```
@article{tba2023,
  title={To Be Announced},
  author={TBA},
  journal={TBA},
  year={TBA}
}
```


## License
[MIT](https://choosealicense.com/licenses/mit/)

