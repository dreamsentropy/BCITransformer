# Hierarchical Transformer for Brain Computer Interface

**Update (9.8.2022)**: Initial code release

This repository is the implementation of our paper entitled "Hierarchical Transformer Learning for Motor Imagery Classification Tasks". Below image is an illustration of the model.

![Alt-Text](/images/model_with_bg.png)

## Requirement

Make sure you have 'python>=3.9' installed on the computer.

Install MOABB library
'''bash
pip install moabb
'''

The environment to support the code is listed below:
1. MOABB
2. PyTorch
3. Scipy
4. Einops
5. TQDM

## Usage
Please follow these following steps to run the code.
1. Run ['generate_dataset.py']:https://github.com/skepsl/BCITransformer/blob/main/generate_dataset.py
This code aims to generate the corresponding MI dataset for each subject. First, it will download raw dataset from MOABB and save it in the local directory. We suggest that the computer has at least 5GB free capacity to store all the dataset. The function 

2. Run ['main.ph']:https://github.com/skepsl/BCITransformer/blob/main/main.py
## Citation
Please properly cite our work. 


## License
[MIT](https://choosealicense.com/licenses/mit/)

