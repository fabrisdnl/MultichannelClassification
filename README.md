# Land Cover Classification

## Overview
This project focuses multi-spectral images analysis. Two applications were tested: satellite image classification tested on the EuroSAT and LCZ42 datasets and  a species-level identification of planktic foraminifera.
The proposed architecture effectively extracts multiscale spatial and spectral representations via a 3D CNN backbone before processing them through a Transformer Encoder, which refines feature dependencies and enhances global context modeling.
The goal is to develop a custom deep learning model based on attention.

## Requirements
The runs have been executed using:
- Python 3.10
- CUDA 11.8.89
- PyTorch 2.5.1+cu118.
- Torchvision 0.20.1+cu118.

## Usage
To execute the code the input parameters must be:

```
usage: main.py --dataset_path "<dataset_path>" --saves_dir "<saves_dir>" --directory --no_valid

optional: --dataset_path path to dataset (or folder of multiple datasets with flag --directory)
          --load_saved_models to use directly the precomputed models in later executions; can't be used in first execution
          --saves flag to indicates the path where to saves results
          --labels_txt to specify for foraminifera that the labels are provided in txt files
          --no_valid indicates if ignore validation and do only training
          --directory' indicates if input is a folder of mat files (group of datasets)
```

The datasets are supposed to be already extracted in local folders (dataset must be cell arrays in mat files).
For the dataset Foraminifera the labels must be provided in txt format (after the substring fold there is 'x' then use label_x.txt).

## License

This software is released under a **Custom Non-Commercial License**.

You are free to use, copy, modify, and distribute this code for **personal and research purposes only**. **Commercial use is strictly prohibited** without prior written permission from the author.

For commercial licensing inquiries, please contact: [fabrisd.wm@gmail.com]

See the [LICENSE](./LICENSE) file for full details.