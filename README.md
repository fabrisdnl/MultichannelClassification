# Land Cover Classification

## Overview
This project focuses on land cover classification using multi-spectral images from the Sentinel-2 satellite, specifically leveraging the EuroSAT dataset.
The goal is to develop a deep learning model that can accurately classify land cover types, such as forests, urban areas, water bodies, and agricultural land, based on spectral features derived from 13 bands of Sentinel-2 imagery.

## Requirements
The runs have been executed using:
- Python 3.10
- CUDA 11.8.89
- PyTorch 2.5.1+cu118.

## Usage
To execute the code the input parameters must be:
--dataset_path "../EuroSAT_MS" --saves_dir "saves"

Obviously the dataset path must be corrected according to the relative path.
The dataset is supposed to be already extracted in a local folder.

