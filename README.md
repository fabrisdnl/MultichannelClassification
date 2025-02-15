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

```
usage: main.py --dataset_path "../EuroSAT_MS" --saves_dir "saves" --mat_format

where instead of "../EuroSAT_MS" must be written the path to the dataset,
while "saves" create a directory in the project main directory to save the models

optional: --load_saved_models to use directly the precomputed models
            in later executions; can't be used in first execution
          --mat_format whether the input dataset in in MATLAB data file format
```

The dataset is supposed to be already extracted in a local folder.

## 📜 Licenza

Questo software è **sotto licenza proprietaria**. L'uso, la modifica, la distribuzione o la vendita **non sono permessi senza un'esplicita autorizzazione**.

🔒 **Uso commerciale?**  
Se desideri utilizzare questo software per scopi commerciali o industriali, contattami per ottenere una licenza commerciale:  
📩 [fabrisd.wm@gmail.com]  

---
© [2025] [Daniele Fabris] - Tutti i diritti riservati.
