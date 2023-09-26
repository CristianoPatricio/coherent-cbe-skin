# Coherent Concept-based Explanations in Medical Image and Its Application to for Skin Lesion Diagnosis

<a href="https://openaccess.thecvf.com/content/CVPR2023W/SAIAD/papers/Patricio_Coherent_Concept-Based_Explanations_in_Medical_Image_and_Its_Application_to_CVPRW_2023_paper.pdf" target="_blank">Paper</a> accepted at the CVPR 2023 workshop SAIAD - Safe Artificial Intelligence for All Domains.

### Citation

If you use this repository, please cite:

```
@inproceedings{patricio2023coherent,
  title={Coherent Concept-based Explanations in Medical Image and Its Application to Skin Lesion Diagnosis},
  author={Patr{\'\i}cio, Cristiano and Neves, Jo{\~a}o C and Teixeira, Luis F},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages={3798--3807},
  year={2023}
}
```

### 1. Dataset Statistics

<table>
    <tr>
        <td>Dataset</td>
        <td colspan="2" align="center">Train</td>
        <td colspan="2" align="center">Validation</td>
        <td colspan="2" align="center">Test</td>
        <td>Total</td>
    </tr>
    <tr>
        <td></td>
        <td>Mel.</td>
        <td>Nev.</td>
        <td>Mel.</td>
        <td>Nev.</td>
        <td>Mel.</td>
        <td>Nev.</td>
        <td></td>
    </tr>
    <tr>
        <td>PH2</td>
        <td>30</td>
        <td>120</td>
        <td>5</td>
        <td>20</td>
        <td>5</td>
        <td>20</td>
        <td>200</td>
    </tr>
    <tr>
        <td>Derm7pt</td>
        <td>90</td>
        <td>256</td>
        <td>61</td>
        <td>100</td>
        <td>101</td>
        <td>219</td>
        <td>827</td>
    </tr>
    <tr>
        <td>PH2D7</td>
        <td>120</td>
        <td>376</td>
        <td>66</td>
        <td>120</td>
        <td>106</td>
        <td>239</td>
        <td>1,027</td>
    </tr>
</table>


### 2. Reproducibility of the results

Download the required datasets:

1. PH2: https://www.fc.up.pt/addi/ph2%20database.html
2. Derm7pt: https://derm.cs.sfu.ca/Welcome.html

First of all, create a new conda environment with the required libraries contained in requirements.txt file:

```bash
conda create --name <env> --file requirements.txt
```

For evaluating the model in a specified dataset, please ensure that you modify the directory paths and specify the parameters in model_params.py

Evaluate a baseline model:

In `model_params.py` set `BASELINE = True`

In `evaluate.py` choose the model: `models = ["resnet101"]  # "densenet201", "seresnext"` and uncomment baseline gammas:

```python
# BASELINE
gammas = {"ph2": [None, None, None],
          "ph2_dlv3_ft": [None, None, None],
          "ph2_manually": [None, None, None],
          "derm7pt": [None, None, None],
          "derm7pt_dlv3_ft": [None, None, None],
          "derm7pt_manually": [None, None, None],
          "ph2derm7pt": [None, None, None],
          "ph2derm7pt_dlv3_ft": [None, None, None],
          "ph2derm7pt_manually": [None, None, None]}
```

and finally, run the script:

```python
python evaluate.py
```

Evaluate the proposed model:

In `model_params.py` set `BASELINE = False`

In `evaluate.py` choose the model: `models = ["resnet101"]  # "densenet201", "seresnext"` and uncomment OUR METHOD gammas:

```python
# OUR METHOD
gammas = {"ph2": [0.6, 0.6, 0.6],
          "ph2_dlv3_ft": [0.6, 0.6, 0.6],
          "ph2_manually": [0.6, 0.6, 0.6],
          "derm7pt": [0.3, 0.7, 0.3],
          "derm7pt_dlv3_ft": [0.6, 0.5, 0.6],
          "derm7pt_manually": [0.6, 0.5, 0.5],
          "ph2derm7pt": [0.4, 0.9, 0.6],
          "ph2derm7pt_dlv3_ft": [0.4, 0.7, 0.6],
          "ph2derm7pt_manually": [0.4, 0.7, 0.6]}
```

and finally, run the script:

```python
python evaluate.py
```

A TXT file will be created with the results at `results/`.

### 2. Optimal Gamma Values

| Dataset             |  ResNet-101  | DenseNet-201 |  SEResNeXt  |
|:--------------------|:------------:|:------------:|:-----------:|
| PH2                 |     0.6      |     0.6      |     0.6     |  
| Derm7pt             |     0.3      |     0.7      |     0.3     | 
| PH2Derm7pt          |     0.4      |     0.9      |     0.6     | 
| PH2_DLV3            |     0.6      |     0.6      |     0.6     | 
| Derm7pt_DLV3        |     0.6      |     0.5      |     0.6     | 
| PH2Derm7pt_DLV3     |     0.4      |     0.7      |     0.6     | 
| PH2_Manually        |     0.6      |     0.6      |     0.6     |
| Derm7pt_Manually    |     0.6      |     0.5      |     0.5     |
| PH2Derm7pt_Manually |     0.4      |     0.7      |     0.6     |

### 3. Training a model

For training the model in a specified dataset, please ensure that you modify the directory paths and specify the parameters in model_params.py

Training a baseline model:

In `model_params.py` set `BASELINE = True`

In `model_training.py` choose the model: `models = ["resnet101"]  # "densenet201", "seresnext"` and uncomment baseline gammas:

```python
# BASELINE
gammas = {"ph2": [None, None, None],
          "ph2_dlv3_ft": [None, None, None],
          "ph2_manually": [None, None, None],
          "derm7pt": [None, None, None],
          "derm7pt_dlv3_ft": [None, None, None],
          "derm7pt_manually": [None, None, None],
          "ph2derm7pt": [None, None, None],
          "ph2derm7pt_dlv3_ft": [None, None, None],
          "ph2derm7pt_manually": [None, None, None]}
```

and finally, run the script:

```python
python model_training.py
```

Training the proposed model:

In `model_params.py` set `BASELINE = False`

In `model_training.py` choose the model: `models = ["resnet101"]  # "densenet201", "seresnext"` and uncomment OURS gammas:

```python
# OURS
gammas = {"ph2": [0.6, 0.6, 0.6],
          "ph2_dlv3_ft": [0.6, 0.6, 0.6],
          "ph2_manually": [0.6, 0.6, 0.6],
          "derm7pt": [0.3, 0.7, 0.3],
          "derm7pt_dlv3_ft": [0.6, 0.5, 0.6],
          "derm7pt_manually": [0.6, 0.5, 0.5],
          "ph2derm7pt": [0.4, 0.9, 0.6],
          "ph2derm7pt_dlv3_ft": [0.4, 0.7, 0.6],
          "ph2derm7pt_manually": [0.4, 0.7, 0.6]}
```

and finally, run the script:

```python
python model_training.py
```

### 4. Plot the results

In `evaluate.py` set the `plot_results` parameter to `True`.

