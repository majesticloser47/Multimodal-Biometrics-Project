# Multimodal-Biometrics-Project

A project for exploring and implementing multimodal biometric authentication systems using a transformer model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project-Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project aims to build a set of multimodal biometric models, which will perform user identification and forgery detection tasks on a combination of signature and electroencephalography (EEG) data received from a user.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/majesticloser47/Multimodal-Biometrics-Project.git
   cd Multimodal-Biometrics-Project
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Open the Jupyter Notebook environment:

```bash
jupyter notebook
```

Please follow these steps
1. Download the dataset from https://zenodo.org/records/8332198. Extract the downloaded ZIP file to a folder of your choice.
2. Come back to the repository and open the .env file
3. Fill in the path to your downloaded dataset for the key DATASET_PATH.
4. If you want save models to or load models from another directory, you can change MODEL_SAVE_PATH or MODEL_LOAD_PATH respectively. The currently trained models are in the folder /models/ and are named appropriately.
5. DO NOT CHANGE THE VALUE FOR USER_IDS_MASTER_LIST_PATH.
6. Open a terminal in the root of the repository, and execute the command:
   ```bash
   pip install -r requirements.txt
   ```
   This will install all necessary dependencies for the notebook to run.
6. If you do not want to run the training logic and directly want to see the model's performance on validation data, open multimodal_final_nb_no_train.ipynb and simply run all cells. (NOTE: All other code pertaining to SHAP feature values, training loops and hyperparameter tuning cells have been commented out for ease of use. CAREFUL NOT TO RUN THESE, if you want to continue seeing the outputs of the cells).
7. If you want to run the training logic too, open multimodal_final_nb.ipynb. BE CAREFUL, there are cells which execute the SHAP library, and it is a compute expensive ibrary so kindly be careful. You can already see the outputs for those cells in case you wish to see, and if you plan on running please do make sure your GPU has enough mmemory to spare.

## Project Structure

```
Multimodal-Biometrics-Project/
├── models/                                     # Pre-trained PyTorch models
├── notebooks/                                  # Jupyter notebooks for experiments and demos
├── multimodal_final_nb_no_train.ipynb          # Source code without training
├── multimodal_final_nb.ipynb                   # Source code with training
├── sign_eeg_feature_extraction.ipynb           # Source code, but with all raw trials
├── requirements.txt                            # Python dependencies
└── README.md                                   # Project documentation
```

## License

This project is licensed under the MIT License.

## Footnote
- Made by Abhay Premkumar Nambiar
- Link for downloading dataset: https://zenodo.org/records/8332198
