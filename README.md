---

````Markdown
# GEPMC-Loc: A Dynamic Gated Ensemble Network Fusing Pre-trained Language Models and Multi-scale Convolution for RNA Subcellular Localization

GEPMC-Loc is a deep learning framework designed to predict the subcellular localization of RNA sequences. It fuses advanced semantic features from pre-trained RNA language models (ERNIE-RNA and ProtRNA) with multi-scale convolutional neural networks and employs a dynamic gating mechanism to adaptively combine multi-source features for accurate prediction.

## 1. Project Structure

The project is organized as follows:
* **Source Code**: Implementation of the model architecture, training loops, and inference logic.
* **Dataset**: The benchmark dataset splits (`.pkl` files) are already included in the `data/` directory of this repository.
* **Features**: Large pre-trained feature embeddings (`.npy`) are hosted on Hugging Face due to size constraints.


## 2. Environment Setup

This project requires Python 3.9. We recommend using Conda to manage your environment for stability and reproducibility.

### 2.1 Install via environment.yaml
To replicate the experimental environment, run:
```bash
conda env create -f environment.yaml
conda activate GEPMC-Loc
````

### 2.2 Key Dependencies

The project utilizes the following core libraries:

Deep Learning: torch, optuna-dashboard

Data Science: pandas, numpy, scikit-learn

Others: pillow, packaging, olefile

## 3. Getting Started (Feature Preparation)

### 3.1 Download Pre-computed Features

If you wish to use our pre-extracted features directly, download them from Hugging Face:

Download Link: 📥 GEPMC-Loc-Embedding.zip

Instruction: Unzip the file and place ERNIE_RNA_embedding/ and ProTRNA_embedding/ folders in the root directory.

## 4. Usage (End-to-End Workflow)

To achieve end-to-end prediction from raw sequences, you need to integrate the source code of the underlying pre-trained models.

### Step 1: Clone and Configure Pre-trained Repositories

Download the official source code for the two RNA language models:

ERNIE-RNA: GitHub - Bruce-ywj/ERNIE-RNA

ProtRNA: GitHub - roxie-zhang/ProtRNA

Important: You must follow the instructions provided in the links above to configure the specific environments and download the model weights for both ERNIE-RNA and ProtRNA before proceeding.

### Step 2: Integrate Extraction Scripts

Copy the following two scripts from this repository into the respective directories:

Copy Extract_ERNIE-RNA_Embedding.py into the ERNIE-RNA directory.

Copy Extract_protRNA_Embedding.py into the ProtRNA directory.

Note: These scripts allow GEPMC-Loc to automatically invoke the pre-trained models for feature extraction during the process.

### Step 3: Configuration

Edit config.py to match your local paths. This file is the central control for the entire project:

Set ERNIE_CWD and PROTRNA_CWD to the absolute paths of the repositories prepared in Step 1.

Configure your local data paths and output directories for logs and models.

### Step 4: Training with 5-Fold Cross-Validation

To perform feature extraction and start the 5-fold training process:

```bash
python train_GEPMC_Loc.py
```

Logs: Saved in the log/ directory.

Model Weights: Best weights for each fold are saved in save_model/.

### Step 5: Inference and Evaluation (5-Fold Averaging)

To evaluate the model on the independent test set using the averaging strategy:

```bash
python predict.py
```

Prediction Strategy: This script implements a Cross-Validation Averaging strategy. It loads the best models saved from all 5 folds, performs inference on the test set with each, and calculates the average of the 5-fold results to produce the final evaluation metrics (ACC, MCC, Sn, Sp, etc.).

```

---
