# GEPMC-Loc: A Dynamic Gated Ensemble Network Fusing Pre-trained Language Models and Multi-scale Convolution for RNA Subcellular Localization

GEPMC-Loc is a deep learning framework designed to predict the subcellular localization of RNA sequences. It fuses semantic features derived from pre-trained RNA language models (ERNIE-RNA and ProtRNA) with multi-scale convolutional neural networks, and employs a dynamic gating mechanism to adaptively combine multi-source features for accurate prediction.

---

## 1. Project Structure

The repository is organized as follows:

* **Source Code**
  Implementation of the model architecture, training pipeline, and inference procedures.

* **Dataset**
  Benchmark dataset splits (`.pkl` files) are provided in the `data/` directory.

* **Features**
  Pre-trained feature embeddings (`.npy`) are hosted externally due to size limitations.

---

## 2. Environment Setup

This project requires **Python 3.9**. It is recommended to use **Conda** for environment management.

### 2.1 Installation via `environment.yaml`

To reproduce the experimental environment, run:

```bash
conda env create -f environment.yaml
conda activate GEPMC-Loc
```

### 2.2 Key Dependencies

Core libraries used in this project include:

* **Deep Learning**: `torch`, `optuna-dashboard`
* **Data Processing**: `pandas`, `numpy`, `scikit-learn`
* **Others**: `pillow`, `packaging`, `olefile`

---

## 3. Getting Started (Feature Preparation)

### 3.1 Download Pre-computed Features

If you prefer to use pre-extracted features:

* **Download**: GEPMC-Loc-Embedding.zip
* **Instructions**:
  Unzip the file and place the following directories into the project root:

```
ERNIE_RNA_embedding/
ProTRNA_embedding/
```

---

## 4. Usage (End-to-End Workflow)

To perform end-to-end prediction starting from raw RNA sequences, you need to integrate external pre-trained models.

### Step 1: Clone and Configure Pre-trained Models

Download the official repositories:

* ERNIE-RNA
* ProtRNA

**Important:**
Follow the official instructions to configure environments and download model weights before proceeding.

---

### Step 2: Integrate Feature Extraction Scripts

Copy the following scripts into the corresponding directories:

* `Extract_ERNIE-RNA_Embedding.py` → ERNIE-RNA directory
* `Extract_protRNA_Embedding.py` → ProtRNA directory

These scripts enable automatic feature extraction during training.

---

### Step 3: Configuration

Edit the `config.py` file to match your local environment:

* Set:

  * `ERNIE_CWD` → path to ERNIE-RNA repository
  * `PROTRNA_CWD` → path to ProtRNA repository

* Configure:

  * dataset paths
  * log directories
  * model saving paths

---

### Step 4: Training (5-Fold Cross-Validation)

Run the following command to start feature extraction and training:

```bash
python train_GEPMC_Loc.py
```

* **Logs**: saved in `log/`
* **Model Weights**: saved in `save_model/` (best model for each fold)

---

### Step 5: Inference and Evaluation

To evaluate the model on the independent test set:

```bash
python predict.py
```

**Prediction Strategy:**

* Loads the best model from each of the 5 folds
* Performs inference independently
* Averages the predictions across all folds

**Evaluation Metrics:**

* Accuracy (ACC)
* Matthews Correlation Coefficient (MCC)
* Sensitivity (Sn)
* Specificity (Sp)
* and others

---

## 5. Notes

* Ensure that all external dependencies (ERNIE-RNA and ProtRNA) are correctly configured before training.
* Pre-computed features can significantly reduce runtime.
* The framework supports both **feature-based training** and **end-to-end extraction + training** workflows.

