# config.py
import os
import torch

BASE_PROJECT_DIR = "/data/ccp/GEPMC-Loc"

PROTRNA_CWD = "/data/ccp/PythonProjects/ProtRNA"
ERNIE_CWD = "/data/ccp/PythonProjects/ERNIE-RNA"

PROTRNA_DIR = os.path.join(BASE_PROJECT_DIR, "ProTRNA_embedding", "33")
ERNIERNA_DIR = os.path.join(BASE_PROJECT_DIR, "ERNIE_RNA_embedding", "11")
MODEL_SAVE_DIR = os.path.join(BASE_PROJECT_DIR, "save_model")
LOG_DIR = os.path.join(BASE_PROJECT_DIR, "log")
DATA_DIR = os.path.join(BASE_PROJECT_DIR, "data")


# ==============================================================================
# lncRNA Parameter Configuration
# ==============================================================================
RNA_TYPE = 'lnc'
GPU_ID = 0
DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
N_SPLITS = 5

MAX_SEQ_LIMIT = 9216
COMPRESSED_DIM = 512
HIDDEN_DIM = 128

# ==============================================================================
# miRNA Parameter Configuration
# ==============================================================================
# RNA_TYPE = 'mi'
# GPU_ID = 0
# DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
#
# SEED = 42
# BATCH_SIZE = 64
# EPOCHS = 15
# LEARNING_RATE = 2e-4
# WEIGHT_DECAY = 1e-4
# N_SPLITS = 5
#
# MAX_SEQ_LIMIT = 9216
# COMPRESSED_DIM = 512
# HIDDEN_DIM = 128
# MLP_DROP = 0.20
# CNN_DROP = 0.45
# ATT_DROP = 0.20

# ==============================================================================
# circRNA Parameter Configuration
# ==============================================================================
# RNA_TYPE = 'circ'
# GPU_ID = 0
# DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
#
# SEED = 42
# BATCH_SIZE = 128
# EPOCHS = 40
# LEARNING_RATE = 2e-4
# WEIGHT_DECAY = 1e-4
# N_SPLITS = 5
#
# MAX_SEQ_LIMIT = 9216
# COMPRESSED_DIM = 512
# HIDDEN_DIM = 128


MLP_DROP = 0.00
CNN_DROP = 0.00
ATT_DROP = 0.00

CONDA_ENV_PROTRNA = "protrna"
CONDA_ENV_ERNIE = "ERNIE-RNA"

TRAIN_PROTRNA_PATH = os.path.join(PROTRNA_DIR, f"{RNA_TYPE}Train_33_embedding.npy")
TEST_PROTRNA_PATH = os.path.join(PROTRNA_DIR, f"{RNA_TYPE}Test_33_embedding.npy")
TRAIN_ERNIERNA_PATH = os.path.join(ERNIERNA_DIR, f"{RNA_TYPE}Train_11_embedding.npy")
TEST_ERNIERNA_PATH = os.path.join(ERNIERNA_DIR, f"{RNA_TYPE}Test_11_embedding.npy")
TRAIN_PKL_PATH = os.path.join(DATA_DIR, f"{RNA_TYPE}Train.pkl")
TEST_PKL_PATH = os.path.join(DATA_DIR, f"{RNA_TYPE}Test.pkl")