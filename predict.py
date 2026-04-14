# -*- coding: utf-8 -*-
import subprocess
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
import logging
from datetime import datetime
from typing import List, Tuple

from config import *
from model_GEPMC_Loc import GEPMC_Loc


# ==============================================================================
# 1. Utilities & Data Loading
# ==============================================================================

def setup_logging(rna_type: str):
    local_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    os.makedirs(local_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{rna_type}_Prediction_logs_{timestamp}.log"
    log_file_path = os.path.join(local_log_dir, log_file_name)

    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)
    logging.info(f"Prediction Logging configured. Log file: {log_file_path}")


def seed_torch(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_and_create_directories(dir_paths: List[str]):
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            logging.info(f"Directory not found. Creating: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)


def verify_files_exist(file_paths: List[str]):
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logging.error(f"CRITICAL ERROR: Required file not found: {file_path}")
            raise FileNotFoundError(f"Missing required file: {file_path}")


def evaluate(y_true, y_pred):
    y_true_np = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    acc = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    return acc, f1, precision, recall


class TriInputRNADataset(Dataset):
    def __init__(self, protrna_embeddings, ernierna_embeddings, sequences, labels):
        self.protrna_embeddings = protrna_embeddings
        self.ernierna_embeddings = ernierna_embeddings
        self.sequences = sequences
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.protrna_embeddings[idx], self.ernierna_embeddings[idx], self.sequences[idx], self.labels[idx]


def load_tri_input_data(protrna_path, ernierna_path, pkl_path):
    logging.info("--- Loading Triple Inputs for Prediction ---")
    try:
        protrna_emb = np.load(protrna_path)
        ernierna_emb = np.load(ernierna_path)
        with open(pkl_path, "rb") as f:
            sequences = pickle.load(f)
            _ = pickle.load(f)
            labels = pickle.load(f)
        return protrna_emb, ernierna_emb, sequences, labels
    except FileNotFoundError as e:
        logging.error(f"FATAL: Could not find an input file. Details: {e}")
        raise e


def run_external_script(conda_env, cwd, script_command):
    full_command = f"conda run -n {conda_env} {script_command}"
    logging.info(f"Executing in env [{conda_env}] at [{cwd}]: {full_command}")
    try:
        subprocess.run(full_command, shell=True, check=True, cwd=cwd)
        logging.info("Feature extraction completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to execute script. Error: {e}")
        raise e


def check_and_generate_features(pkl_path, protrna_path, ernierna_path, rna_type, protrna_cwd, ernie_cwd):
    logging.info(f"--- Checking Test Features for {rna_type} ---")
    if not os.path.exists(protrna_path):
        logging.info(f"ProtRNA Test feature missing. Generating...")
        run_external_script(CONDA_ENV_PROTRNA, protrna_cwd,
                            f"python Extract_protRNA_Embedding.py --pkl_path {pkl_path} --save_path {protrna_path}")
    if not os.path.exists(ernierna_path):
        logging.info(f"ERNIE-RNA Test feature missing. Generating...")
        run_external_script(CONDA_ENV_ERNIE, ernie_cwd,
                            f"python Extract_ERNIE-RNA_Embedding.py --pkl_path {pkl_path} --save_path {ernierna_path}")


# ==============================================================================
# 2. Reproduction Test Logic
# ==============================================================================

def test_reproduce(model, test_loader, device, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds_ens = []
    all_probs_ens = []
    all_targets = []

    with torch.no_grad():
        for protrna_x, ernierna_x, seqs, y in test_loader:
            protrna_x, ernierna_x, y = protrna_x.to(device).float(), ernierna_x.to(device).float(), y.to(device).long()

            l_ens, _, _ = model(protrna_x, ernierna_x, seqs)

            all_preds_ens.append(torch.argmax(l_ens, dim=1))
            all_probs_ens.append(F.softmax(l_ens, dim=1).cpu().numpy())
            all_targets.append(y)

    targets_cat = torch.cat(all_targets)
    preds_cat = torch.cat(all_preds_ens)
    probs_cat = np.concatenate(all_probs_ens, axis=0)

    acc, f1, prec, rec = evaluate(targets_cat, preds_cat)

    return (acc, f1, prec, rec), probs_cat, targets_cat.cpu().numpy()


# ==============================================================================
# 3. Main Execution
# ==============================================================================

def main():
    seed_torch(SEED)

    local_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save_model")

    check_and_create_directories([local_save_dir, PROTRNA_CWD, ERNIE_CWD])
    verify_files_exist([TEST_PKL_PATH])

    setup_logging(RNA_TYPE)

    try:
        check_and_generate_features(TEST_PKL_PATH, TEST_PROTRNA_PATH, TEST_ERNIERNA_PATH, RNA_TYPE, PROTRNA_CWD,
                                    ERNIE_CWD)
    except Exception as e:
        logging.error(f"Feature generation halted: {e}")
        return

    try:
        test_pro_emb, test_ern_emb, test_seqs, test_labels = load_tri_input_data(TEST_PROTRNA_PATH, TEST_ERNIERNA_PATH,
                                                                                 TEST_PKL_PATH)
    except Exception as e:
        logging.error(f"Failed during data loading: {e}")
        return

    test_dataset = TriInputRNADataset(test_pro_emb, test_ern_emb, test_seqs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    protrna_dim = test_pro_emb.shape[1]
    ernierna_dim = test_ern_emb.shape[1]
    output_dim = len(np.unique(test_labels))

    all_test_metrics = []
    all_fold_probs = []

    model = GEPMC_Loc(
        protrna_dim=protrna_dim,
        ernierna_dim=ernierna_dim,
        seq_max_len_limit=MAX_SEQ_LIMIT,
        num_classes=output_dim,
        compressed_dim=COMPRESSED_DIM,
        hidden_dim=HIDDEN_DIM,
        mlp_dropout=MLP_DROP,
        cnn_dropout=CNN_DROP,
        att_dropout=ATT_DROP
    ).to(DEVICE)

    logging.info(f"\n{'=' * 20} Starting Reproduction of Fold Models {'=' * 20}")

    for fold in range(1, N_SPLITS + 1):
        model_path = os.path.join(local_save_dir, f"{RNA_TYPE}_GEPMC_Loc_fold_{fold}.pth")

        if not os.path.exists(model_path):
            logging.error(f"Model not found: {model_path}")
            continue

        metrics_tuple, probs, targets_np = test_reproduce(model, test_loader, DEVICE, model_path)
        all_test_metrics.append(metrics_tuple)
        all_fold_probs.append(probs)

    if len(all_test_metrics) > 0:
        mean_test = np.mean(all_test_metrics, axis=0)
        std_test = np.std(all_test_metrics, axis=0)

        logging.info(f"\n{'=' * 20} Reproduced Final Summary {'=' * 20}")
        logging.info("-" * 40)
        logging.info(f"Test Set Accuracy:        {mean_test[0]:.3f}+-{std_test[0]:.3f}")
        logging.info(f"Test Set F1:              {mean_test[1]:.3f}+-{std_test[1]:.3f}")
        logging.info(f"Test Set Precision:       {mean_test[2]:.3f}+-{std_test[2]:.3f}")
        logging.info(f"Test Set Recall:          {mean_test[3]:.3f}+-{std_test[3]:.3f}")
        logging.info("-" * 40)

        avg_probs = np.mean(all_fold_probs, axis=0)
        final_preds = np.argmax(avg_probs, axis=1)
        logging.info("-" * 40)
        logging.info(f"Final Ensemble Test Predictions List:")
        logging.info(str(final_preds.tolist()))


if __name__ == '__main__':
    main()
