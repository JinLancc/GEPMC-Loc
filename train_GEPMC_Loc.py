# -*- coding: utf-8 -*-
import subprocess
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
import logging
from tqdm import tqdm
from datetime import datetime
from typing import List, Tuple, Dict, Any

from config import *
from model_GEPMC_Loc import GEPMC_Loc


# ==============================================================================
# 1. Utilities
# ==============================================================================

def setup_logging(rna_type: str):
    local_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    os.makedirs(local_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{rna_type}_GEPMC_Loc_logs_{timestamp}.log"
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
    logging.info(f"Logging configured. Log file: {log_file_path}")


def seed_torch(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def class_distribution(labels):
    classes, counts = np.unique(labels, return_counts=True)
    distribution = counts / counts.sum()
    return dict(zip(classes, distribution))


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


# ==============================================================================
# 2. Losses & Evaluation
# ==============================================================================

def orthogonality_loss(feat1, feat2):
    f1_norm = F.normalize(feat1, p=2, dim=1)
    f2_norm = F.normalize(feat2, p=2, dim=1)
    return torch.mean(torch.abs(torch.sum(f1_norm * f2_norm, dim=1)))


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean', device='cpu'):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float).to(device)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = ((1 - pt) ** self.gamma) * BCE_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            F_loss = alpha_t * F_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        return torch.sum(F_loss) if self.reduction == 'sum' else F_loss


def evaluate(y_true, y_pred):
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    acc = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    return acc, f1, precision, recall


# ==============================================================================
# 3. Dataset & Data Loading
# ==============================================================================

class TriInputRNADataset(Dataset):
    def __init__(self, protrna_embeddings, ernierna_embeddings, sequences: List[str], labels: np.ndarray):
        self.protrna_embeddings = protrna_embeddings
        self.ernierna_embeddings = ernierna_embeddings
        self.sequences = sequences
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.protrna_embeddings[idx], self.ernierna_embeddings[idx], self.sequences[idx], self.labels[idx]


def load_tri_input_data(protrna_path, ernierna_path, pkl_path) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    logging.info("--- Loading Triple Inputs ---")
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


def check_and_generate_features(pkl_path, protrna_path, ernierna_path, rna_type, protrna_cwd, ernie_cwd, is_train=True):
    dataset_type = "Train" if is_train else "Test"
    logging.info(f"--- Checking {dataset_type} Features for {rna_type} ---")

    if not os.path.exists(protrna_path):
        logging.info(f"ProtRNA {dataset_type} feature missing. Generating...")
        cmd = f"python Extract_protRNA_Embedding.py --pkl_path {pkl_path} --save_path {protrna_path}"
        run_external_script(CONDA_ENV_PROTRNA, protrna_cwd, cmd)

    if not os.path.exists(ernierna_path):
        logging.info(f"ERNIE-RNA {dataset_type} feature missing. Generating...")
        cmd = f"python Extract_ERNIE-RNA_Embedding.py --pkl_path {pkl_path} --save_path {ernierna_path}"
        run_external_script(CONDA_ENV_ERNIE, ernie_cwd, cmd)


# ==============================================================================
# 4. Core Training Logic
# ==============================================================================

def train_fold(model, train_loader, val_loader, full_dataset, train_idx, device, epochs, lr, weight_decay, model_path,
               rna_type, fold_idx):
    if rna_type == 'lnc':
        train_labels = full_dataset.labels[train_idx]
        dist = class_distribution(train_labels)
        sorted_classes = sorted(dist.keys())
        weights = [1.0 / dist[c] for c in sorted_classes]
        normalized_weights = [w / sum(weights) for w in weights]
        criterion = FocalLoss(alpha=normalized_weights, gamma=1.5, device=device)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_ensemble_f1 = 0.0
    best_metrics_summary = (0.0, 0.0, 0.0, 0.0)

    epoch_pbar = tqdm(range(epochs), desc=f"Fold {fold_idx} Progress", leave=False)

    for epoch in epoch_pbar:
        model.train()
        train_loss = 0

        for protrna_x, ernierna_x, seqs, y in train_loader:
            protrna_x, ernierna_x, y = protrna_x.to(device).float(), ernierna_x.to(device).float(), y.to(device).long()

            optimizer.zero_grad()

            l_ens, (l_prot, l_ernie, l_seq), (f_prot, f_ernie, f_seq) = model(protrna_x, ernierna_x, seqs)

            loss_ens = criterion(l_ens, y)
            loss_prot = criterion(l_prot, y)
            loss_ernie = criterion(l_ernie, y)
            loss_seq = criterion(l_seq, y)
            loss_aux = loss_prot + loss_ernie + loss_seq

            loss_diff = orthogonality_loss(f_prot, f_ernie) + \
                        orthogonality_loss(f_prot, f_seq) + \
                        orthogonality_loss(f_ernie, f_seq)

            loss = loss_ens + 0.5 * loss_aux + 0.1 * loss_diff

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        preds_ens = []
        all_targets = []

        with torch.no_grad():
            for protrna_x, ernierna_x, seqs, y in val_loader:
                protrna_x, ernierna_x, y = protrna_x.to(device).float(), ernierna_x.to(device).float(), y.to(
                    device).long()

                l_ens, _, _ = model(protrna_x, ernierna_x, seqs)
                preds_ens.append(torch.argmax(l_ens, dim=1))
                all_targets.append(y)

        targets_cat = torch.cat(all_targets)
        preds_cat = torch.cat(preds_ens)

        acc, f1, prec, rec = evaluate(targets_cat, preds_cat)

        if f1 > best_ensemble_f1:
            best_ensemble_f1 = f1
            best_metrics_summary = (acc, f1, prec, rec)
            torch.save(model.state_dict(), model_path)

        epoch_pbar.set_postfix(
            {'Train Loss': f'{avg_train_loss:.4f}', 'Cur F1': f'{f1:.3f}', 'Best F1': f'{best_ensemble_f1:.3f}'})

    acc, f1, prec, rec = best_metrics_summary
    logging.info(f"--- Fold {fold_idx} Best Validation State ---")
    logging.info(f"  Ensemble | Acc: {acc:.3f} | F1: {f1:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f}")

    return best_metrics_summary


def test(model, test_loader, device, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for protrna_x, ernierna_x, seqs, y in test_loader:
            protrna_x, ernierna_x, y = protrna_x.to(device).float(), ernierna_x.to(device).float(), y.to(device).long()

            l_ens, _, _ = model(protrna_x, ernierna_x, seqs)

            all_logits.append(l_ens)
            all_targets.append(y)

    targets_cat = torch.cat(all_targets)
    logits_cat = torch.cat(all_logits)
    preds_cat = torch.argmax(logits_cat, dim=1)

    acc, f1, prec, rec = evaluate(targets_cat, preds_cat)

    return (acc, f1, prec, rec), logits_cat


# ==============================================================================
# 5. Main Execution
# ==============================================================================

def main():
    seed_torch(SEED)

    local_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save_model")

    check_and_create_directories([local_save_dir, PROTRNA_CWD, ERNIE_CWD])
    verify_files_exist([TRAIN_PKL_PATH, TEST_PKL_PATH])

    setup_logging(RNA_TYPE)

    try:
        check_and_generate_features(
            TRAIN_PKL_PATH, TRAIN_PROTRNA_PATH, TRAIN_ERNIERNA_PATH, RNA_TYPE,
            protrna_cwd=PROTRNA_CWD, ernie_cwd=ERNIE_CWD, is_train=True
        )
        check_and_generate_features(
            TEST_PKL_PATH, TEST_PROTRNA_PATH, TEST_ERNIERNA_PATH, RNA_TYPE,
            protrna_cwd=PROTRNA_CWD, ernie_cwd=ERNIE_CWD, is_train=False
        )
    except Exception as e:
        logging.error(f"Feature generation halted: {e}")
        return

    try:
        train_pro_emb, train_ern_emb, train_seqs, train_labels = load_tri_input_data(
            TRAIN_PROTRNA_PATH, TRAIN_ERNIERNA_PATH, TRAIN_PKL_PATH
        )
        test_pro_emb, test_ern_emb, test_seqs, test_labels = load_tri_input_data(
            TEST_PROTRNA_PATH, TEST_ERNIERNA_PATH, TEST_PKL_PATH
        )
    except Exception as e:
        logging.error(f"Failed during data loading: {e}")
        return

    protrna_dim = train_pro_emb.shape[1]
    ernierna_dim = train_ern_emb.shape[1]
    output_dim = len(np.unique(train_labels))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    full_dataset = TriInputRNADataset(train_pro_emb, train_ern_emb, train_seqs, train_labels)
    test_dataset = TriInputRNADataset(test_pro_emb, test_ern_emb, test_seqs, test_labels)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_best_val_metrics = []
    all_test_metrics = []

    test_ensemble_logits = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(full_dataset)), full_dataset.labels)):
        logging.info(f"\n{'=' * 20} Fold {fold + 1}/{N_SPLITS} {'=' * 20}")

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

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

        if fold == 0:
            logging.info("================ MODEL ARCHITECTURE ================")
            logging.info(f"{model.__class__.__name__}(")

            def log_framework(mod, ind=1):
                for n, c in mod.named_children():
                    logging.info("  " * ind + f"({n}): {c.__class__.__name__}")
                    log_framework(c, ind + 1)

            log_framework(model)
            logging.info(")")
            logging.info("====================================================")

        model_path = os.path.join(local_save_dir, f"{RNA_TYPE}_GEPMC_Loc_fold_{fold + 1}.pth")

        best_val = train_fold(
            model, train_loader, val_loader, full_dataset, train_idx, DEVICE,
            EPOCHS, LEARNING_RATE, WEIGHT_DECAY, model_path, RNA_TYPE, fold + 1
        )
        all_best_val_metrics.append(best_val)

        test_res, test_logits = test(model, test_loader, DEVICE, model_path)
        all_test_metrics.append(test_res)

        if test_ensemble_logits is None:
            test_ensemble_logits = test_logits.clone()
        else:
            test_ensemble_logits += test_logits

    logging.info(f"\n{'=' * 20} Final Summary {'=' * 20}")

    mean_val = np.mean(all_best_val_metrics, axis=0)
    std_val = np.std(all_best_val_metrics, axis=0)
    logging.info(f"Cross-Val Best Accuracy:  {mean_val[0]:.3f}+-{std_val[0]:.4f}")
    logging.info(f"Cross-Val Best F1:        {mean_val[1]:.3f}+-{std_val[1]:.4f}")
    logging.info(f"Cross-Val Best Precision: {mean_val[2]:.3f}+-{std_val[2]:.4f}")
    logging.info(f"Cross-Val Best Recall:    {mean_val[3]:.3f}+-{std_val[3]:.4f}")

    mean_test = np.mean(all_test_metrics, axis=0)
    std_test = np.std(all_test_metrics, axis=0)
    logging.info("-" * 40)
    logging.info(f"Test Set Accuracy:        {mean_test[0]:.3f}+-{std_test[0]:.4f}")
    logging.info(f"Test Set F1:              {mean_test[1]:.3f}+-{std_test[1]:.4f}")
    logging.info(f"Test Set Precision:       {mean_test[2]:.3f}+-{std_test[2]:.4f}")
    logging.info(f"Test Set Recall:          {mean_test[3]:.3f}+-{std_test[3]:.4f}")

    final_test_preds = torch.argmax(test_ensemble_logits, dim=1).cpu().numpy().tolist()
    logging.info("-" * 40)
    logging.info(f"Final Ensemble Test Predictions List:")
    logging.info(str(final_test_preds))


if __name__ == '__main__':
    main()
