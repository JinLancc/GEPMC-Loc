import pickle
import numpy as np
import tensorflow as tf
import os
import argparse
from tqdm import tqdm
from pretrained import load_pretrained_model


def load_sequences_from_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        sequences = pickle.load(f)
    return sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True, help="Input PKL path")
    parser.add_argument("--save_path", type=str, required=True, help="Output NPY path")
    parser.add_argument("--target_layer", type=int, default=33)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu_id", type=str, default="1")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU Error: {e}")

    model = load_pretrained_model(name='ProtRNA_pretrained')
    batch_converter = model.alphabet.get_batch_converter()

    sequences = load_sequences_from_pkl(args.pkl_path)
    all_sequence_embeddings = []

    for i in tqdm(range(0, len(sequences), args.batch_size)):
        batch_seqs = sequences[i:i + args.batch_size]
        seq_tokens = batch_converter(batch_seqs)
        results = model(seq_tokens, repr_layers=[args.target_layer])

        token_embeddings = results["representations"][args.target_layer]
        sequence_embeddings_tensor = tf.reduce_mean(token_embeddings, axis=1)
        all_sequence_embeddings.append(sequence_embeddings_tensor.numpy())

    final_embeddings = np.concatenate(all_sequence_embeddings, axis=0)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    np.save(args.save_path, final_embeddings)
    print(f"Saved to: {args.save_path}")


if __name__ == '__main__':
    main()

