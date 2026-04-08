import os
import gc
import random
import numpy as np
import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import scipy.sparse as sp

from clac_metric import cv_model_evaluate
from utils import *
from model import HDGATModel, HDGATModel2, HDGAT_BioBERT_Model
from opt import Optimizer

from transformers import BertTokenizer, BertModel
from pathlib import Path
import pandas as pd

# ================= GPU SETUP =================
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU:", physical_devices[0].name)
    except RuntimeError as e:
        print("GPU error:", e)
else:
    print("No GPU found, using CPU")

# ================= TEXT ENCODING =================
MAX_LEN = 128
BIOBERT_HIDDEN_SIZE = 768
BIOBERT_BATCH_SIZE = 16

tokenizer = BertTokenizer.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1"
)

def encode_texts_with_biobert(texts, max_len=MAX_LEN, batch_size=BIOBERT_BATCH_SIZE):
    device = torch.device("cpu")
    print("BioBERT encoder device: CPU")
    model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model.to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(
                batch,
                padding='max_length',
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
            embeddings.append(cls_emb)
            print(f"BioBERT embedding progress: {min(start + batch_size, len(texts))}/{len(texts)}")

    del model

    return np.vstack(embeddings)


def pick_existing(*paths):
    for p in paths:
        if Path(p).exists():
            return p
    return paths[0]


def load_texts_aligned(vocab_path, texts_path, id_col, name_col, default_prefix):
    vocab = pd.read_csv(vocab_path)
    if Path(texts_path).exists():
        texts_df = pd.read_csv(texts_path)
        text_map = dict(zip(texts_df[id_col], texts_df["Text"]))
    else:
        text_map = {}

    texts = []
    for _, row in vocab.iterrows():
        key = row[id_col]
        text = text_map.get(key)
        if not text:
            text = f"{default_prefix}: {row[name_col]}."
        texts.append(text)
    return texts


def build_node_texts(drug_vocab_path, disease_vocab_path, drug_texts_path, disease_texts_path):
    # Order must match graph node order: all drugs first, then diseases
    drug_texts = load_texts_aligned(
        drug_vocab_path,
        drug_texts_path,
        id_col="ChemicalID",
        name_col="ChemicalName",
        default_prefix="Drug"
    )
    disease_texts = load_texts_aligned(
        disease_vocab_path,
        disease_texts_path,
        id_col="DiseaseID",
        name_col="DiseaseName",
        default_prefix="Disease"
    )
    return drug_texts + disease_texts


def assert_text_alignment(num_nodes, num_drugs, num_diseases, node_texts):
    if num_nodes != num_drugs + num_diseases:
        raise ValueError(
            f"Graph node count mismatch: num_nodes={num_nodes} "
            f"but num_drugs+num_diseases={num_drugs + num_diseases}."
        )
    if len(node_texts) != num_nodes:
        raise ValueError(
            f"Text count mismatch: node_texts={len(node_texts)} "
            f"but num_nodes={num_nodes}."
        )


# =================================================
def PredictScore(
    train_drug_dis_matrix,
    drug_matrix,
    dis_matrix,
    text_embeddings,
    seed,
    epochs,
    emb_dim,
    dp,
    lr,
    adjdp
):
    np.random.seed(seed)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)

    # -------- Graph Construction --------
    adj = constructHNet(train_drug_dis_matrix, drug_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)

    association_nam = train_drug_dis_matrix.sum()

    X = constructNet(train_drug_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    adj_orig = sparse_to_tuple(sp.csr_matrix(train_drug_dis_matrix.copy()))
    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]

    # -------- Placeholders --------
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'adjdp': tf.compat.v1.placeholder_with_default(0., shape=()),

        # 🔹 BioBERT
        'text_embeddings': tf.compat.v1.placeholder(tf.float32, [None, BIOBERT_HIDDEN_SIZE])
    }

    # -------- Models --------
    model1 = HDGATModel(
        placeholders, num_features, emb_dim,
        features_nonzero, adj_nonzero,
        train_drug_dis_matrix.shape[0],
        name='HDGAT'
    )

    model2 = HDGATModel2(
        placeholders, num_features, emb_dim,
        features_nonzero, adj_nonzero,
        train_drug_dis_matrix.shape[0],
        name='HDSAGE'
    )

    model3 = HDGAT_BioBERT_Model(
        placeholders, num_features, emb_dim,
        features_nonzero, adj_nonzero,
        train_drug_dis_matrix.shape[0],
        name='HDGAT_BioBERT'
    )

    # -------- Optimizers --------
    with tf.compat.v1.name_scope('optimizer'):
        opt1 = Optimizer(
            preds=model1.reconstructions,
            labels=tf.reshape(tf.sparse.to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model1,
            lr=lr,
            num_u=train_drug_dis_matrix.shape[0],
            num_v=train_drug_dis_matrix.shape[1],
            association_nam=association_nam
        )

        opt2 = Optimizer(
            preds=model2.reconstructions,
            labels=tf.reshape(tf.sparse.to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model2,
            lr=lr,
            num_u=train_drug_dis_matrix.shape[0],
            num_v=train_drug_dis_matrix.shape[1],
            association_nam=association_nam
        )

        opt3 = Optimizer(
            preds=model3.reconstructions,
            labels=tf.reshape(tf.sparse.to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model3,
            lr=lr,
            num_u=train_drug_dis_matrix.shape[0],
            num_v=train_drug_dis_matrix.shape[1],
            association_nam=association_nam
        )

    # -------- Training --------
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    print("\n=== Training HDGAT / HDSAGE / HDGAT+BioBERT ===")

    for epoch in range(epochs):
        feed_dict = {
            placeholders['features']: features,
            placeholders['adj']: adj_norm,
            placeholders['adj_orig']: adj_orig,
            placeholders['dropout']: dp,
            placeholders['adjdp']: adjdp,
            placeholders['text_embeddings']: text_embeddings
        }

        _, loss1 = sess.run([opt1.opt_op, opt1.cost], feed_dict)
        _, loss2 = sess.run([opt2.opt_op, opt2.cost], feed_dict)
        _, loss3 = sess.run([opt3.opt_op, opt3.cost], feed_dict)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch+1:04d} | "
                f"HDGAT={loss1:.5f} | "
                f"HDSAGE={loss2:.5f} | "
                f"HDGAT+BioBERT={loss3:.5f}"
            )

    print("\n=== Training Finished ===")

    # -------- Prediction --------
    feed_dict[placeholders['dropout']] = 0
    feed_dict[placeholders['adjdp']] = 0

    res1 = sess.run(model1.reconstructions, feed_dict)
    res2 = sess.run(model2.reconstructions, feed_dict)
    res3 = sess.run(model3.reconstructions, feed_dict)
    gate_values = sess.run(model3.fusion_gate, feed_dict)

    sess.close()
    return res1, res2, res3, gate_values


# =================================================
def cross_validation_experiment(
    drug_dis_matrix,
    drug_matrix,
    dis_matrix,
    text_embeddings,
    seed,
    epochs,
    emb_dim,
    dp,
    lr,
    adjdp
):
    index_matrix = np.mat(np.where(drug_dis_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()

    random.seed(seed)
    random.shuffle(random_index)

    k_folds = 5
    CV_size = int(association_nam / k_folds)

    temp = np.array(
        random_index[:association_nam - association_nam % k_folds]
    ).reshape(k_folds, CV_size, -1).tolist()

    temp[k_folds - 1] += random_index[association_nam - association_nam % k_folds:]
    random_index = temp

    metric1 = np.zeros((1, 7))
    metric2 = np.zeros((1, 7))
    metric3 = np.zeros((1, 7))

    for k in range(k_folds):
        print(f"\n------ Fold {k+1} ------")

        train_matrix = np.matrix(drug_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0

        drug_len, dis_len = drug_dis_matrix.shape

        res1, res2, res3, gate_values = PredictScore(
            train_matrix,
            drug_matrix,
            dis_matrix,
            text_embeddings,
            seed,
            epochs,
            emb_dim,
            dp,
            lr,
            adjdp
        )

        pred1 = res1.reshape(drug_len, dis_len)
        pred2 = res2.reshape(drug_len, dis_len)
        pred3 = res3.reshape(drug_len, dis_len)

        print(
            "Score ranges | "
            f"HDGAT=({pred1.min():.4f}, {pred1.max():.4f}) | "
            f"HDSAGE=({pred2.min():.4f}, {pred2.max():.4f}) | "
            f"HDGAT+BioBERT=({pred3.min():.4f}, {pred3.max():.4f})"
        )
        print(
            "Fusion gate | "
            f"mean={gate_values.mean():.4f} | "
            f"min={gate_values.min():.4f} | "
            f"max={gate_values.max():.4f}"
        )

        metric_tmp1 = cv_model_evaluate(drug_dis_matrix, pred1, train_matrix)
        metric_tmp2 = cv_model_evaluate(drug_dis_matrix, pred2, train_matrix)
        metric_tmp3 = cv_model_evaluate(drug_dis_matrix, pred3, train_matrix)

        print("HDGAT:", metric_tmp1)
        print("HDSAGE:", metric_tmp2)
        print("HDGAT+BioBERT:", metric_tmp3)

        metric1 += metric_tmp1
        metric2 += metric_tmp2
        metric3 += metric_tmp3

        del train_matrix
        gc.collect()

    print("\n=== CV Finished ===")
    print("HDGAT avg:", metric1 / k_folds)
    print("HDSAGE avg:", metric2 / k_folds)
    print("HDGAT+BioBERT avg:", metric3 / k_folds)

    return metric1 / k_folds, metric2 / k_folds, metric3 / k_folds


# =================================================
if __name__ == "__main__":
    drug_sim = np.loadtxt('../data/drug_sim_small.csv', delimiter=',')
    dis_sim = np.loadtxt('../data/dis_sim_small.csv', delimiter=',')
    drug_dis_matrix = np.loadtxt('../data/drug_disease_small.csv', delimiter=',')

    drug_vocab_path = pick_existing('../data/drug_vocab_small.csv', '../data/drug_vocab.csv')
    disease_vocab_path = pick_existing('../data/disease_vocab_small.csv', '../data/disease_vocab.csv')
    drug_texts_path = pick_existing('../data/drug_texts_small.csv', '../data/drug_texts.csv')
    disease_texts_path = pick_existing('../data/disease_texts_small.csv', '../data/disease_texts.csv')

    node_texts = build_node_texts(
        drug_vocab_path,
        disease_vocab_path,
        drug_texts_path,
        disease_texts_path
    )

    num_drugs = drug_dis_matrix.shape[0]
    num_diseases = drug_dis_matrix.shape[1]
    num_nodes = num_drugs + num_diseases
    assert_text_alignment(num_nodes, num_drugs, num_diseases, node_texts)
    print(f"Text alignment OK: nodes={num_nodes}, texts={len(node_texts)}")

    print("Building BioBERT embeddings once before TF training...")
    node_text_embeddings = encode_texts_with_biobert(node_texts)
    print(f"BioBERT embeddings ready: {node_text_embeddings.shape}")

    tf.compat.v1.disable_eager_execution()

    epoch = 400
    emb_dim = 64
    lr = 0.01
    adjdp = 0.6
    dp = 0.4
    simw = 6
    seed = 3407

    print("\n========== START ==========")

    hdgat_res, hdsage_res, biobert_res = cross_validation_experiment(
        drug_dis_matrix,
        drug_sim * simw,
        dis_sim * simw,
        node_text_embeddings,
        seed,
        epoch,
        emb_dim,
        dp,
        lr,
        adjdp
    )

    print("\nFINAL RESULTS")
    print("HDGAT:", hdgat_res)
    print("HDSAGE:", hdsage_res)
    print("HDGAT+BioBERT:", biobert_res)
