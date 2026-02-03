import gc
import random
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from clac_metric import cv_model_evaluate
from utils import *
from model import HDGATModel, HDGATModel2, HDGAT_BioBERT_Model
from opt import Optimizer

from transformers import BertTokenizer

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

tf.compat.v1.disable_eager_execution()

# ================= TEXT ENCODING =================
MAX_LEN = 128

tokenizer = BertTokenizer.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1"
)

def encode_texts(texts, max_len=MAX_LEN):
    enc = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='np'
    )
    return enc['input_ids'], enc['attention_mask']


# =================================================
def PredictScore(
    train_drug_dis_matrix,
    drug_matrix,
    dis_matrix,
    drug_texts,
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

    # -------- BioBERT Inputs --------
    input_ids, attention_mask = encode_texts(drug_texts)

    # -------- Placeholders --------
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'adjdp': tf.compat.v1.placeholder_with_default(0., shape=()),

        # 🔹 BioBERT
        'input_ids': tf.compat.v1.placeholder(tf.int32, [None, MAX_LEN]),
        'attention_mask': tf.compat.v1.placeholder(tf.int32, [None, MAX_LEN])
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

    model3 = None

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

        opt3 = None

    # -------- Training --------
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    print("\n=== Training HDGAT / HDSAGE ===")

    for epoch in range(epochs):
        feed_dict = {
            placeholders['features']: features,
            placeholders['adj']: adj_norm,
            placeholders['adj_orig']: adj_orig,
            placeholders['dropout']: dp,
            placeholders['adjdp']: adjdp,
            placeholders['input_ids']: input_ids,
            placeholders['attention_mask']: attention_mask
        }

        _, loss1 = sess.run([opt1.opt_op, opt1.cost], feed_dict)
        _, loss2 = sess.run([opt2.opt_op, opt2.cost], feed_dict)
        loss3 = None

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch+1:04d} | "
                f"HDGAT={loss1:.5f} | "
                f"HDSAGE={loss2:.5f}"
            )

    print("\n=== Training Finished ===")

    # -------- Prediction --------
    feed_dict[placeholders['dropout']] = 0
    feed_dict[placeholders['adjdp']] = 0

    res1 = sess.run(model1.reconstructions, feed_dict)
    res2 = sess.run(model2.reconstructions, feed_dict)
    res3 = None

    sess.close()
    return res1, res2, res3


# =================================================
def cross_validation_experiment(
    drug_dis_matrix,
    drug_matrix,
    dis_matrix,
    drug_texts,
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

        res1, res2, res3 = PredictScore(
            train_matrix,
            drug_matrix,
            dis_matrix,
            drug_texts,
            seed,
            epochs,
            emb_dim,
            dp,
            lr,
            adjdp
        )

        pred1 = res1.reshape(drug_len, dis_len)
        pred2 = res2.reshape(drug_len, dis_len)
        pred3 = None

        metric_tmp1 = cv_model_evaluate(drug_dis_matrix, pred1, train_matrix)
        metric_tmp2 = cv_model_evaluate(drug_dis_matrix, pred2, train_matrix)
        metric_tmp3 = None

        print("HDGAT:", metric_tmp1)
        print("HDSAGE:", metric_tmp2)
        print("HDGAT+BioBERT: disabled")

        metric1 += metric_tmp1
        metric2 += metric_tmp2
        # metric3 disabled

        del train_matrix
        gc.collect()

    print("\n=== CV Finished ===")
    print("HDGAT avg:", metric1 / k_folds)
    print("HDSAGE avg:", metric2 / k_folds)
    print("HDGAT+BioBERT avg: disabled")

    return metric1 / k_folds, metric2 / k_folds, metric3 / k_folds


# =================================================
if __name__ == "__main__":
    drug_sim = np.loadtxt('../data/drug_sim_small.csv', delimiter=',')
    dis_sim = np.loadtxt('../data/dis_sim_small.csv', delimiter=',')
    drug_dis_matrix = np.loadtxt('../data/drug_disease_small.csv', delimiter=',')

    # 🔹 ONE text per drug (same order as matrix rows)
    drug_texts = [
        "Paracetamol is used to treat fever and mild pain",
        "Ibuprofen is a non-steroidal anti-inflammatory drug",
        # ...
    ]

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
        drug_texts,
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
    print("HDGAT+BioBERT: disabled")
