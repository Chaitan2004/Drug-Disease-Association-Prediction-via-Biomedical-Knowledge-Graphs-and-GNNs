import gc
import random
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from clac_metric import cv_model_evaluate
from utils import *
from model import HDGATModel, HDGATModel2
from opt import Optimizer

tf.compat.v1.disable_eager_execution()


def PredictScore(train_drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    np.random.seed(seed)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)

    # Build graph & features
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

    # Placeholders
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'adjdp': tf.compat.v1.placeholder_with_default(0., shape=())
    }

    # ---- Model 1: HDGAT (BiLSTM + Attention) ----
    model1 = HDGATModel(
        placeholders, num_features, emb_dim,
        features_nonzero, adj_nonzero,
        train_drug_dis_matrix.shape[0],
        name='HDGAT'
    )

    # ---- Model 2: HDSAGE (GraphSAGE variant) ----
    model2 = HDGATModel2(
        placeholders, num_features, emb_dim,
        features_nonzero, adj_nonzero,
        train_drug_dis_matrix.shape[0],
        name='HDSAGE'
    )

    # Optimizers for both
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

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    print("\n=== Training HDGAT and HDSAGE Models ===")
    for epoch in range(epochs):
        feed_dict = {
            placeholders['features']: features,
            placeholders['adj']: adj_norm,
            placeholders['adj_orig']: adj_orig,
            placeholders['dropout']: dp,
            placeholders['adjdp']: adjdp
        }

        _, loss1 = sess.run([opt1.opt_op, opt1.cost], feed_dict=feed_dict)
        _, loss2 = sess.run([opt2.opt_op, opt2.cost], feed_dict=feed_dict)

        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1:04d}: HDGAT loss = {loss1:.5f}, HDSAGE loss = {loss2:.5f}")

    print("\n=== Optimization Finished ===")

    # Get final reconstructions
    feed_dict[placeholders['dropout']] = 0
    feed_dict[placeholders['adjdp']] = 0

    res1 = sess.run(model1.reconstructions, feed_dict=feed_dict)
    res2 = sess.run(model2.reconstructions, feed_dict=feed_dict)

    sess.close()

    return res1, res2


def cross_validation_experiment(drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    index_matrix = np.mat(np.where(drug_dis_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)

    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam % k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] += random_index[association_nam - association_nam % k_folds:]
    random_index = temp

    metric1 = np.zeros((1, 7))
    metric2 = np.zeros((1, 7))

    print(f"seed={seed}, evaluating drug-disease...")

    for k in range(k_folds):
        print(f"\n------ Fold {k + 1} ------")
        train_matrix = np.matrix(drug_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0

        drug_len, dis_len = drug_dis_matrix.shape
        res1, res2 = PredictScore(train_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp)

        # Evaluate both models
        pred1 = res1.reshape(drug_len, dis_len)
        pred2 = res2.reshape(drug_len, dis_len)

        metric_tmp1 = cv_model_evaluate(drug_dis_matrix, pred1, train_matrix)
        metric_tmp2 = cv_model_evaluate(drug_dis_matrix, pred2, train_matrix)

        print("HDGAT:", metric_tmp1)
        print("HDSAGE:", metric_tmp2)

        metric1 += metric_tmp1
        metric2 += metric_tmp2

        del train_matrix
        gc.collect()

    print("\n=== Cross-validation completed ===")
    print("HDGAT average:", metric1 / k_folds)
    print("HDSAGE average:", metric2 / k_folds)

    return metric1 / k_folds, metric2 / k_folds


if __name__ == "__main__":
    drug_sim = np.loadtxt('../data/drug_sim.csv', delimiter=',')
    dis_sim = np.loadtxt('../data/dis_sim.csv', delimiter=',')
    drug_dis_matrix = np.loadtxt('../data/therapeutic.txt')

    epoch = 400
    emb_dim = 64
    lr = 0.01
    adjdp = 0.6
    dp = 0.4
    simw = 6
    seed = 3407

    print("\n========== STARTING EXPERIMENT ==========")
    hdgat_results, hdsage_results = cross_validation_experiment(
        drug_dis_matrix, drug_sim * simw, dis_sim * simw, seed, epoch, emb_dim, dp, lr, adjdp
    )

    print("\nFinal Average Results:")
    print("HDGAT Model:", hdgat_results)
    print("HDSAGE Model:", hdsage_results)
