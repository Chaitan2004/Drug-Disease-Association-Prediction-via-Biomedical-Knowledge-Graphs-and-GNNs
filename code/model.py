from layers import GraphAttentionConvolution, GraphConvolutionSparse,InnerProductDecoder,GraphAttentionBiLSTMConvolution
from utils import *
from keras.layers import LayerNormalization
from keras import layers
from layers import GraphSAGEConvolution


class HDGATModel():
    def __init__(self, placeholders, num_features, emb_dim, features_nonzero, adj_nonzero, num_r, name, act=tf.compat.v1.nn.elu):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.act = act
        self.att = tf.compat.v1.Variable(tf.compat.v1.constant([0.9,0.45,0.4]))
        self.num_r = num_r
        self.layer_norm=LayerNormalization(axis=1)


        with tf.compat.v1.variable_scope(self.name):
            self.build()

    def build(self):
        self.adj = dropout_sparse(self.adj, 1-self.adjdp, self.adj_nonzero)

        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)


        self.hidden2 = GraphAttentionBiLSTMConvolution(
            name='gat_dense_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout,
            )(self.hidden1,self.adj)


        self.emb = GraphAttentionConvolution(
            name='gat_dense_layer1',
            former=self.hidden1,
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout
        )(self.hidden2, self.adj)



        self.embeddings=self.hidden1*self.att[0]+self.hidden2*self.att[1]+self.emb*self.att[2]

        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim, num_r=self.num_r, act=tf.compat.v1.nn.sigmoid)(self.embeddings)
        

class HDGATModel2():
    def __init__(self, placeholders, num_features, emb_dim, features_nonzero, adj_nonzero, num_r, name, act=tf.compat.v1.nn.elu):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.act = act
        self.att = tf.compat.v1.Variable(tf.compat.v1.constant([0.9,0.45,0.4]))
        self.num_r = num_r
        self.layer_norm = LayerNormalization(axis=1)

        with tf.compat.v1.variable_scope(self.name):
            self.build()

    def build(self):
    # Apply dropout to adjacency
        self.adj = dropout_sparse(self.adj, 1 - self.adjdp, self.adj_nonzero)

        # 🔹 3-layer GraphSAGE (to match GCN timestamps)
        self.hidden1 = GraphSAGEConvolution(
            name='sage_layer1',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout,
            act=self.act
        )(self.inputs, self.adj)

        self.hidden2 = GraphSAGEConvolution(
            name='sage_layer2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout,
            act=self.act
        )(self.hidden1, self.adj)

        self.hidden3 = GraphSAGEConvolution(
            name='sage_layer3',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout,
            act=self.act
        )(self.hidden2, self.adj)

        # 🔹 Collect embeddings [A1, A2, A3] for BiLSTM (like GCN)
        self.hidden_outputs = [self.hidden1, self.hidden2, self.hidden3]

        # Stack GraphSAGE outputs as timesteps [A1, A2, A3]
        hidden_seq = tf.stack([self.hidden1, self.hidden2, self.hidden3], axis=1)  # shape: [num_nodes, 3, emb_dim]

        # Pass through BiLSTM
        self.hidden_bilstm = GraphAttentionBiLSTMConvolution(
            name='gat_dense_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout
        )(hidden_seq, self.adj)

        # 🔹 Final graph attention layer
        self.emb = GraphAttentionConvolution(
            name='gat_dense_layer1',
            former=self.hidden2,
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout
        )(self.hidden_bilstm, self.adj)

        # 🔹 Weighted combination (same idea as before)
        self.embeddings = (
            self.hidden1 * self.att[0] +
            self.hidden2 * self.att[1] +
            self.emb * self.att[2]
        )

        # 🔹 Decoder (unchanged)
        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim,
            num_r=self.num_r,
            act=tf.compat.v1.nn.sigmoid
        )(self.embeddings)
