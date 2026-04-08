from layers import GraphAttentionConvolution, GraphConvolutionSparse,InnerProductDecoder,GraphAttentionBiLSTMConvolution
from utils import *
from keras.layers import LayerNormalization
from keras import layers
from layers import GraphSAGEConvolution
from layers import GraphBERTLayer, BioBERTEncoder, GatedFusion


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
        self.att = tf.compat.v1.Variable(tf.compat.v1.constant([0.9, 0.45, 0.4]))
        self.num_r = num_r
        self.layer_norm = LayerNormalization(axis=1)

        with tf.compat.v1.variable_scope(self.name):
            self.build()

    def build(self):
        # Apply dropout to adjacency
        self.adj = dropout_sparse(self.adj, 1 - self.adjdp, self.adj_nonzero)

        # ----- GCN → BiLSTM → GAT -----
        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act
        )(self.inputs)

        self.hidden2 = GraphAttentionBiLSTMConvolution(
            name='gat_dense_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout
        )(self.hidden1, self.adj)

        self.emb = GraphAttentionConvolution(
            name='gat_dense_layer1',
            former=self.hidden1,
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout
        )(self.hidden2, self.adj)

        # Weighted combination
        self.embeddings = (
            self.hidden1 * self.att[0] +
            self.hidden2 * self.att[1] +
            self.emb * self.att[2]
        )

        # ----- GraphBERT refinement -----
        # Smaller num_heads is safer to start (you can increase later)
        self.graphbert_layer1 = GraphBERTLayer(embed_dim=self.emb_dim, num_heads=2, dropout_rate=0.1)
        self.graphbert_layer2 = GraphBERTLayer(embed_dim=self.emb_dim, num_heads=2, dropout_rate=0.1)

        # Ensure embeddings are 2D: [N, D]
        x = self.embeddings  # [N, emb_dim]

        # --- Pass through GraphBERT blocks safely ---
        # The GraphBERTLayer handles rank automatically,
        # so no need to manually expand/squeeze.
        x = self.graphbert_layer1(x, training=True)
        x = self.graphbert_layer2(x, training=True)

        # Now x is [N, D] again — refined embeddings
        self.refined_embeddings = x

        # ----- Decoder -----
        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim,
            num_r=self.num_r,
            act=lambda x: x
        )(self.refined_embeddings)


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

        # 🔹 BiLSTM (unchanged)
        self.hidden_bilstm = GraphAttentionBiLSTMConvolution(
            name='gat_dense_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout
        )(self.hidden3, self.adj)

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
            act=lambda x: x
        )(self.embeddings)


class HDGAT_BioBERT_Model():
    def __init__(
        self,
        placeholders,
        num_features,
        emb_dim,
        features_nonzero,
        adj_nonzero,
        num_r,
        name,
        act=tf.compat.v1.nn.elu
    ):
        self.name = name

        # 🔹 Graph inputs
        self.inputs = placeholders['features']
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']

        # 🔹 Text inputs (ONLY for drugs)
        self.text_embeddings_input = placeholders['text_embeddings']

        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.num_r = num_r
        self.act = act

        self.att = tf.compat.v1.Variable(
            tf.compat.v1.constant([0.9, 0.45, 0.4])
        )

        with tf.compat.v1.variable_scope(self.name):
            self.build()

    def build(self):
        # =====================================================
        # 1️⃣ GRAPH ENCODER
        # =====================================================
        self.adj = dropout_sparse(self.adj, 1 - self.adjdp, self.adj_nonzero)

        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act
        )(self.inputs)

        self.hidden2 = GraphAttentionBiLSTMConvolution(
            name='gat_bilstm_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout
        )(self.hidden1, self.adj)

        self.hidden3 = GraphAttentionConvolution(
            name='gat_layer',
            former=self.hidden1,
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout
        )(self.hidden2, self.adj)

        # 🔹 Combined graph embeddings [num_nodes, emb_dim]
        self.graph_embeddings = (
            self.hidden1 * self.att[0] +
            self.hidden2 * self.att[1] +
            self.hidden3 * self.att[2]
        )
        self.graph_embeddings = tf.ensure_shape(
            self.graph_embeddings, [None, self.emb_dim]
        )

        self.graph_norm = LayerNormalization(axis=-1)
        self.graph_embeddings = tf.ensure_shape(
            self.graph_norm(self.graph_embeddings), [None, self.emb_dim]
        )

        # =====================================================
        # 2️⃣ BIOBERT TEXT ENCODER (DRUGS + DISEASES)
        # =====================================================
        self.text_embeddings = tf.ensure_shape(
            self.text_embeddings_input, [None, 768]
        )
        self.text_proj = tf.keras.layers.Dense(self.emb_dim)
        self.text_embeddings = self.text_proj(self.text_embeddings)
        self.text_norm = LayerNormalization(axis=-1)
        self.text_embeddings = tf.ensure_shape(
            self.text_norm(self.text_embeddings), [None, self.emb_dim]
        )

        # =====================================================
        # 3️⃣ FUSION (GRAPH + TEXT)
        # =====================================================
        self.fusion_layer = GatedFusion(self.emb_dim)
        self.fused_embeddings = self.fusion_layer(
            self.graph_embeddings, self.text_embeddings
        )
        self.fusion_gate = self.fusion_layer.last_gate
        self.fused_embeddings = tf.ensure_shape(
            self.fused_embeddings, [None, self.emb_dim]
        )

        # =====================================================
        # 4️⃣ LIGHTWEIGHT REFINEMENT
        # =====================================================
        self.refine_dense = tf.keras.layers.Dense(
            self.emb_dim, activation=tf.nn.elu
        )
        self.refine_norm = LayerNormalization(axis=-1)
        x = self.refine_dense(self.fused_embeddings)
        x = tf.nn.dropout(x, rate=self.dropout)
        x = self.refine_norm(x + self.fused_embeddings)

        self.refined_embeddings = tf.ensure_shape(
            x, [None, self.emb_dim]
        )

        # =====================================================
        # 5️⃣ DECODER
        # =====================================================
        self.reconstructions = InnerProductDecoder(
            name='decoder',
            input_dim=self.emb_dim,
            num_r=self.num_r,
            act=lambda x: x
        )(self.refined_embeddings)
