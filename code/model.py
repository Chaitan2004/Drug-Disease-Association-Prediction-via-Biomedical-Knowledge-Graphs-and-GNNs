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
            act=tf.compat.v1.nn.sigmoid
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
            act=tf.compat.v1.nn.sigmoid
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
        self.input_ids = placeholders['input_ids']
        self.attention_mask = placeholders['attention_mask']

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

        # =====================================================
        # 2️⃣ BIOBERT TEXT ENCODER (DRUGS ONLY)
        # =====================================================
        self.text_encoder = BioBERTEncoder(
            emb_dim=self.emb_dim,
            trainable=False
        )

        # Raw text embeddings → [num_drugs, emb_dim]
        text_embeddings_raw = self.text_encoder(
            self.input_ids,
            self.attention_mask
        )

        text_embeddings_raw = tf.ensure_shape(
            text_embeddings_raw, [None, self.emb_dim]
        )

        # =====================================================
        # 3️⃣ ALIGN TEXT WITH GRAPH NODES
        # =====================================================
        num_nodes = tf.shape(self.graph_embeddings)[0]
        num_drugs = tf.shape(text_embeddings_raw)[0]

        # Zero padding for disease nodes
        padding = tf.zeros(
            [num_nodes - num_drugs, self.emb_dim],
            dtype=text_embeddings_raw.dtype
        )

        # Final aligned text embeddings [num_nodes, emb_dim]
        self.text_embeddings = tf.concat(
            [text_embeddings_raw, padding],
            axis=0
        )

        self.text_embeddings = tf.ensure_shape(
            self.text_embeddings, [None, self.emb_dim]
        )

        # =====================================================
        # 4️⃣ FUSION (GRAPH + TEXT)
        # =====================================================
        self.fusion = GatedFusion(self.emb_dim)

        self.fused_embeddings = self.fusion(
            self.graph_embeddings,
            self.text_embeddings
        )

        self.fused_embeddings = tf.ensure_shape(
            self.fused_embeddings, [None, self.emb_dim]
        )

        # =====================================================
        # 5️⃣ GRAPHBERT REFINEMENT
        # =====================================================
        self.graphbert1 = GraphBERTLayer(
            embed_dim=self.emb_dim,
            num_heads=2,
            dropout_rate=0.1
        )

        self.graphbert2 = GraphBERTLayer(
            embed_dim=self.emb_dim,
            num_heads=2,
            dropout_rate=0.1
        )

        x = self.graphbert1(self.fused_embeddings, training=True)
        x = self.graphbert2(x, training=True)

        self.refined_embeddings = tf.ensure_shape(
            x, [None, self.emb_dim]
        )

        # =====================================================
        # 6️⃣ DECODER
        # =====================================================
        self.reconstructions = InnerProductDecoder(
            name='decoder',
            input_dim=self.emb_dim,
            num_r=self.num_r,
            act=tf.compat.v1.nn.sigmoid
        )(self.refined_embeddings)
