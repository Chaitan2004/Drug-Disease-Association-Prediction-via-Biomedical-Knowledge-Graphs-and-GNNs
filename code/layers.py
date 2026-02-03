
from utils import *
import tensorflow as tf
from keras.layers import LayerNormalization
from keras import layers
from keras.models import Sequential
from keras.layers import  Bidirectional,LSTM
from transformers import TFBertModel


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = tf.sqrt(6.0 / (tf.cast(input_dim + output_dim, tf.float32)))
    initial = tf.random.uniform([input_dim, output_dim], -init_range, init_range)
    return tf.Variable(initial, name=name)


class GraphAttentionBiLSTMConvolution():

    def __init__(self, input_dim, output_dim, name, dropout=0.2, concat=True, act=tf.nn.relu):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vars = {}
        self.issparse = False
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout

        self.alpha = 0.2
        self.concat = concat

        # TF1 cells for dynamic RNN (avoids Keras build-time shape checks)
        self.fwd_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=int(self.output_dim / 2))
        self.bwd_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=int(self.output_dim / 2))

        self.W = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(input_dim, output_dim)))
        self.a = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(2 * output_dim, 1)))
        self.leakyrelu = tf.keras.layers.LeakyReLU(self.alpha)
        self.layer = LayerNormalization(axis=1)
        self.act = act

    def __call__(self, h, adj, training=True):
        with tf.compat.v1.name_scope(self.name):
            # Decide 2D vs 3D at runtime
            rank_h = tf.rank(h)

            def _expand():    # [N, D] -> [1, N, D]
                return tf.expand_dims(h, axis=0)

            def _keep():
                return h

            h_expanded = tf.cond(tf.equal(rank_h, 2), _expand, _keep)

            # -------------------------
            # IMPORTANT FIX: attach a partial static shape so TF1's
            # bidirectional_dynamic_rnn internal checks don't fail.
            # We statically set feature dimension (last axis) to input_dim.
            # batch and time dims remain unknown (None).
            try:
                # set_shape works in graph mode; if it fails silently that's OK.
                h_expanded.set_shape([None, None, self.input_dim])
            except Exception:
                # keep going; set_shape may raise in some TF builds — not fatal
                pass
            # -------------------------

            # bidirectional dynamic rnn (TF1) accepts dynamic shapes at runtime
            (output_fw, output_bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                cell_fw=self.fwd_cell,
                cell_bw=self.bwd_cell,
                inputs=h_expanded,
                dtype=tf.float32,
                time_major=False
            )

            h_lstm_out = tf.concat([output_fw, output_bw], axis=-1)  # [batch, timesteps, output_dim]

            # If we expanded earlier, squeeze batch dim back
            def _squeeze():  # from [1, N, out_dim] -> [N, out_dim]
                return tf.squeeze(h_lstm_out, axis=0)

            def _keep2():
                return h_lstm_out

            h_after = tf.cond(tf.equal(rank_h, 2), _squeeze, _keep2)

            # If h_after is 3D (rare), collapse to last timestep to match attention expectation
            rank_wh = tf.rank(h_after)
            def _collapse():
                return h_after[:, -1, :]

            def _keep_wh():
                return h_after

            h_node = tf.cond(tf.equal(rank_wh, 3), _collapse, _keep_wh)

            # Attention & aggregation (same as original)
            Wh = tf.matmul(h_node, self.W)
            e = self._prepare_attentional_mechanism_input(Wh)
            zero_vec = -9e15 * tf.ones_like(e)

            adj_dense = tf.sparse.to_dense(tf.sparse.reorder(adj))
            adj_dense = adj_dense + tf.eye(tf.shape(adj_dense)[0], dtype=adj_dense.dtype)

            attention = tf.where(adj_dense > 0, e, zero_vec)
            attention = tf.nn.softmax(attention, axis=-1)
            attention = tf.nn.dropout(attention, rate=self.dropout)
            h_prime = tf.matmul(attention, Wh)

        return self.leakyrelu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        # maintain original behavior (reinit a)
        self.a = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(2 * self.output_dim, 1)))
        a_input = tf.concat([Wh, Wh], axis=-1)
        e = tf.matmul(self.leakyrelu(a_input), self.a)
        output = tf.squeeze(e, axis=-1)
        return output
        
class GraphAttentionConvolution():


    def __init__(self,former,input_dim, output_dim, name, dropout=0.2,concat=True,act=tf.nn.relu):
        self.name = name
        self.input_dim=input_dim
        self.former=former
        self.output_dim=output_dim
        self.vars = {}
        self.issparse = False
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.alpha = 0.2
        self.concat = concat

        self.W = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(input_dim, output_dim)))
        self.a = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(2 * output_dim, 1)))
        self.leakyrelu = tf.keras.layers.LeakyReLU(self.alpha)
        self.act = act
    def __call__(self, h, adj, training=True):
        with tf.compat.v1.name_scope(self.name):
            Wh = tf.matmul(h, self.W)
            e = self._prepare_attentional_mechanism_input(Wh)
            zero_vec = -9e15 * tf.ones_like(e)
            adj=tf.sparse.to_dense(tf.sparse.reorder(adj))
            adj = adj + tf.eye(tf.shape(adj)[0])
            attention = tf.where(adj > 0, e, zero_vec)
            attention = tf.nn.softmax(attention, axis=-1)
            attention = tf.nn.dropout(attention, rate=self.dropout)
            h_prime = tf.matmul(attention, Wh)
        return self.leakyrelu(h_prime)+self.former

    def _prepare_attentional_mechanism_input(self, Wh):
        self.a = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(2 * self.output_dim, 1)))
        a_input = tf.concat([Wh, Wh], axis=-1)
        e = tf.matmul(self.leakyrelu(a_input), self.a)
        output = tf.squeeze(e, axis=-1)

        return output
class GraphConvolutionSparse():

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.compat.v1.nn.relu):
        self.name = name
        self.vars = {}
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.compat.v1.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
            x = tf.compat.v1.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.compat.v1.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs
class InnerProductDecoder():


    def __init__(self, input_dim, name, num_r, dropout=0., act=tf.compat.v1.nn.sigmoid):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, input_dim, name='weights')

    def __call__(self, inputs):
        with tf.compat.v1.name_scope(self.name):
            inputs = tf.compat.v1.nn.dropout(inputs, 1-self.dropout)
            rank = tf.rank(inputs)
            inputs = tf.cond(
                tf.equal(rank, 1),
                lambda: tf.expand_dims(inputs, 0),
                lambda: inputs
            )

            # Defensive reshape for dynamic shapes
            inputs = tf.reshape(inputs, [-1, tf.shape(inputs)[-1]])

            R = inputs[0:self.num_r, :]
            D = inputs[self.num_r:, :]
            R = tf.compat.v1.matmul(R, self.vars['weights'])
            D = tf.compat.v1.transpose(D)
            x = tf.compat.v1.matmul(R, D)
            x = tf.compat.v1.reshape(x, [-1])
            outputs = self.act(x)
        return outputs
    
class GraphSAGEConvolution():
    """
    GraphSAGE Layer: Sample and Aggregate approach
    Uses mean aggregation similar to the GraphSAGE paper.
    """

    def __init__(self, input_dim, output_dim, name, dropout=0.2, act=tf.nn.relu):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vars = {}
        self.dropout = dropout
        self.act = act

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            # Weight for transforming concatenated [self || neighbor_agg]
            self.vars['weights'] = weight_variable_glorot(
                input_dim * 2, output_dim, name='weights')

    def __call__(self, inputs, adj, training=True):
        with tf.compat.v1.name_scope(self.name):
            h = inputs

            # ✅ Safely apply dropout (works with TF1 symbolic tensors)
            if training:
                if isinstance(h, tf.SparseTensor):
                    h = tf.sparse.to_dense(h)
                h = tf.nn.dropout(h, rate=tf.cast(self.dropout, tf.float32))

            # ✅ Convert adjacency to dense safely
            if isinstance(adj, tf.SparseTensor):
                adj = tf.sparse.to_dense(tf.sparse.reorder(adj))

            # ✅ Add self-loops
            adj = adj + tf.eye(tf.shape(adj)[0], dtype=adj.dtype)

            # ✅ Row-normalize adjacency
            deg = tf.reduce_sum(adj, axis=-1, keepdims=True)
            adj_norm = adj / (deg + 1e-10)

            # ✅ Mean neighbor aggregation
            neigh_agg = tf.matmul(adj_norm, h)

            # ✅ Concatenate self and neighbor embeddings
            h_concat = tf.concat([h, neigh_agg], axis=-1)

            # ✅ Linear transformation
            h_out = tf.matmul(h_concat, self.vars['weights'])

            # ✅ Activation
            outputs = self.act(h_out)

            return outputs


class GraphBERTLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=4, dropout_rate=0.1, name=None):
        super(GraphBERTLayer, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # ensure key_dim divides embed_dim
        if embed_dim % num_heads != 0:
            # fallback: use floor division but warn
            self.key_dim = embed_dim // num_heads
        else:
            self.key_dim = embed_dim // num_heads

        # Keras MHA expects key_dim = dimension per head
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            name=(name or "graphbert_mha")
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.embed_dim * 4, activation='relu'),
            tf.keras.layers.Dense(self.embed_dim)
        ], name=(name or "graphbert_ffn"))
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # x can be [N, D] (nodes as one sequence) or [batch, seq_len, D]
        # Convert to [batch, seq_len, D] safely where batch==1 if needed.

        # dynamic rank
        rank = tf.rank(x)
        def expand():          # when x is [N, D]
            return tf.expand_dims(x, axis=0)   # -> [1, N, D]
        def keep():            # when x is already [B, S, D]
            return x
        x_exp = tf.cond(tf.equal(rank, 2), expand, keep)

        # ensure static last dim known to MHA: give Keras some shape hints
        # If possible, set static shape (best-effort)
        try:
            x_exp = tf.ensure_shape(x_exp, [None, None, self.embed_dim])
        except Exception:
            # ensure_shape can raise if shapes unknown; ignore safely
            pass

        # Self-attention
        attn_output = self.att(x_exp, x_exp, training=training)  # [B, S, D]
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x_exp + attn_output)

        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)  # [B, S, D]

        # If original input was rank 2, return rank-2 [S, D], else return [B, S, D]
        def squeeze(): return tf.squeeze(out2, axis=0)
        def passthrough(): return out2
        out_final = tf.cond(tf.equal(rank, 2), squeeze, passthrough)

        return out_final


class BioBERTEncoder(tf.keras.layers.Layer):
    def __init__(self, emb_dim, trainable=False, **kwargs):
        super().__init__(**kwargs)

        self.biobert = TFBertModel.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.1",
            from_pt=True          # 🔥 critical fix
        )

        self.biobert.trainable = trainable
        self.proj = tf.keras.layers.Dense(emb_dim)

    def call(self, input_ids, attention_mask):
        outputs = self.biobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_emb = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        return self.proj(cls_emb)                     # [B, emb_dim]

class GatedFusion(tf.keras.layers.Layer):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.gate = tf.keras.layers.Dense(emb_dim, activation="sigmoid")

    def call(self, graph_emb, text_emb):
        # 🔥 Force static feature dimension for Keras Dense
        graph_emb = tf.ensure_shape(graph_emb, [None, self.emb_dim])
        text_emb  = tf.ensure_shape(text_emb,  [None, self.emb_dim])

        gate = self.gate(graph_emb)
        return gate * graph_emb + (1.0 - gate) * text_emb

