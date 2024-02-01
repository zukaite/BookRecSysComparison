import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import pandas as pd


# Graph Convolution Layer
class GraphConv(tf.keras.layers.Layer):
    def __init__(self, adj_mat):
        super(GraphConv, self).__init__()
        self.adj_mat = adj_mat

    def build(self, input_shape):
        self.W = self.add_weight(
            "kernel", shape=[int(input_shape[-1]), int(input_shape[-1])]
        )

    def call(self, ego_embeddings):
        pre_embed = tf.sparse.sparse_dense_matmul(self.adj_mat, ego_embeddings)
        return tf.transpose(
            tf.matmul(self.W, pre_embed, transpose_a=False, transpose_b=True)
        )


# NGCF Model
class NGCF(tf.keras.Model):
    def __init__(
        self, adj_mat, R, n_users, n_items, n_layers=3, emb_dim=64, decay=0.0001
    ):
        super(NGCF, self).__init__()
        self.adj_mat = adj_mat
        self.R = R
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.decay = decay

        # Initialize user and item embeddings.
        initializer = tf.keras.initializers.GlorotNormal()
        self.user_embedding = tf.Variable(
            initializer([self.n_users, self.emb_dim]), name="user_embedding"
        )
        self.item_embedding = tf.Variable(
            initializer([self.n_items, self.emb_dim]), name="item_embedding"
        )

        # Stack graph convolutional layers.
        self.gcn = []
        for layer in range(n_layers):
            self.gcn.append(GraphConv(adj_mat))
            self.gcn.append(tf.keras.layers.LeakyReLU())

    def call(self, user_emb, item_emb):
        output_embeddings = tf.concat([user_emb, item_emb], axis=0)
        all_embeddings = [output_embeddings]

        # Graph convolutions.
        for i in range(0, self.n_layers):
            output_embeddings = self.gcn[i](output_embeddings)
            all_embeddings += [output_embeddings]

        # Compute the mean of all layers
        all_embeddings = tf.stack(all_embeddings, axis=1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)

        # Split into users and items embeddings
        new_user_embeddings, new_item_embeddings = tf.split(
            all_embeddings, [self.n_users, self.n_items], axis=0
        )

        return new_user_embeddings, new_item_embeddings

    def recommend(self, users, k):
        new_user_embed, new_item_embed = self(self.user_embedding, self.item_embedding)
        user_embed = tf.nn.embedding_lookup(new_user_embed, users)
        test_scores = tf.matmul(user_embed, new_item_embed, transpose_b=True)
        test_scores = np.array(test_scores)

        # Removing books already seen
        test_scores += sp.csr_matrix(self.R)[users, :] * -np.inf

        top_items = []
        top_scores = []
        for user_score in test_scores:
            user_score = np.squeeze(np.array(user_score))  # Ensuring 1D
            k_adj = min(k, len(user_score))
            top_idx = np.argpartition(user_score, -k_adj)[-k_adj:]
            top_items.extend(top_idx)
            top_scores.extend(user_score[top_idx])

        # Creating DataFrame
        topk_scores = pd.DataFrame(
            {
                "user_id": np.repeat(users, k),
                "book_name": np.array(top_items),
                "prediction": np.array(top_scores),
            }
        )

        return topk_scores
