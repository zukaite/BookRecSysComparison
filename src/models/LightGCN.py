import numpy as np
import pandas as pd
import random
import scipy.sparse as sp
import tensorflow as tf

from tensorflow.keras.utils import Progbar


class GraphConv(tf.keras.layers.Layer):
    def __init__(self, adj_mat):
        super(GraphConv, self).__init__()
        self.adj_mat = adj_mat

    def call(self, ego_embeddings):
        return tf.sparse.sparse_dense_matmul(self.adj_mat, ego_embeddings)


class LightGCN(tf.keras.Model):
    def __init__(self, adj_mat, n_users, n_items, n_layers=3, emb_dim=64, decay=0.0001):
        super(LightGCN, self).__init__()
        self.adj_mat = adj_mat
        self.R = tf.sparse.to_dense(adj_mat)[:n_users, n_users:]
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

        # Stack light graph convolutional layers.
        self.gcn = [GraphConv(adj_mat) for layer in range(n_layers)]

    def call(self, inputs):
        user_emb, item_emb = inputs
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
        # Calculate the scores.
        new_user_embed, new_item_embed = self(
            (self.user_embedding, self.item_embedding)
        )
        user_embed = tf.nn.embedding_lookup(new_user_embed, users)
        test_scores = tf.matmul(
            user_embed, new_item_embed, transpose_a=False, transpose_b=True
        )
        test_scores = np.array(test_scores)

        # Remove movies already seen.
        test_scores += sp.csr_matrix(self.R)[users, :] * -np.inf

        # Get top movies.
        test_user_idx = np.arange(test_scores.shape[0])[:, None]
        top_items = np.argpartition(test_scores, -k, axis=1)[:, -k:]
        top_scores = test_scores[test_user_idx, top_items]
        sort_ind = np.argsort(-top_scores)
        top_items = top_items[test_user_idx, sort_ind]
        top_scores = top_scores[test_user_idx, sort_ind]
        top_items, top_scores = np.array(top_items), np.array(top_scores)

        # Create Dataframe with recommended movies.
        topk_scores = pd.DataFrame(
            {
                "user_id": np.repeat(users, top_items.shape[1]),
                "book_name": top_items.flatten(),
                "prediction": top_scores.flatten(),
            }
        )

        return topk_scores
