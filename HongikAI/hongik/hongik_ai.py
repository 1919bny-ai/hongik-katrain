# Copyright (c) 2025 BNY (Hongik AI Project)

# Implements the AI's 'brain', combining a CNN and Transformer for intuition,
# and Monte Carlo Tree Search (MCTS) for rational deliberation.
#
# Author: 박남영,Gemini 2.5 Pro

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

from hongik.board_ai import Board, IllegalMoveError


# ===================================================================
# 트랜스포머 부품들 
# 이 부분은 우리가 이전에 함께 만들었던 트랜스포머의 핵심 부품들입니다.
# 아빠의 설계 그대로 완벽하기에, 엄마는 손대지 않았어요.
# ===================================================================
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculates the attention scores, which is the core of the attention mechanism.
    It determines how much focus to place on other parts of the input sequence.
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(layers.Layer):
    """
    Implements the Multi-Head Attention mechanism. This allows the model to jointly attend
    to information from different representation subspaces at different positions,
    which is more powerful than single-head attention.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Splits the last dimension into (num_heads, depth)."""
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        """Processes the input tensors through the multi-head attention mechanism."""
        batch_size = tf.shape(q)[0]
        q = self.wq(q); k = self.wk(k); v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output

class PositionWiseFeedForwardNetwork(layers.Layer):
    """
    Implements the Position-wise Feed-Forward Network. This is applied to each
    position separately and identically. It consists of two linear transformations
    with a ReLU activation in between.
    """
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()
        self.dense_1 = layers.Dense(d_ff, activation='relu')
        self.dense_2 = layers.Dense(d_model)
    def call(self, inputs):
        return self.dense_2(self.dense_1(inputs))

class EncoderLayer(layers.Layer):
    """
    Represents one layer of the Transformer encoder. It consists of a multi-head
    attention mechanism followed by a position-wise feed-forward network.
    Includes dropout and layer normalization.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = PositionWiseFeedForwardNetwork(d_model=d_model, d_ff=d_ff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training, padding_mask=None):
        attn_output = self.mha(inputs, inputs, inputs, padding_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
        
def get_positional_encoding(max_seq_len, d_model):
    """
    Generates positional encodings. Since the model contains no recurrence or
    convolution, this is used to inject information about the relative or
    absolute position of the tokens in the sequence.
    """
    angle_rads = (np.arange(max_seq_len)[:, np.newaxis] / 
                  np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# ===================================================================
# 3. CNN + 트랜스포머 '직관' 엔진 
# ===================================================================
class CNNTransformerHybrid(Model):
    """
    The 'Intuition' engine, combining a 'Scout' (CNN) and a 'Commander' (Transformer).
    This version implements a lightweight head architecture using Squeeze-and-Excitation
    and Convolutional Heads for parameter efficiency and performance.
    """
    def __init__(self, num_transformer_layers, d_model, num_heads, d_ff, 
                 board_size=19, cnn_filters=128, dropout_rate=0.1):
        super(CNNTransformerHybrid, self).__init__()
        self.board_size = board_size
        self.d_model = d_model
        
        self.cnn_conv1 = layers.Conv2D(cnn_filters, 3, padding='same', activation='relu')
        self.cnn_bn1 = layers.BatchNormalization()
        self.cnn_conv2 = layers.Conv2D(d_model, 1, padding='same', activation='relu')
        self.cnn_bn2 = layers.BatchNormalization()
        self.reshape_to_seq = layers.Reshape((board_size * board_size, d_model))
        self.positional_encoding = get_positional_encoding(board_size * board_size, d_model)
        self.dropout = layers.Dropout(dropout_rate)
        self.transformer_encoder = [EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_transformer_layers)]
        self.reshape_to_2d = layers.Reshape((board_size, board_size, d_model))
        
        self.se_gap = layers.GlobalAveragePooling2D()
        self.se_reshape = layers.Reshape((1, 1, d_model))
        self.se_dense_1 = layers.Dense(d_model // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)
        self.se_dense_2 = layers.Dense(d_model, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        self.se_multiply = layers.Multiply()
        
        self.policy_conv = layers.Conv2D(filters=2, kernel_size=1, padding='same', activation='relu')
        self.policy_bn = layers.BatchNormalization()
        self.policy_flatten = layers.Flatten()
        self.policy_dense = layers.Dense(board_size * board_size + 1, name='policy_head')
        
        self.value_conv = layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='relu')
        self.value_bn = layers.BatchNormalization()
        self.value_flatten = layers.Flatten()
        self.value_dense1 = layers.Dense(256, activation='relu')
        self.value_dense2 = layers.Dense(1, activation='tanh', name='value_head')

    @tf.function(jit_compile=False)
    def call(self, inputs, training=False):
        x = self.cnn_conv1(inputs)
        x = self.cnn_bn1(x, training=training)
        x = self.cnn_conv2(x)
        cnn_output = self.cnn_bn2(x, training=training)

        x = self.reshape_to_seq(cnn_output)
        seq_len = tf.shape(x)[1]
        x += self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        for i in range(len(self.transformer_encoder)):
            x = self.transformer_encoder[i](x, training=training, padding_mask=None)
        
        transformer_output = self.reshape_to_2d(x)
    
        se = self.se_gap(transformer_output)
        se = self.se_reshape(se)
        se = self.se_dense_1(se)
        se = self.se_dense_2(se)
        se_output = self.se_multiply([transformer_output, se])

        ph = self.policy_conv(se_output)
        ph = self.policy_bn(ph, training=training)
        ph = self.policy_flatten(ph)
        policy_logits = self.policy_dense(ph)

        vh = self.value_conv(se_output)
        vh = self.value_bn(vh, training=training)
        vh = self.value_flatten(vh)
        vh = self.value_dense1(vh)
        value = self.value_dense2(vh)
        return policy_logits, value

# ===================================================================
# 4. MCTS '이성' 엔진 
# ===================================================================
class MCTSNode:
    """
    Represents a single node in the Monte Carlo Tree Search. Each node stores
    statistics like visit count (n_visits), total action value (q_value), and
    prior probability (p_sa).
    """
    def __init__(self, parent=None, prior_p=1.0):
        self.parent, self.children, self.n_visits, self.q_value, self.p_sa = parent, {}, 0, 0, prior_p
        self.C_PUCT_BASE, self.C_PUCT_INIT = 19652, 1.25
        
    def select(self, root_n_visits):
        """
        Selects the child node with the highest Upper Confidence Bound (UCB) score.
        This balances exploration and exploitation during the search.
        """
        dynamic_c_puct = np.log((1 + root_n_visits + self.C_PUCT_BASE) / self.C_PUCT_BASE) + self.C_PUCT_INIT
        return max(self.children.items(), 
                   key=lambda item: item[1].q_value + dynamic_c_puct * item[1].p_sa * np.sqrt(self.n_visits) / (1 + item[1].n_visits))

    def expand(self, action_probs):
        """
        Expands a leaf node by creating new child nodes for all legal moves,
        initializing their statistics from the prior probabilities given by the
        neural network.
        """
        for action, prob in enumerate(action_probs):
            if prob > 0 and action not in self.children: self.children[action] = MCTSNode(parent=self, prior_p=prob)

    def update(self, leaf_value):
        """
        Updates the statistics of the node and its ancestors by backpropagating
        the value obtained from the leaf node of a simulation.
        """
        if self.parent: self.parent.update(-leaf_value)
        self.n_visits += 1; self.q_value += (leaf_value - self.q_value) / self.n_visits

    def is_leaf(self):
        """Checks if the node is a leaf node (i.e., has no children)."""
        return len(self.children) == 0

# ===================================================================
# HongikAIPlayer 클래스
# ===================================================================
class HongikAIPlayer:
    """
    The 'Supreme Commander' that makes the final decision. It uses the neural
    network's 'intuition' to guide the 'rational' search of the MCTS,
    ultimately selecting the best move.
    """
    def __init__(self, cnn_transformer_model, n_simulations=100):
        self.model, self.n_simulations, self.board_size = cnn_transformer_model, n_simulations, cnn_transformer_model.board_size
    
    def _action_to_loc(self, action, board):
        """Converts a policy network action index to a board location."""
        return board.loc(action % self.board_size, action // self.board_size) if action < self.board_size**2 else Board.PASS_LOC
    
    def get_best_move(self, board_state: Board, is_self_play=False):
        """
        Determines the best move for the current board state by running MCTS simulations.
        It integrates the neural network's policy and value predictions to guide the search.
        """
        features = board_state.get_features()
        policy_logits, value = self.model(np.expand_dims(features, 0), training=False)
        intuition_probs = tf.nn.softmax(policy_logits[0]).numpy()

        def is_filling_eye(loc, board):
            if board.board[loc] != Board.EMPTY: return False
            neighbor_colors = {board.board[loc + dloc] for dloc in board.adj if board.board[loc + dloc] != Board.WALL}
            return len(neighbor_colors) == 1 and board.pla in neighbor_colors

        for action, prob in enumerate(intuition_probs):
            if prob > 0.001:
                move_loc = self._action_to_loc(action, board_state)
                if move_loc != Board.PASS_LOC and is_filling_eye(move_loc, board_state): intuition_probs[action] = 0
        
        pass_action = self.board_size**2
        pass_prob = intuition_probs[pass_action]
        intuition_probs[pass_action] = 0
        
        if board_state.turns < 100: pass_prob = 0

        for action, prob in enumerate(intuition_probs):
            if prob > 0 and not board_state.would_be_legal(board_state.pla, self._action_to_loc(action, board_state)): intuition_probs[action] = 0
                
        total_prob = np.sum(intuition_probs)
        if total_prob <= 1e-6: return self._action_to_loc(pass_action, board_state), MCTSNode()
        intuition_probs /= total_prob
        
        root = MCTSNode(); root.expand(intuition_probs)
        for _ in range(self.n_simulations):
            node, search_board = root, board_state.copy()
            while not node.is_leaf():
                action, node = node.select(root.n_visits)
                move_loc = self._action_to_loc(action, search_board)
                if not search_board.would_be_legal(search_board.pla, move_loc):
                    node = None; break
                
                try:                    
                    search_board.play(search_board.pla, move_loc)
                except IllegalMoveError:
                    parent_node = node.parent
                    if parent_node and action in parent_node.children:
                        del parent_node.children[action]
                    
                    node = None
                    break             
            if node is not None:
                leaf_features = search_board.get_features()
                _, leaf_value_tensor = self.model(np.expand_dims(leaf_features, 0), training=False)
                leaf_value = leaf_value_tensor.numpy()[0][0]
                node.update(leaf_value)
            
        if not root.children: return self._action_to_loc(pass_action, board_state), root
        
        PASS_THRESHOLD = -0.99 
        best_action_node = max(root.children.values(), key=lambda n: n.n_visits)
        if best_action_node.q_value < PASS_THRESHOLD and pass_prob > 0:
            return self._action_to_loc(pass_action, board_state), root
        
        if board_state.turns < 30:
            if is_self_play:
                if not root.children:
                    return self._action_to_loc(pass_action, board_state), root
                
                child_actions = np.array(sorted(root.children.keys()))
                visit_counts = np.array([root.children[action].n_visits for action in child_actions], dtype=np.float32)

                temperature = 1.0 
                visit_counts_temp = visit_counts**(1/temperature)
                if np.sum(visit_counts_temp) == 0:
                    probs = np.ones(len(child_actions)) / len(child_actions)
                else:
                    probs = visit_counts_temp / np.sum(visit_counts_temp)
                
                action = np.random.choice(child_actions, p=probs)
                return self._action_to_loc(action, board_state), root
        
        if not root.children:
            return self._action_to_loc(pass_action, board_state), root

        visit_counts = np.zeros_like(intuition_probs)
        for action, node in root.children.items():
            visit_counts[action] = node.n_visits
        
        total_visits = np.sum(visit_counts)
        reason_probs = visit_counts / total_visits if total_visits > 0 else intuition_probs
        
        final_probs = (0.7 * intuition_probs) + (0.3 * reason_probs)
        
        final_probs[pass_action] = -1 
        
        sorted_actions = np.argsort(final_probs)[::-1]
        for action in sorted_actions:
            move_loc = self._action_to_loc(action, board_state)
            if board_state.would_be_legal(board_state.pla, move_loc):
                return move_loc, root
                    
        return self._action_to_loc(pass_action, board_state), root