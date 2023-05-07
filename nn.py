import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.losses import Huber
from functools import lru_cache

from mcts import *
from rewards import *


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Board and Input Conversion Functions
def board_to_input(board, history=1):
    # Map piece characters to integers for encoding
    pieces = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }
    # Initialize input tensor
    input_tensor = np.zeros((1, 8, 8, 12 * history), dtype=np.float32)

    # Create list of board states for input tensor
    board_states = [board]
    for _ in range(history - 1):
        if len(board.move_stack) > 0:
            move = board.pop()
            new_board = board.copy()
            board.push(move)
            board_states.insert(0, new_board)
        else:
            board_states.insert(0, None)

    # Encode pieces in input tensor
    for h, state in enumerate(board_states):
        if state is not None:
            for i in range(8):
                for j in range(8):
                    piece = state.piece_at(chess.square(i, j))
                    if piece:
                        input_tensor[0, i, j, pieces[str(piece)] + 12 * h] = 1

    return input_tensor

def moves_to_array(moves):
    # Create numpy array for legal moves
    move_array = np.zeros(4672)
    # encode moves in array
    for move in moves:
        move_idx = move.from_square * 73 + move.to_square
        move_array[move_idx] = 1
    return move_array

def array_to_move(board, move_array):
    # Get list of legal moves for board
    legal_moves = list(board.legal_moves)
    
    # Create numpy array for legal move probabilities
    legal_move_probs = np.zeros(len(legal_moves))

    # Map move array to legal move probabilities
    for i, move in enumerate(legal_moves):
        move_idx = move.from_square * 73 + move.to_square
        legal_move_probs[i] = move_array[move_idx]

    # Get legal move with highest probability
    best_move_idx = np.argmax(legal_move_probs)

    return legal_moves[best_move_idx] if legal_moves else None

# Evaluation and Backpropagation Functions
def backpropagate(node, value):
    # Update values of nodes in tree
    while node is not None:
        node.update(value)
        node = node.parent
        value = -value

# Memoize material balance function for performance
@lru_cache(maxsize=None)
def cached_evaluate_board(board_fen):
    # Create chess board from FEN string
    board = chess.Board(board_fen)
    return material_balance(board)

@tf.function(reduce_retracing=True)  
def model_predict(chess_model, board_input):
    # Convert input tensor to TensorFlow tensor
    board_input = tf.convert_to_tensor(board_input, dtype=tf.float32) 
    return chess_model(board_input) 

def evaluate(self, board):
    # Convert chess board to input tensor
    board_input = board_to_input(board).reshape(1, 8, 8, 12)
    move_probs, value_estimate = model_predict(self, board_input)
    return value_estimate.numpy().flatten()[0]

# Chess Model and Training Functions
def residual_block(inputs, num_filters, kernel_size=(3, 3)):
    # Defines a single residual block
    x = layers.Conv2D(num_filters, kernel_size, padding="same", activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_filters, kernel_size, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([inputs, x])
    x = layers.ReLU()(x)
    return x


def create_chess_model(num_res_blocks=19, num_filters=256):
    # Defines a deep residual network for playing chess
    input_shape = (8, 8, 12)

    # Input layer
    inputs = layers.Input(shape=input_shape, dtype=tf.float16)

    # Initial convolution layer
    x = layers.Conv2D(num_filters, kernel_size=(3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    # Residual blocks
    for _ in range(num_res_blocks):
        x = residual_block(x, num_filters)

    # Policy head
    policy = layers.Conv2D(2, kernel_size=(1, 1), padding="same", activation="relu")(x)
    policy = layers.BatchNormalization()(policy)
    policy = layers.Flatten()(policy)
    policy = layers.Dense(4672, activation="softmax")(policy)

    # Value head
    value = layers.Conv2D(1, kernel_size=(1, 1), padding="same", activation="relu")(x)
    value = layers.BatchNormalization()(value)
    value = layers.Flatten()(value)
    value = layers.Dense(1, activation="tanh")(value)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=[policy, value])

    def evaluate(self, board):
        # Evaluates a board state using the model
        board_input = board_to_input(board).reshape(1, 8, 8, 12)
        move_probs, value_estimate = model_predict(self, board_input)
        return value_estimate.numpy().flatten()[0]

    # Add the evaluate method to the model object
    model.evaluate = evaluate.__get__(model)
    return model

def compute_advantages(rewards, values, gamma=0.99, lambda_=0.95):
    # Compute the generalized advantage estimate (GAE) from the rewards and values
    advantages = np.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages[t] = gae
    return advantages

def ppo_loss_fn(advantages, old_probs, actions, logits, values, clip_epsilon=0.2, value_loss_coeff=0.5, entropy_coeff=0.01):
    # Calculate the PPO loss function
    prob_ratio = tf.exp(tf.nn.log_softmax(logits) - tf.stop_gradient(tf.nn.log_softmax(old_probs)))
    prob_ratio = tf.reduce_sum(prob_ratio * actions, axis=-1)
    clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    surrogate_loss = -tf.reduce_mean(tf.minimum(prob_ratio * advantages, clipped_prob_ratio * advantages))

    value_loss = Huber()(values, old_values)

    entropy_loss = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=-1))

    total_loss = surrogate_loss + value_loss_coeff * value_loss - entropy_coeff * entropy_loss
    return total_loss