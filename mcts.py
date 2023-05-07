
import math
import concurrent.futures
import numpy as np

from rewards import *
from nn import *

# MCTS Functions and Classes
class MCTSNode:
    # Initialize MCTSNode object
    def __init__(self, board, parent=None, move=None, prior=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0
        self.prior = prior
        self.additional_rewards = 0  
        self.eval_value = self.evaluate_board()

    def add_child(self, child_node):
        # Add child node to MCTSNode
        self.children.append(child_node)

    def update(self, value):
        self.visits += 1
        self.value += value

    def expand_children_parallel(self, chess_model, num_threads=4):
        # Expand children of MCTSNode in parallel using threads
        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 0:
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self._expand_child, move, chess_model) for move in legal_moves]

            for future in concurrent.futures.as_completed(futures):
                child = future.result()
                if child is not None:
                    self.children.append(child)

    def _expand_child(self, move, chess_model):
    # Create new child node with specified move and evaluate it
      child_board = self.board.copy()
      child_board.push(move)
      child_node = MCTSNode(child_board, self, move)
      child_node.evaluate(chess_model)

      # Reward implementation
      rewards = 0
      rewards += material_balance(child_board)
      rewards += piece_mobility(child_board)
      rewards += king_safety(child_board)
      rewards += center_control(child_board)
      rewards += pawn_structure(child_board)

      if child_node.board.is_checkmate():
          rewards += 1000

      if child_node.board.is_insufficient_material() or child_node.board.can_claim_draw() or child_node.board.is_stalemate():
                        rewards -= 500

      child_node.additional_rewards = rewards
      
      return child_node


    def fully_expanded(self):
        # Check if MCTSNode is fully expanded
        return len(self.children) == len(list(self.board.legal_moves))

    def evaluate_board(self):
        return material_balance(self.board)

    def select_child(self, temperature):
        ucb_scores = [
            (child.value / (child.visits + 1e-10) +
             math.sqrt(2 * math.log(self.visits + 1e-10) / (child.visits + 1e-10)))
            for child in self.children
        ]

        if temperature == 0:
            best_child_index = ucb_scores.index(max(ucb_scores))
        else:
            ucb_scores = [x**(1/temperature) for x in ucb_scores]
            total_score = sum(ucb_scores)
            probabilities = [x / total_score for x in ucb_scores]
            best_child_index = np.random.choice(len(self.children), p=probabilities)

        return self.children[best_child_index]

    def is_leaf(self):
        return len(self.children) == 0

    def evaluate(self, chess_model):
        # Evaluate the MCTSNode using the specified chess model
        self.visits += 1
        self.value = chess_model.evaluate(self.board)

    def best_child(self, c_param=1, eval_weight=0, temperature=1, noise=0):
        # Choose the best child of the MCTSNode
        if self.children == []:
            return None

        # Calculate the weights of all possible choices
        choices_weights = [
            ((c.value / (c.visits + 1e-8) + c_param * (c.prior + np.random.randn() * noise) * np.sqrt(self.visits) / (1 + c.visits) + eval_weight * c.eval_value) / temperature)
            for c in self.children
        ]

        return self.children[np.argmax(choices_weights)]

def expand(node, chess_model):
    # Expand the given node by creating child nodes for each legal move
    board_input = board_to_input(node.board).reshape(1, 8, 8, 12)
    move_probs, value_estimate = model_predict(chess_model, board_input)
    move_probs = move_probs.numpy().flatten()

    legal_moves = list(node.board.legal_moves)
    for move in legal_moves:
        # Create new child node for each legal move
        new_board = node.board.copy()
        new_board.push(move)
        move_idx = move.from_square * 73 + move.to_square
        prior = move_probs[move_idx]
        child_node = MCTSNode(new_board, parent=node, move=move, prior=prior)
        node.add_child(child_node)

    # Choose random child node to return
    return node.children[np.random.choice(len(node.children))]

def mcts(board, chess_model, num_simulations, temperature=1.0, noise=0.0, parallel_sims=True):
    # Run Monte Carlo Tree Search on the specified chess board and model
    root = MCTSNode(board, chess_model)

    # Run simulations sequentially or in parallel
    if not parallel_sims:
        for _ in range(num_simulations):
            current_board = board.copy()
            node = root
            while not node.is_leaf():
                node = node.select_child(temperature)
                current_board.push(node.move)
            if not current_board.is_game_over():
                node.expand_children(chess_model)
    else:
        def simulate(root):
            current_board = board.copy()
            node = root
            while not node.is_leaf():
                node = node.select_child(temperature)
                current_board.push(node.move)
            if not current_board.is_game_over():
                node.expand_children_parallel(chess_model)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_simulations) as executor:
            executor.map(simulate, [root]*num_simulations)
    
    best_child = root.best_child()
    # Return the move with the highest weight
    return best_child.move, best_child.additional_rewards
     