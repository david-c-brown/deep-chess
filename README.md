# Chess AI using Monte Carlo Tree Search and Neural Networks

[Medium write up](https://dvdbrwn.medium.com/mastering-chess-with-deep-learning-a-monte-carlo-tree-search-and-ppo-loss-approach-24223ba7ab26)

This project aims to develop a sophisticated chess AI using a combination of Monte Carlo Tree Search (MCTS) and neural networks. The AI is designed to play as both white and black pieces, offering a challenging and engaging gameplay experience.

## Overview
The core components of this Chess AI are:

1. Neural networks for position evaluation and move selection
2. Monte Carlo Tree Search (MCTS) for game-tree exploration and move selection
3. Training process for the neural networks

## Neural Networks
The Chess AI leverages two separate neural networks, one for the white pieces and one for the black pieces. These networks are responsible for evaluating the current position of the board and predicting the best move. The architecture consists of multiple layers, including convolutional layers, fully connected layers, and a softmax output layer. 

## Monte Carlo Tree Search
MCTS is an algorithm for exploring game trees in a more efficient and intelligent manner compared to traditional brute-force methods. The algorithm relies on random sampling to estimate the probability of winning from a given position. It balances exploration (finding new moves) with exploitation (focusing on the most promising moves) using the Upper Confidence Bound applied to Trees (UCT) formula.

MCTS is particularly well-suited for chess for several reasons:
1. Complexity: Chess has a vast state-space and a high branching factor, making exhaustive search infeasible. MCTS provides a more efficient exploration strategy by focusing on promising moves.
2. Non-determinism: Although chess is a deterministic game, MCTS's random sampling simulates non-determinism, leading to a more robust AI that can handle uncertainty.
3. Adaptive: MCTS can adapt to changing game situations, as it doesn't rely on a fixed-depth search. This allows it to allocate more time to complex positions and less time to simple ones.

## Training Process
The neural networks are trained using a self-play method. The AI plays games against itself, and the resulting positions and moves are used to train the networks. This approach allows the AI to learn from its own mistakes and successes, improving its performance over time. The optimizer used is Adam, and the loss function is categorical cross-entropy. The models are compiled and trained using the Keras library with TensorFlow backend.

## Getting Started

To run the Chess AI, follow these steps:

    Install the required dependencies: pip install -r requirements.txt
    Run the main script: Open jupyter notebook, or google colab and hit alt-enter a bunch.

You can adjust the AI's strength by modifying the number of MCTS simulations or the neural network's architecture.
