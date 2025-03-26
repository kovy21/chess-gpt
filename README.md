# chess-gpt

### Chess Move Prediction with Deep Learning

This project explores the use of deep learning to predict chess moves from board positions. It utilizes historical chess game data in PGN format, encodes the board state, and trains a neural network model to suggest the most likely next move.

## Overview

The notebook uses a pipeline that:
- Parses PGN files to extract game positions and moves
- Encodes board states into a 3D tensor format
- Encodes legal moves as one-hot vectors
- Trains a Keras neural network model
- Evaluates the model's ability to predict next moves

## Files

- `chess.ipynb`: The main notebook containing the full pipeline.

## Requirements

The project uses the following Python libraries:
- NumPy, python-chess

Install dependencies with:

```bash
pip install chess tensorflow numpy tqdm
```

## Model Architecture

The notebook defines a simple deep learning model using Keras:
- Input: Encoded board state (8x8x12)
- Hidden layers: Dense layers
- Output: Softmax activation over move classes

## Functionality

Key functionalities in the notebook:
- Move prediction, PGN file parsing

## Limitations

- Only supports prediction of a single move
- Requires sufficient training data to generalize
- Evaluation metrics are minimal

## Future Improvements

- Enhance evaluation (e.g. top-k accuracy, Elo-based)
- Expand data preprocessing
- Use advanced architectures (e.g. CNNs or Transformers)
- Deploy as a playable move suggestion engine

## License

MIT License

------

## Key Figures

<img width="732" alt="Screenshot 2025-03-26 at 09 12 51" src="https://github.com/user-attachments/assets/e0bf7be0-2125-4935-86e6-be874302e0fd" />

Figure 1: Demonstrating the increasing accuracy of the transformer model with a growing sample size

<img width="833" alt="Screenshot 2025-03-26 at 09 13 06" src="https://github.com/user-attachments/assets/b4ac83d2-fee3-4df0-9998-05f422e6adf4" />

Figure 2: Normalised confusion matrix for most frequent moves
