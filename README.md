# chess-gpt

### Chess Move Prediction with Deep Learning

This project explores the use of deep learning to predict chess moves from board positions. It utilizes historical chess game data in PGN format, encodes the board state, and trains a neural network model to suggest the most likely next move.

## 🧠 Overview

The notebook uses a pipeline that:
- Parses PGN files to extract game positions and moves
- Encodes board states into a 3D tensor format
- Encodes legal moves as one-hot vectors
- Trains a Keras neural network model
- Evaluates the model's ability to predict next moves

## 📁 Files

- `chess.ipynb`: The main notebook containing the full pipeline.

## 📦 Requirements

The project uses the following Python libraries:
- NumPy, python-chess

Install dependencies with:

```bash
pip install chess tensorflow numpy tqdm
```

## 🏗 Model Architecture

The notebook defines a simple deep learning model using Keras:
- Input: Encoded board state (8x8x12)
- Hidden layers: Dense layers
- Output: Softmax activation over move classes

## 📊 Functionality

Key functionalities in the notebook:
- Move prediction, PGN file parsing

## 🚧 Limitations

- Only supports prediction of a single move
- Requires sufficient training data to generalize
- Evaluation metrics are minimal

## 📌 Future Improvements

- Enhance evaluation (e.g. top-k accuracy, Elo-based)
- Expand data preprocessing
- Use advanced architectures (e.g. CNNs or Transformers)
- Deploy as a playable move suggestion engine

## 📜 License

MIT License
