# Neural Name Forge

A neural network playground for understanding character-level language models through name generation.

## About the Project

NeuralNameForge is an educational project that implements various neural network architectures to generate names. It serves as a practical introduction to different types of neural networks, from simple bigram models to complex transformers. By focusing on the specific task of name generation, it provides a concrete way to understand how different neural architectures process and generate sequential data.

## Components

The project implements several neural network architectures, each representing a different approach to sequence modeling:

1. **Bigram Model**
   - Simplest form of statistical language model
   - Based on character pair probabilities

2. **Multi-Layer Perceptron (MLP)**
   - Implementation based on Bengio et al. 2003
   - Demonstrates basic neural network concepts

3. **Recurrent Neural Networks (RNN)**
   - Vanilla RNN
   - GRU (Gated Recurrent Unit)
   - LSTM (Long Short-Term Memory)

4. **Transformer**
   - Based on the architecture used in GPT
   - Implements self-attention mechanism

5. **Bag of Words (BoW)**
   - Alternative approach to sequence modeling
   - Demonstrates context aggregation

## How to Use

1. **Setup**
   ```bash
   git clone https://github.com/yourusername/NeuralNameForge
   cd NeuralNameForge
   pip install torch tensorboard
   ```

2. **Basic Usage**
   ```bash
   # Train a transformer model (default)
   python makemore.py -i names.txt -o out

   # Try different architectures
   python makemore.py -i names.txt -o out --type lstm
   python makemore.py -i names.txt -o out --type gru
   python makemore.py -i names.txt -o out --type bigram

   # Generate names without training
   python makemore.py -i names.txt -o out --sample-only
   ```

3. **Monitor Training**
   ```bash
   tensorboard --logdir out
   ```

## Note to Myself: Neural Network Concepts

### Key Concepts Used in This Project:

1. **Embeddings**
   - Convert discrete characters into continuous vectors
   - Learned representation of characters

2. **Sequential Processing**
   - RNN: Processes data one step at a time, maintaining hidden state
   - LSTM/GRU: Advanced RNNs with gating mechanisms
   - Transformer: Processes entire sequence using attention

3. **Attention Mechanism**
   - Self-attention in transformer
   - Allows model to focus on relevant parts of input
   - Parallel processing advantage over RNNs

4. **Loss Functions**
   - Cross-entropy loss for character prediction
   - Measures prediction accuracy

5. **Model Architecture Components**
   - Linear layers
   - Activation functions (Tanh, ReLU, Sigmoid)
   - Layer normalization
   - Residual connections

6. **Training Dynamics**
   - Gradient descent optimization
   - Learning rate importance
   - Batch processing
   - Model evaluation and validation

### Architecture-Specific Notes:

- **Bigram**: Simplest model, uses counting statistics
- **MLP**: Feedforward network, fixed context window
- **RNN**: Processes sequences recursively
- **LSTM/GRU**: Solves vanishing gradient problem
- **Transformer**: Uses attention for global context

## License

MIT License - See LICENSE file for details.

## Acknowledgements

This project is a learning-focused reimagining of Andrej Karpathy's makemore project. Key papers that influenced this implementation:

- Bengio et al. 2003 (Neural Probabilistic Language Models)
- Graves et al. 2014 (LSTM)
- Cho et al. 2014 (GRU)
- Vaswani et al. 2017 (Transformer)

Special thanks to the PyTorch team for their excellent deep learning framework.

---
*Note: This project is designed for learning purposes. While it can generate names, its primary value lies in understanding neural network architectures and their implementations.*
