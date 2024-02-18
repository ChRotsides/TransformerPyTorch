# Transformer Model for Machine Translation

This project implements a Transformer model for machine translation, focusing on translating from English to Italian. The model architecture is based on the seminal paper "Attention is All You Need" by Vaswani et al., utilizing self-attention mechanisms for efficient translation.

## Features

- Implementation of the Transformer model in PyTorch.
- Training on the OPUS Books dataset for English to Italian translation.
- Custom tokenizer handling and dataset preparation for efficient training.
- Integration with TensorBoard for training visualization.

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://your-repository-url.git
   cd your-repository-directory
   ```

2. Install the required packages:
   ```bash
   pip install torch torchvision torchaudio
   pip install datasets tokenizers tqdm
   ```

## Usage

To train the model, run the `train.py` script:

```bash
python train.py
```

This will start the training process and save model checkpoints to the `weights` directory. You can monitor the training progress using TensorBoard:

```bash
tensorboard --logdir=runs
```

## Configuration

The model and training configurations can be adjusted in the `config.py` file. This includes settings such as batch size, learning rate, and the number of epochs.

## Acknowledgments

- The Transformer model architecture is based on the paper "Attention is All You Need" by Vaswani et al.
- The dataset used for training is sourced from the OPUS collection of translated texts.

## License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for more details.
