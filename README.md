# MNIST Digit Classifier

Your first neural network! This project trains a Convolutional Neural Network (CNN) to recognize handwritten digits.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install torch torchvision matplotlib numpy pillow
```

## Training the Model

Run the training script:
```bash
python train_mnist.py
```

This will:
- Download the MNIST dataset (60,000 training images, 10,000 test images)
- Train a CNN for 5 epochs (~5-10 minutes on CPU)
- Save the trained model as `mnist_model.pth`
- Generate two visualizations:
  - `predictions.png` - Sample predictions
  - `training_history.png` - Loss and accuracy over time

Expected accuracy: ~98-99%

## Testing the Model

Test on your own image:
```bash
python test_model.py path/to/your/digit.png
```

Run Prediction on your own image:
```bash
python predict.py
# Then enter: test_images/my_digit_7.png
```
Then run Visualizations:
```bash
python visualize.py
```


The image should:
- Be a handwritten digit (0-9)
- Have dark digit on light background (or it will invert automatically)
- Be any size (will be resized to 28x28)

## Understanding the Code

### Model Architecture (train_mnist.py)
- **Conv Layer 1**: Extracts basic features (edges, curves)
- **Conv Layer 2**: Extracts complex features (digit shapes)
- **Pooling Layers**: Reduces image size, focuses on important features
- **Fully Connected Layers**: Makes the final classification decision

### Key Concepts
- **Epochs**: Number of times the model sees the entire dataset
- **Batch Size**: Number of images processed at once (64)
- **Learning Rate**: How much the model adjusts weights (0.001)
- **Dropout**: Randomly turns off neurons to prevent overfitting

### What's Happening During Training
1. **Forward Pass**: Image → Model → Prediction
2. **Loss Calculation**: How wrong was the prediction?
3. **Backward Pass**: Calculate how to adjust weights
4. **Optimization**: Update weights to improve

## Next Steps

Once you have a trained model (mnist_model.pth), you're ready for:

**Phase 2: Build a Django/Flask API** to serve predictions
**Phase 3: Build a React frontend** where users can draw digits

## Troubleshooting

**Training is slow**: That's normal on CPU! Should take 5-10 minutes for 5 epochs.

**ImportError**: Make sure you activated the virtual environment and installed all dependencies.

**Model not found**: Make sure `mnist_model.pth` exists in the same directory as `test_model.py`.

## Files Generated
- `mnist_model.pth` - Your trained model (save this!)
- `predictions.png` - Sample predictions visualization
- `training_history.png` - Training metrics
- `data/` - MNIST dataset (downloaded automatically)