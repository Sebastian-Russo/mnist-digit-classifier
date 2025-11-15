import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the CNN architecture
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional layer: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers # Analogy: "Spread all 64 expert reports on one giant conference table in a single line."
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 7x7 because of two pooling layers on 28x28 image
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5) # Analogy: During training, randomly send half home, so detectives (neuorons) don't depend on each other, become self reliant
        # Activation function
        self.relu = nn.ReLU() # Analogy: Quality control approves confident conclusions (only positive scores, remove 0 and negative)

    def forward(self, x):
        # Conv layer 1 + activation + pooling
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        # Conv layer 2 + activation + pooling
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        # Fully connected layer 1 + activation + dropout
        x = self.dropout(self.relu(self.fc1(x)))
        # Output layer
        x = self.fc2(x)
        return x

# Data preprocessing and loading
def load_data():
    # Transform: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(), # Conver data to same format (ToTensor)
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Download and load training data
    # # Analogy: "Prepare the case files consistently:"
        # Convert all photos to the same format (ToTensor)
        # Adjust lighting to standard conditions (Normalize) so detectives can compare cases fairly
    train_dataset = datasets.MNIST(root='test_images/data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Download and load test data
    test_dataset = datasets.MNIST(root='test_images/data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader

# Training function
# Analogy Training Academy
    # Analogy: "It's training day! Everyone report to the academy. Dropout is active - random people will be sent home during exercises."
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Analogy: "Here comes a batch of 64 practice cases with answer sheets."
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # device => CPU or GPU

        # Zero the gradients
        optimizer.zero_grad() # Analogy: Clear the whiteboard from last training session. Fresh start!

        # Forward pass
        output = model(data) # Analogy: Run the cases through the entire agency pipeline and get 64 verdicts.
        loss = criterion(output, target) # Analogy: "Compare your verdicts to the answer sheet. How wrong were you? That's your 'loss score'."

        # Backward pass and optimize
        # Analogy: "INVESTIGATION TIME! The chief traces back through the entire agency:"
            # "Verdict specialists, your weights were off by this much..."
            # "Analysts, you need to adjust these connections..."
            # "Senior detectives, pay more attention to this feature..."
            # "Junior detectives, your filters need tweaking here..."
            # This is backpropagation - figuring out exactly who needs to improve and by how much.
        loss.backward()
        optimizer.step() # Analogy: "Everyone make those adjustments NOW!"

        # Track statistics
        # Analogy: "Count how many cases we got right in this batch."
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# Testing function
# Analogy: Final Exam (test function)
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    # Analogy: "This is just an exam - don't learn from it! No adjustments, no backpropagation, just see how you perform."
    with torch.no_grad():
        for data, target in test_loader:
            # Analogy: "Process all 10,000 secret cases and compare to answers. What's the final score?"
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Analogy: "Results are in! You got x out of 10,000 correct. That's xx.xx% accuracy!"
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return test_loss, accuracy

# Visualize some predictions
# Analogy: "Let's see some example cases the agency solved."
def visualize_predictions(model, device, test_loader, num_images=6):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_images].to(device), labels[:num_images]

    # Analogy: "Run 6 cases through without learning, just to see the verdicts."
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    #Analogy: "Create a bulletin board showing:"
        # The photo evidence
        # The correct answer
        # What your agency guessed
    for idx in range(num_images):
        img = images[idx].cpu().squeeze()
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f'True: {labels[idx].item()}, Predicted: {predictions[idx].item()}')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png')
    print("Saved predictions visualization to predictions.png")

# Main training loop
# Analogy: The Grand Opening
def main():
    # Load data
    # Analogy: "Gather all the case files from the archives."
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_data()

    # Initialize model
    # Analogy: "Build the entire detective agency according to our blueprint and move everyone into the crime lab (GPU/CPU)."
    model = DigitClassifier().to(device)
    print(f"\nModel architecture:\n{model}")

    # Loss function and optimizer
    # Analogy: "Define how we measure mistakes. CrossEntropyLoss is like a strict grading rubric that penalizes confident wrong answers heavily."
    criterion = nn.CrossEntropyLoss()
    # Analogy: "Hire Adam, the training coordinator, who decides how much each person adjusts after mistakes."
        # lr=0.001 (learning rate) = "Make small, careful adjustments. Don't overcorrect!"
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    # Analogy: "We'll run 5 complete training sessions. Each session, everyone sees all 60,000 training cases."
    num_epochs = 5

    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # Analogy: "Each day follows the same pattern:"
        # Day 1:
            # Morning: Train on 60,000 cases (with mistakes and corrections)
            # Afternoon: Take final exam on 10,000 secret cases
            # Record: Training accuracy 94%, Test accuracy 92%
        # Day 2, repeat, Day 3-5, Keep improving!
    print("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    # Save the model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved as mnist_model.pth")

    # Visualize some predictions
    visualize_predictions(model, device, test_loader)

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Analogy: "Create performance charts:"
        # Graph 1: Loss over time (Should go DOWN - fewer mistakes each day)
        # Graph 2: Accuracy over time (Should go UP - more correct answers each day)

    ax1.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    ax1.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()

    ax2.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    ax2.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Saved training history to training_history.png")

    # Analogy: "GRAND OPENING ANNOUNCEMENT: Our agency achieves 98.5% accuracy! Open for business!"
    print(f"\nFinal Test Accuracy: {test_accuracies[-1]:.2f}%")

if __name__ == '__main__':
    main()


# The Complete Story Arc:

# Planning phase: Design the agency structure
# Hiring phase: Set up all departments and specialists
# Preparation phase: Gather and organize 70,000 case files
# Training phase: 5 days of intensive practice with corrections
# Testing phase: Regular exams to measure real performance
# Graduation: Save the trained agency
# Showcase: Display examples and performance graphs
# Open for business: Ready to classify real handwritten digits!