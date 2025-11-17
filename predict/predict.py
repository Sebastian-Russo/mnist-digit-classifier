import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

"""
DETECTIVE AGENCY ANALOGY:
This script is like bringing NEW cases to your trained detective agency.
Your agency has already graduated from training - now it's time to solve real cases!
"""

# Import your model architecture (copy from train_mnist.py)
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        # ANALOGY: Setting up the detective agency departments
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 32 junior detectives
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 senior detectives
        self.pool = nn.MaxPool2d(2, 2)  # Assistant who summarizes reports
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 128 analysts in the war room
        self.fc2 = nn.Linear(128, 10)  # 10 verdict specialists (one per digit)
        self.dropout = nn.Dropout(0.5)  # Training policy (randomly send people home)
        self.relu = nn.ReLU()  # Quality control manager (only confident findings!)

    def forward(self, x):
        # ANALOGY: The case processing workflow
        x = self.pool(self.relu(self.conv1(x)))  # Juniors investigate → QC → Summarize
        x = self.pool(self.relu(self.conv2(x)))  # Seniors review → QC → Summarize
        x = x.view(-1, 64 * 7 * 7)  # Spread all reports on conference table
        x = self.dropout(self.relu(self.fc1(x)))  # Analysts combine evidence
        x = self.fc2(x)  # Verdict specialists make final call
        return x

# ANALOGY: Reopening the detective agency with all their training/experience intact
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DigitClassifier().to(device)
model.load_state_dict(torch.load('mnist_model.pth'))  # Load their learned expertise!
model.eval()  # Set to "real cases" mode (everyone shows up, no training exercises)
print("Model loaded successfully!")
print("🔍 Detective agency is open for business!\n")

# ANALOGY: Standard case file preparation procedures
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert photo to grayscale (agency standard)
    transforms.Resize((28, 28)),  # Resize to standard evidence photo size
    transforms.ToTensor(),  # Convert to digital format
    transforms.Normalize((0.1307,), (0.3081,))  # Adjust lighting to standard conditions
])

def predict_digit(image_path):
    """
    ANALOGY: Process a single case through the entire detective agency

    The case (image) goes through:
    1. Case file preparation (preprocessing)
    2. Investigation by all departments (forward pass)
    3. Final verdict with confidence level
    """
    # ANALOGY: Receive the case photo and prepare it in standard format
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)  # Put in case folder, send to crime lab

    # ANALOGY: Run the case through the entire agency
    with torch.no_grad():  # This is a real case, not a training exercise!
        output = model(img_tensor)  # Agency processes the case
        probabilities = torch.softmax(output, dim=1)  # Convert to confidence percentages
        prediction = output.argmax(dim=1).item()  # Which verdict specialist was most confident?
        confidence = probabilities[0][prediction].item()  # How confident are they?

    print(f"📁 Case File: {image_path}")
    print(f"🔎 Agency Verdict: This is digit {prediction}")
    print(f"💯 Confidence Level: {confidence*100:.2f}%")
    print(f"📊 All Verdict Specialists' Confidence:")
    for digit in range(10):
        print(f"   Digit {digit}: {probabilities[0][digit].item()*100:.2f}%")
    print("-" * 50)

    return prediction, confidence

def predict_and_visualize(image_path):
    """
    ANALOGY: Present the case photo and verdict together on a bulletin board
    Like showing "Here's the evidence, here's what our agency concluded"
    """
    # Load original image for display
    img = Image.open(image_path).convert('L')

    # Get agency's verdict
    prediction, confidence = predict_digit(image_path)

    # ANALOGY: Post on the bulletin board for everyone to see
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f'🎯 Agency Verdict: Digit {prediction}\n'
              f'Confidence: {confidence*100:.1f}%',
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    output_file = f'prediction_{os.path.basename(image_path)}'
    plt.savefig(output_file)
    plt.show()
    print(f"📌 Posted on bulletin board: {output_file}\n")

def predict_batch(image_folder):
    """
    ANALOGY: Process a stack of cases that came in today
    The agency tackles multiple cases and presents all results on one big board
    """
    image_files = [f for f in os.listdir(image_folder)
                   if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"📭 No cases found in {image_folder}")
        return

    print(f"📬 Received {len(image_files)} cases. Agency processing...\n")
    print("=" * 60)

    # ANALOGY: Create a case board showing up to 6 cases
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, img_file in enumerate(image_files[:6]):
        img_path = os.path.join(image_folder, img_file)

        # ANALOGY: Each case goes through the full agency process
        img = Image.open(img_path).convert('L')
        prediction, confidence = predict_digit(img_path)

        # ANALOGY: Pin this case result to the board
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f'📁 {img_file}\n🔎 Verdict: {prediction} ({confidence*100:.1f}% confident)',
                           fontsize=10)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(len(image_files[:6]), 6):
        axes[idx].axis('off')

    plt.suptitle('🏢 Detective Agency: Today\'s Case Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('batch_predictions.png', dpi=150)
    plt.show()
    print("\n" + "=" * 60)
    print("📌 All cases posted on board: batch_predictions.png")

# Main execution
if __name__ == '__main__':
    """
    ANALOGY: The agency's front desk - where clients bring their cases
    You can bring one case, multiple cases, or keep bringing cases interactively!
    """

    print("=" * 60)
    print("🏢 MNIST DETECTIVE AGENCY - Case Processing Center")
    print("=" * 60)
    print("Our trained detectives are ready to identify handwritten digits!")
    print("Bring us your mystery digit photos and we'll solve them!\n")

    # Uncomment to use different modes:

    # MODE 1: Process a single case
    # print("📋 MODE: Single Case Processing")
    # predict_and_visualize('test_images/digit_7.png')

    # MODE 2: Process all cases in a folder
    # print("📋 MODE: Batch Case Processing")
    # predict_batch('test_images/')

    # MODE 3: Interactive front desk (ask for cases one by one)
    print("📋 MODE: Interactive Front Desk")
    print("Bring cases one at a time, or type 'quit' when done.\n")

    image_path = input("🔍 Enter path to case photo: ")

    while image_path.lower() != 'quit':
        if os.path.exists(image_path):
            print("\n📨 Case received! Sending to detective agency...\n")
            predict_and_visualize(image_path)
        else:
            print(f"❌ Case file not found: {image_path}")
            print("Please check the path and try again.\n")

        image_path = input("🔍 Enter path to next case (or 'quit'): ")

    print("\n👋 Thank you for visiting the MNIST Detective Agency!")
    print("All case files have been processed. Have a great day!")
