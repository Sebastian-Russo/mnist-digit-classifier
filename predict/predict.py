import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import numpy as np
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
    transforms.Lambda(lambda x: TF.invert(x)),
    transforms.ToTensor(),  # Convert to digital format
    transforms.Normalize((0.1307,), (0.3081,))  # Adjust lighting to standard conditions
])


def preprocess_handwritten_digit(image_path):
    """
    ANALOGY: The case file preparation team cleans up messy photos
    to look like proper MNIST evidence photos
    """
    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')

    # Invert: black on white → white on black
    img = ImageOps.invert(img)

    # Convert to numpy array
    img_array = np.array(img)

    # Threshold to clean up (remove gray noise)
    threshold = 30
    img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)

    # Find bounding box of the digit (where the white pixels are)
    rows = np.any(img_array > 0, axis=1)
    cols = np.any(img_array > 0, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        img_array = img_array[rmin:rmax+1, cmin:cmax+1]

    # Convert back to PIL Image
    img = Image.fromarray(img_array)

    # Make lines thicker
    img = img.filter(ImageFilter.MaxFilter(3))

    # Resize to fit in 20x20 box
    width, height = img.size
    if width > height:
        new_width = 20
        new_height = max(int(height * 20 / width), 1)
    else:
        new_height = 20
        new_width = max(int(width * 20 / height), 1)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create 28x28 black canvas and paste centered
    canvas = Image.new('L', (28, 28), color=0)
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2
    canvas.paste(img, (paste_x, paste_y))

    return canvas


def predict_digit(image_path):
    """
    ANALOGY: Process a single case through the entire detective agency
    """
    # Use the improved preprocessing
    processed_img = preprocess_handwritten_digit(image_path)

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # "transform" Converts the image to a tensor and normalizes it.
    # "unsqueeze(0)" adds a "batch dimension" at position 0. The model expects batches of images, not single images!
    img_tensor = transform(processed_img).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()

    print(f"📁 Case File: {image_path}")
    print(f"🔎 Agency Verdict: This is digit {prediction}")
    print(f"💯 Confidence Level: {confidence*100:.2f}%")
    print(f"📊 All Verdict Specialists' Confidence:")
    for digit in range(10):
        print(f"   Digit {digit}: {probabilities[0][digit].item()*100:.2f}%")
    print("-" * 50)

    return prediction, confidence, processed_img


def predict_and_visualize(image_path):
    """
    ANALOGY: Present the case photo and verdict together
    Shows BOTH original and processed images!
    Like showing "Here's the evidence, here's what our agency concluded"

    """
    # Load original image
    original_img = Image.open(image_path).convert('L')

    # Get verdict AND processed image
    prediction, confidence, processed_img = predict_digit(image_path)

    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Your Original Drawing', fontsize=12)
    axes[0].axis('off')

    # Processed image (what model sees)
    axes[1].imshow(processed_img, cmap='gray')
    axes[1].set_title(f'What Model Sees\nVerdict: {prediction} ({confidence*100:.1f}%)',
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle('MNIST Detective Agency - Case Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = f'prediction_{os.path.basename(image_path)}'
    plt.savefig(output_file)
    plt.close()
    print(f"📌 Posted on bulletin board: {output_file}\n")


def predict_batch(image_folder):
    """
    ANALOGY: The morning mail truck arrives with a stack of cases!
    Instead of processing one case at a time at the front desk,
    the agency tackles the entire stack and posts all results on
    a big summary board at the end of the day.
    """
    # ANALOGY: Sort through the mail - only keep valid case files (images)
    image_files = [f for f in os.listdir(image_folder)
                   if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"📭 No cases found in {image_folder}")
        return

    print(f"📬 Received {len(image_files)} cases. Agency processing...\n")
    print("=" * 60)

    # ANALOGY: The results clipboard - track each case's outcome
    results = []

    # ANALOGY: Work through the stack of cases one by one
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)

        # ANALOGY: Send each case through the full detective agency pipeline
        prediction, confidence, processed_img = predict_digit(img_path)

        # ANALOGY: Try to extract the true answer from filename (e.g., "digit_7.png" → 7)
        true_digit = None
        for char in img_file:
            if char.isdigit():
                true_digit = int(char)
                break

        # ANALOGY: Write the verdict on our clipboard
        results.append((img_file, prediction, confidence, true_digit))

    # ANALOGY: End of the day - post the summary board in the lobby
    print("\n" + "=" * 60)
    print("📊 DAILY CASE SUMMARY BOARD")
    print("=" * 60)

    correct_count = 0
    high_confidence_count = 0

    for img_file, pred, conf, true_digit in results:
        # ANALOGY: Check if we got it right (if we know the answer)
        if true_digit is not None:
            is_correct = pred == true_digit
            if is_correct:
                correct_count += 1
            status = "✅" if is_correct else "❌"
        else:
            status = "❓"  # Unknown - can't verify

        if conf > 0.8:
            high_confidence_count += 1

        # ANALOGY: Each row on the board shows one solved case
        confidence_bar = "█" * int(conf * 10)
        print(f"   {status} {img_file:<20} Verdict: {pred}  ({conf*100:>5.1f}%) {confidence_bar}")

    print("-" * 60)
    print(f"📈 Total cases processed: {len(results)}")
    print(f"💪 High confidence (>80%): {high_confidence_count}/{len(results)}")
    print(f"🎯 Correct predictions: {correct_count}/{len(results)}")
    print("=" * 60)

    return results


# Main execution
if __name__ == '__main__':
    """
    ANALOGY: The agency's front desk - where clients bring their cases
    You can bring one case, multiple cases, or keep bringing cases interactively!
    """
    import sys

    print("=" * 60)
    print("🏢 MNIST DETECTIVE AGENCY - Case Processing Center")
    print("=" * 60)

    # Check if user provided arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == '--batch' or arg == '-b':
            # Process all images in test_images/
            print("📋 MODE: Batch Processing")
            predict_batch('test_images/')

        elif arg == '--help' or arg == '-h':
            print("Usage:")
            print("  python3 predict/predict.py              Interactive mode")
            print("  python3 predict/predict.py --batch      Process all in test_images/")
            print("  python3 predict/predict.py image.png    Process single image")

        elif os.path.exists(arg):
            # Process single image provided as argument
            print(f"📋 MODE: Single Image")
            predict_and_visualize(arg)

        else:
            print(f"❌ File not found: {arg}")

    else:
        # Default: Interactive mode
        print("📋 MODE: Interactive Front Desk")
        print("Bring cases one at a time, or type 'quit' when done.\n")

        image_path = input("🔍 Enter path to case photo: ")

        while image_path.lower() != 'quit':
            if os.path.exists(image_path):
                print("\n📨 Case received! Sending to detective agency...\n")
                predict_and_visualize(image_path)
            else:
                print(f"❌ Case file not found: {image_path}")

            image_path = input("🔍 Enter path to next case (or 'quit'): ")

        print("\n👋 Thank you for visiting!")