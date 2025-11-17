import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

"""
DETECTIVE AGENCY ANALOGY:
This script lets you peek INSIDE the detective agency to see:
1. What skills each detective learned (filter visualization)
2. How they actually process cases step-by-step (feature maps)

It's like getting a behind-the-scenes tour of the agency!
"""

# Import your model architecture
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ANALOGY: Open the agency and examine their trained expertise
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DigitClassifier().to(device)
model.load_state_dict(torch.load('../train/mnist_model.pth'))
model.eval()
print("🏢 Detective agency loaded - ready for behind-the-scenes tour!\n")

def visualize_conv1_filters():
    """
    ANALOGY: Look at each junior detective's magnifying glass!

    After training, each of the 32 junior detectives has specialized their
    3×3 magnifying glass to look for specific patterns:
    - Some look for vertical edges
    - Some look for horizontal edges
    - Some look for curves
    - Some look for corners

    This shows you WHAT they learned to detect!
    """
    print("🔍 TOUR STOP 1: Junior Detectives' Department")
    print("Let's see what each of the 32 junior detectives learned to look for...\n")

    # ANALOGY: Examine each detective's specialized magnifying glass (filter)
    filters = model.conv1.weight.data.cpu()

    print(f"📊 Each detective has a {filters.shape} magnifying glass")
    print(f"   - 32 detectives")
    print(f"   - Each looks at 1 input channel (grayscale)")
    print(f"   - Each uses a 3×3 viewing window\n")

    # ANALOGY: Display all 32 detectives' specializations on a wall chart
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))

    for i, ax in enumerate(axes.flat):
        if i < 32:
            # ANALOGY: This is what Detective #i's magnifying glass looks for
            filter_img = filters[i, 0].numpy()

            # Normalize for better visibility
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())

            ax.imshow(filter_img, cmap='gray')
            ax.set_title(f'Detective {i+1}\nSpecialty', fontsize=8)
            ax.axis('off')

            # ANALOGY: Describe what this detective might be looking for
            if i == 0:
                ax.text(0.5, -0.3, '(e.g., edges, curves, corners)',
                       transform=ax.transAxes, ha='center', fontsize=6, style='italic')
        else:
            ax.axis('off')

    plt.suptitle('👮 32 Junior Detectives: What Each One Learned to Detect',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/learned_filters_conv1.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved junior detectives' specializations: visualizations/learned_filters_conv1.png\n")

def visualize_conv2_filters():
    """
    ANALOGY: Look at the senior detectives' expertise!

    The 64 senior detectives combine what the juniors found.
    Each senior detective looks at ALL 32 junior reports and
    learns to recognize more complex patterns like:
    - "Top curve + vertical line" (could be a 5 or 2)
    - "Two circles stacked" (probably an 8)
    - "Diagonal with horizontal top" (likely a 7)
    """
    print("🔍 TOUR STOP 2: Senior Detectives' Department")
    print("Senior detectives combine junior findings to spot complex patterns...\n")

    # ANALOGY: Each senior detective reviews all 32 junior reports
    filters = model.conv2.weight.data.cpu()

    print(f"📊 Each senior detective reviews {filters.shape}")
    print(f"   - 64 senior detectives")
    print(f"   - Each reviews 32 junior reports")
    print(f"   - Each uses 3×3 analysis on each report\n")
    print("   (Showing a sample of their specializations...)\n")

    # ANALOGY: Show how 24 senior detectives analyze the FIRST junior's report
    fig, axes = plt.subplots(4, 6, figsize=(12, 8))

    for i, ax in enumerate(axes.flat):
        if i < 24:
            # ANALOGY: How Senior Detective #i analyzes Junior Detective #0's report
            filter_img = filters[i, 0].numpy()  # First input channel only
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())

            ax.imshow(filter_img, cmap='gray')
            ax.set_title(f'Senior {i+1}\nAnalysis', fontsize=8)
            ax.axis('off')

    plt.suptitle('👔 Senior Detectives: How They Analyze Junior Reports\n'
                 '(Showing how each senior processes one junior\'s findings)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/learned_filters_conv2.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved senior detectives' analysis patterns: visualizations/learned_filters_conv2.png\n")

def visualize_feature_maps(image_idx=0):
    """
    ANALOGY: Watch a case flow through the entire agency in real-time!

    Like a security camera showing:
    1. Original photo arrives
    2. Junior detectives mark what THEY see (32 different perspectives)
    3. Senior detectives mark what THEY see (64 different perspectives)

    Each detective highlights different aspects of the same digit!
    """
    print("🔍 TOUR STOP 3: Live Case Processing")
    print("Let's watch a real case flow through the agency step-by-step...\n")

    # ANALOGY: Grab a case from the archives
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root='../.data', train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # ANALOGY: Pick a random case to observe
    for i, (image, label) in enumerate(test_loader):
        if i == image_idx:
            break

    image = image.to(device)

    print(f"📁 Selected Case: Digit {label.item()}")
    print(f"🎬 Processing begins...\n")

    # ANALOGY: Set up security cameras at each department
    activations = {}

    def get_activation(name):
        """Security camera that records what each department outputs"""
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # ANALOGY: Install cameras at junior and senior departments
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))

    # ANALOGY: Run the case through the agency while cameras record
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()

    print(f"✅ Case processed!")
    print(f"   True Label: {label.item()}")
    print(f"   Agency Verdict: {prediction}")
    print(f"   {'✓ CORRECT!' if label.item() == prediction else '✗ INCORRECT'}\n")

    # ANALOGY: Review the security footage from each department
    fig = plt.figure(figsize=(16, 14))

    # ===== ORIGINAL CASE PHOTO =====
    plt.subplot(3, 1, 1)
    plt.imshow(image.cpu().squeeze(), cmap='gray')
    plt.title(f'📁 ORIGINAL CASE PHOTO\n'
              f'True Label: {label.item()} | Agency Verdict: {prediction} '
              f'{"✓" if label.item() == prediction else "✗"}',
              fontsize=14, fontweight='bold', pad=15)
    plt.axis('off')

    # ===== JUNIOR DETECTIVES' PERSPECTIVE =====
    print("📹 Reviewing Junior Detectives' Reports...")
    conv1_act = activations['conv1'].cpu().squeeze()
    plt.subplot(3, 1, 2)

    # ANALOGY: Create a board showing what 16 juniors saw (out of 32 total)
    # Each junior highlights different features they detected
    grid_img = np.zeros((4*28, 4*28))
    for i in range(16):
        row = i // 4
        col = i % 4
        # ANALOGY: This is what Junior Detective #i marked as important
        grid_img[row*28:(row+1)*28, col*28:(col+1)*28] = conv1_act[i].numpy()

    plt.imshow(grid_img, cmap='viridis')
    plt.title('👮 JUNIOR DETECTIVES\' FINDINGS (showing 16 of 32 detectives)\n'
              'Each panel shows what ONE junior detective highlighted\n'
              'Bright areas = "I found something important here!"',
              fontsize=12, pad=15)
    plt.axis('off')

    # Add detective labels
    for i in range(16):
        row = i // 4
        col = i % 4
        plt.text(col*28 + 14, row*28 + 3, f'Det {i+1}',
                ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # ===== SENIOR DETECTIVES' PERSPECTIVE =====
    print("📹 Reviewing Senior Detectives' Reports...")
    conv2_act = activations['conv2'].cpu().squeeze()
    plt.subplot(3, 1, 3)

    # ANALOGY: Board showing what 16 seniors concluded (out of 64 total)
    # Seniors combine junior findings into higher-level insights
    grid_img = np.zeros((4*14, 4*14))
    for i in range(16):
        row = i // 4
        col = i % 4
        # ANALOGY: This is what Senior Detective #i determined is important
        grid_img[row*14:(row+1)*14, col*14:(col+1)*14] = conv2_act[i].numpy()

    plt.imshow(grid_img, cmap='plasma')
    plt.title('👔 SENIOR DETECTIVES\' ANALYSIS (showing 16 of 64 detectives)\n'
              'Each panel shows ONE senior\'s combined analysis\n'
              'Bright areas = "This complex pattern is key to identification!"',
              fontsize=12, pad=15)
    plt.axis('off')

    # Add detective labels
    for i in range(16):
        row = i // 4
        col = i % 4
        plt.text(col*14 + 7, row*14 + 2, f'Sr {i+1}',
                ha='center', va='top', fontsize=7,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    output_path = f'visualizations/feature_maps_digit_{label.item()}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ Saved case processing footage: {output_path}\n")
    print("=" * 70)
    print("💡 INSIGHTS:")
    print("   - Each detective sees the SAME image differently")
    print("   - Juniors detect simple features (edges, curves)")
    print("   - Seniors detect complex patterns (digit shapes)")
    print("   - Together, they identify the digit accurately!")
    print("=" * 70 + "\n")

# Main menu
if __name__ == '__main__':
    """
    ANALOGY: Welcome to the Detective Agency Tour!
    You can:
    1. See what junior detectives learned (their filter specializations)
    2. See what senior detectives learned (their analysis patterns)
    3. Watch a live case being processed step-by-step
    4. Take the full tour (all of the above!)
    """
    os.makedirs('visualizations', exist_ok=True)

    print("=" * 70)
    print("🏢 WELCOME TO THE MNIST DETECTIVE AGENCY")
    print("🎬 Behind-the-Scenes Tour")
    print("=" * 70)
    print("\n👋 Hello! Thanks for visiting our trained detective agency!")
    print("We'd love to show you how our detectives identify handwritten digits.\n")
    print("Tour Options:")
    print("=" * 70)
    print("1. 👮 Visit Junior Detectives (see their 32 specialized skills)")
    print("2. 👔 Visit Senior Detectives (see their 64 analysis patterns)")
    print("3. 🎬 Watch Live Case Processing (see a case flow through the agency)")
    print("4. 🌟 Full Agency Tour (all of the above!)")
    print("=" * 70)

    choice = input("\n🎫 Choose your tour (1-4): ")
    print()

    if choice == '1':
        print("🚶 Heading to the Junior Detectives' Department...\n")
        visualize_conv1_filters()
        print("✅ Tour of Junior Detectives complete!\n")

    elif choice == '2':
        print("🚶 Heading to the Senior Detectives' Department...\n")
        visualize_conv2_filters()
        print("✅ Tour of Senior Detectives complete!\n")

    elif choice == '3':
        print("🚶 Heading to the Live Case Processing Area...\n")
        visualize_feature_maps()
        print("✅ Live processing tour complete!\n")

    elif choice == '4':
        print("🚶 Starting the full agency tour...\n")
        print("=" * 70)
        visualize_conv1_filters()
        input("Press Enter to continue to the next department...")
        print()
        visualize_conv2_filters()
        input("Press Enter to watch live case processing...")
        print()
        visualize_feature_maps()
        print("=" * 70)
        print("\n🎉 FULL TOUR COMPLETE!")
        print("✅ All visualizations have been saved in the 'visualizations/' folder")
        print("📸 You can now see exactly how your agency identifies digits!")
        print("\nThank you for visiting the MNIST Detective Agency! 🏢")
    else:
        print("❌ Invalid tour selection. Please run again and choose 1-4.")

    print("\n" + "=" * 70)
