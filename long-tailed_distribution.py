import os
import random
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# Custom Dataset class
class SkyViewDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Basic transform (no augmentation)
def get_basic_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# Construct long-tailed training set + test set with original distribution
def load_image_dataloaders(data_root, batch_size=16, test_size=0.2, use_fixed_seed=True, simulate_long_tail=True):
    all_image_paths = []
    all_labels = []
    class_to_idx = {}
    idx_to_class = {}

    # Traverse the directory and collect all image paths and labels
    for idx, class_name in enumerate(sorted(os.listdir(data_root))):
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_to_idx[class_name] = idx
        idx_to_class[idx] = class_name

        image_list = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir)
                      if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

        all_image_paths.extend(image_list)
        all_labels.extend([idx] * len(image_list))

    # Split into train/test + construct long-tailed distribution for training set
    train_paths, test_paths, train_labels, test_labels = [], [], [], []
    for class_idx in set(all_labels):
        class_images = [img for img, label in zip(all_image_paths, all_labels) if label == class_idx]
        class_labels = [class_idx] * len(class_images)

        # Original train/test split
        train_img, test_img, train_lbl, test_lbl = train_test_split(
            class_images,
            class_labels,
            test_size=test_size,
            random_state=42 if use_fixed_seed else None
        )

        # Apply long-tail sampling only to the training set
        if simulate_long_tail:
            keep_ratio = 1.0 / (class_idx + 1)  # Larger class index → fewer samples
            num_keep = max(1, int(len(train_img) * keep_ratio))
            if use_fixed_seed:
                random.seed(42)
            train_img = random.sample(train_img, num_keep)
            train_lbl = [class_idx] * len(train_img)

        train_paths.extend(train_img)
        train_labels.extend(train_lbl)
        test_paths.extend(test_img)
        test_labels.extend(test_lbl)

    # Construct Dataset and DataLoader
    transform = get_basic_transforms()
    train_dataset = SkyViewDataset(train_paths, train_labels, transform=transform)
    test_dataset = SkyViewDataset(test_paths, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, class_to_idx, train_labels, test_labels


# Usage example
if __name__ == '__main__':
    train_loader, test_loader, class_to_idx, train_labels, test_labels = load_image_dataloaders(
        data_root='../data/Aerial_Landscapes/',
        batch_size=16,
        test_size=0.2,
        use_fixed_seed=True,
        simulate_long_tail=True
    )

    # Print class mapping & distributions
    print("Class mapping:", class_to_idx)
    print("Training set class distribution:", Counter(train_labels))
    print("Test set class distribution:", Counter(test_labels))

    # Check DataLoader output
    for images, labels in train_loader:
        print("Image batch shape:", images.shape)
        print("Label batch:", labels)
        break
