import os
import random
import numpy as np
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch

# --- transform ---
def get_basic_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_strong_transforms():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.3),
        A.GaussNoise(p=0.2),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# --- dataset ---
class SkyViewDataset(Dataset):
    def __init__(self, image_paths, labels, transform_basic, transform_strong, minority_classes):
        self.image_paths = image_paths
        self.labels = labels
        self.transform_basic = transform_basic
        self.transform_strong = transform_strong
        self.minority_classes = set(minority_classes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        label = self.labels[idx]

        if label in self.minority_classes:
            image = self.transform_strong(image=image)["image"]
        else:
            image = self.transform_basic(image=image)["image"]

        return image, label

# --- dataloader ---
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

    # Train/test split + simulate long-tailed distribution
    train_paths, test_paths, train_labels, test_labels = [], [], [], []
    for class_idx in set(all_labels):
        class_images = [img for img, label in zip(all_image_paths, all_labels) if label == class_idx]
        class_labels = [class_idx] * len(class_images)

        train_img, test_img, train_lbl, test_lbl = train_test_split(
            class_images,
            class_labels,
            test_size=test_size,
            random_state=42 if use_fixed_seed else None
        )

        if simulate_long_tail:
            keep_ratio = 1.0 / (class_idx + 1)
            num_keep = max(1, int(len(train_img) * keep_ratio))
            if use_fixed_seed:
                random.seed(42)
            train_img = random.sample(train_img, num_keep)
            train_lbl = [class_idx] * len(train_img)

        train_paths.extend(train_img)
        train_labels.extend(train_lbl)
        test_paths.extend(test_img)
        test_labels.extend(test_lbl)

    # Identify minority classes (those with counts less than the median)
    label_counts = Counter(train_labels)
    median_count = np.median(list(label_counts.values()))
    minority_classes = [cls for cls, count in label_counts.items() if count < median_count]

    # Define transforms
    transform_basic = get_basic_transforms()
    transform_strong = get_strong_transforms()

    # Create Datasets
    train_dataset = SkyViewDataset(
        train_paths, train_labels,
        transform_basic=transform_basic,
        transform_strong=transform_strong,
        minority_classes=minority_classes
    )

    test_dataset = SkyViewDataset(
        test_paths, test_labels,
        transform_basic=transform_basic,
        transform_strong=transform_strong,
        minority_classes=[]  # No augmentation for test set
    )

    # WeightedRandomSampler for oversampling
    class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, class_to_idx, train_labels, test_labels


# --- Test run ---
if __name__ == '__main__':
    train_loader, test_loader, class_to_idx, train_labels, test_labels = load_image_dataloaders(
        data_root='../data/Aerial_Landscapes',
        batch_size=16,
        test_size=0.2,
        use_fixed_seed=True,
        simulate_long_tail=True
    )

    print("Class mapping:", class_to_idx)
    print("Training set class distribution:", Counter(train_labels))
    print("Test set class distribution:", Counter(test_labels))

    for images, labels in train_loader:
        print("Image batch shape:", images.shape)
        print("Label batch:", labels)
        break
