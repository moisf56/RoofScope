import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class SatelliteDataset(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        
        # Set directory paths based on whether it's test or train
        if test:
            image_dir = os.path.join(root_path, "manual_test")
            mask_dir = os.path.join(root_path, "manual_test_masks")
        else:
            image_dir = os.path.join(root_path, "train")
            mask_dir = os.path.join(root_path, "train_masks")

        # Initialize lists to store image and mask paths
        self.image_paths = []
        self.mask_paths = []

        # Collect matching images and masks
        for img_name in sorted(os.listdir(image_dir)):
            base_name = img_name.replace(".tif", "")  # Remove file extension
            mask_name = f"{base_name}_label.tif"       # Add "_label" suffix for mask
            img_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, mask_name)

            # Only add pairs if both image and mask exist
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
            else:
                print(f"Warning: Missing pair for {img_name}")

        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load image and mask
        img = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")  # Grayscale mask
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        return img, mask
