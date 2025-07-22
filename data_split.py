import os
import shutil
import random

#original image and label folders
image_dir = 'images'  # satellite images
label_dir = 'label'  # roof masks

#new train and test folders
train_image_dir = 'train'
train_label_dir = 'train_mask'
test_image_dir = 'manual_test'
test_label_dir = 'manual_test_mask'

# create directories 
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

images = sorted([img for img in os.listdir(image_dir) if img.endswith(".tif")])
labels = sorted([lbl for lbl in os.listdir(label_dir) if lbl.endswith("_label.tif")])

# ensure images and labels match in naming convention
image_label_pairs = []
for img in images:
    base_name = img.replace(".tif", "")
    label_name = f"{base_name}_label.tif"
    if label_name in labels:
        image_label_pairs.append((img, label_name))
    else:
        print(f"Warning: No label found for image {img}")

# shuffle and split the dataset
random.shuffle(image_label_pairs)
split_ratio = 0.75  # 75% for training, 20% for testing
split_index = int(len(image_label_pairs) * split_ratio)
train_data = image_label_pairs[:split_index]
test_data = image_label_pairs[split_index:]

# func to copy files to a specified directory
def copy_files(data, image_src_dir, label_src_dir, image_dest_dir, label_dest_dir):
    for image_name, label_name in data:
        # source paths
        image_src_path = os.path.join(image_src_dir, image_name)
        label_src_path = os.path.join(label_src_dir, label_name)
        
        # destination paths
        image_dest_path = os.path.join(image_dest_dir, image_name)
        label_dest_path = os.path.join(label_dest_dir, label_name)
        
        # copy files
        shutil.copy(image_src_path, image_dest_path)
        shutil.copy(label_src_path, label_dest_path)

# copy training and test data
copy_files(train_data, image_dir, label_dir, train_image_dir, train_label_dir)
copy_files(test_data, image_dir, label_dir, test_image_dir, test_label_dir)

print("Dataset split complete!")
