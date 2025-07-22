import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from Satellite_Dataset import SatelliteDataset
from UNET import UNet
import numpy as np

def save_binary_mask(mask, filename):
    # threshold to convert probabilities to binary
    thresholded_mask = (mask > 0.5).float().squeeze().cpu().numpy()
    binary_mask = (thresholded_mask * 255).astype(np.uint8)  # scale to 0-255
    Image.fromarray(binary_mask).save(filename)

def calculate_accuracy(pred_mask: torch.Tensor, true_mask: torch.Tensor):
    pred_mask = pred_mask > 0.5  # threshold to get binary predictions
    true_mask = true_mask > 0.5  # Convert to boolean
    correct = pred_mask == true_mask  # Element-wise comparison
    accuracy = correct.float().mean()  # percentage of correct pixels
    return accuracy

def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, epsilon=1e-6):
    pred_mask = pred_mask > 0.5  # Apply threshold to get binary predictions
    true_mask = true_mask > 0.5  # Convert to boolean
    intersection = (pred_mask & true_mask).float().sum((1, 2))  # Intersection
    union = (pred_mask | true_mask).float().sum((1, 2))         # Union
    iou = (intersection + epsilon) / (union + epsilon)  # To avoid division by zero
    return iou.mean()  # Average

def evaluate_model_on_manual_test(data_path, mask_path, model_path, output_dir, batch_size=8):
    # Load the model
    print("Loading the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # set the model to evaluation mode
    print("Model is loaded!")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    print("Loading the dataset...")
    test_dataset = SatelliteDataset(data_path, test=True) 	# load the dataset and set it to the test mode 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Dataset is loaded!")

    print("Begin inference...")
    with torch.no_grad():  # no need to compute gradients during evaluation
        total_accuracy = 0.0
        total_iou = 0.0
        num_batches = 0

        for batch_idx, (images, true_masks) in enumerate(test_loader):
            images, true_masks = images.to(device), true_masks.to(device)
            
            # Predict
            pred_masks = model(images)
            pred_masks = torch.sigmoid(pred_masks)  #sigmoid to get probabilities
            
            # Save each mask in the batch
            for i in range(pred_masks.size(0)):
                save_binary_mask(pred_masks[i], os.path.join(output_dir, f"pred_{batch_idx * batch_size + i}.png"))

            # Calculate accuracy and IoU for the batch
            batch_accuracy = calculate_accuracy(pred_masks, true_masks)
            batch_iou = calculate_iou(pred_masks, true_masks)

            total_accuracy += batch_accuracy
            total_iou += batch_iou
            num_batches += 1

            print(f"\tBatch {num_batches} Accuracy: {batch_accuracy:.4f} IoU {batch_iou:.4f}")

    #average metrics
    avg_accuracy = total_accuracy / num_batches
    avg_iou = total_iou / num_batches

    print("============================================================================")
    print(f"\nAverage Pixel Accuracy on Manual Test Set: {avg_accuracy:.4f}")
    print(f"Average IoU on Manual Test Set: {avg_iou:.4f}")
    print("============================================================================")

if __name__ == "__main__":
    # paths to your test dataset and model weights
    data_path = "data"
    mask_path = "data/manual_test_masks"
    model_path = "Models/model_weights_2024-14-11.pth"
    output_dir = "output_masks"

    #evaluation
    evaluate_model_on_manual_test(data_path, mask_path, model_path, output_dir, batch_size=8)
