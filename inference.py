import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from Satellite_Dataset import SatelliteDataset
from UNET import UNet

def multiple_image_predict(data_path, model_path, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    image_dataset = SatelliteDataset(data_path, test=True)

    images = []
    original_masks = []
    prediction_masks = []

    for img, original_mask in image_dataset:
        img = img.float().to(device).unsqueeze(0)  # Add batch dimension
        prediction_mask = model(img)

        img = img.squeeze(0).cpu().detach().permute(1, 2, 0)  # Revert to HWC format
        prediction_mask = prediction_mask.squeeze(0).cpu().detach().permute(1, 2, 0)
        prediction_mask = (prediction_mask > 0).float()  # Convert to binary mask

        original_mask = original_mask.cpu().detach().permute(1, 2, 0)

        images.append(img)
        original_masks.append(original_mask)
        prediction_masks.append(prediction_mask)

    # Display images, original masks, and predictions
    fig = plt.figure(figsize=(15, 5))
    for idx in range(len(image_dataset)):
        fig.add_subplot(3, len(image_dataset), idx + 1)
        plt.imshow(images[idx], cmap="gray")
        
        fig.add_subplot(3, len(image_dataset), len(image_dataset) + idx + 1)
        plt.imshow(original_masks[idx], cmap="gray")
        
        fig.add_subplot(3, len(image_dataset), 2 * len(image_dataset) + idx + 1)
        plt.imshow(prediction_masks[idx], cmap="gray")
    
    plt.show()


def single_image_predict(image_pth, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=device))
    model.eval()  # Set model to evaluation mode

    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(Image.open(image_pth)).float().to(device).unsqueeze(0)

    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach().permute(1, 2, 0)
    pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1, 2, 0)
    pred_mask[pred_mask < 0] = 0      # Convert to binary mask
    pred_mask[pred_mask > 0] = 1
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    fig.add_subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap="gray")
    plt.show()

if __name__ == "__main__":
    single_img_PTH = "data/manual_test/tile1_2000_5600.tif"
    data_PTH = "data"
    model_PTH = "Models/model_weights.pth"
    img_PTH2 = "data/manual_test/tile1_3600_8000.tif"
    img_PTH3 = "data/manual_test/tile_2800_5200.tif"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    single_image_predict(single_img_PTH, model_PTH, device)
    single_image_predict(img_PTH2, model_PTH, device)
    single_image_predict(img_PTH3, model_PTH, device)
