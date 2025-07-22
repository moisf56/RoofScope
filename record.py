import torch


from UNET import UNet
from Satellite_Dataset import SatelliteDataset



model = UNet(in_channels=3, num_classes=1)  # Replace with your model structure
checkpoint = torch.load("Models/model_checkpoint.pth")  # Replace with your checkpoint path
model.load_state_dict(checkpoint['model_state_dict'])

torch.save(model.state_dict(), "Models/model_weights.pth")
