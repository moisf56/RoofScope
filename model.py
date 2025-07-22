import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


from UNET import UNet
from Satellite_Dataset import SatelliteDataset


if __name__ == "__main__":                          #hyperparameters assignment
    LEARNING_RATE = 0.001
    EPOCHS = 7
    BATCH_SIZE = 8
    data_path = "data" 
    Model_Save_path = "Models/model_checkpoint.pth"  

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = SatelliteDataset(data_path)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.7, 0.3], generator=generator)               #split the training data into train data and validation data

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
 

    val_data_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)



    model = UNet(in_channels=3, num_classes=1).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCEWithLogitsLoss()              #can use other types of losses?
    scaler = GradScaler()

    #training loop 
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0
        for idx, img_mask in enumerate(tqdm(train_data_loader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_predict = model(img)              #get the prediction
            optimizer.zero_grad()
            with autocast():
                loss = loss_function(y_predict, mask)              #comput the loss
                train_loss += loss.item()                       #increment running loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


        train_loss_final = train_loss / (idx+1)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            #validation loop
            for idx, img_mask in enumerate(tqdm(val_data_loader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)                


                y_predict = model(img) 
                loss = loss_function(y_predict, mask)

                val_loss += loss.item()

            val_loss_final = val_loss / (idx+1)


        print("-"*50)
        print(f"Train Loss For Epoch {epoch+1} is : {train_loss_final:.4f}")
        print(f"Validation Loss For Epoch {epoch+1} is : {val_loss_final:.4f}")
        print("-"*50)



checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss_final': train_loss_final,  # Optional
    'val_loss_final': val_loss_final       # Optional
}
torch.save(checkpoint, Model_Save_path)

