import matplotlib.pyplot as plt

# Data from logs
train_losses = [0.2586, 0.2235, 0.2188, 0.2073, 0.1956, 0.1814, 0.1751]
val_losses = [0.3779, 0.2497, 0.2309, 0.2349, 0.2032, 0.1890, 0.1615]
epochs = range(1, len(train_losses) + 1)

# Plotting the losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss', marker='o')
plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Plotting with dual y-axes for training and validation losses
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting training loss on the left y-axis
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color='tab:blue')
ax1.plot(epochs, train_losses, label='Training Loss', marker='o', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Creating a second y-axis for validation loss
ax2 = ax1.twinx()
ax2.set_ylabel('Validation Loss', color='tab:red')
ax2.plot(epochs, val_losses, label='Validation Loss', marker='o', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Adding titles and grid
plt.title('Training and Validation Loss per Epoch')
fig.tight_layout()
plt.grid(True)
plt.show()
