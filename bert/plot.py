import matplotlib.pyplot as plt
from model import train_losses, test_losses, accuracies, num_epochs
# Plotting the loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.tight_layout()

# Save the loss plot as an image
plt.savefig('loss_plot.png')
plt.show()

# Plotting the accuracy
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), accuracies, label='Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()

# Save the accuracy plot as an image
plt.savefig('accuracy_plot.png')
plt.show()