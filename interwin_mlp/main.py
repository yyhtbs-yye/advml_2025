import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the neural network
class CustomMNISTModel(nn.Module):
    def __init__(self):
        super(CustomMNISTModel, self).__init__()
        
        # First path (main)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)

        # Second path (skip connection)
        self.skip_fc1 = nn.Linear(784, 32)

        # Processing after concatenation
        self.concat_fc = nn.Linear(64, 128)
        self.skip_fc2 = nn.Linear(32, 128)

        # Final output layer
        self.final_fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input (28x28 -> 784)

        # First path (main)
        out1 = torch.relu(self.fc1(x))
        out1 = torch.relu(self.fc2(out1))

        # Second path (skip connection)
        out2 = torch.relu(self.skip_fc1(x))

        # Concatenation
        concat_out = torch.cat((out1, out2), dim=1)

        # Processing concatenated output
        processed_concat = torch.relu(self.concat_fc(concat_out))

        # Further processing of skip connection
        processed_skip = torch.relu(self.skip_fc2(out2))

        # Add the processed outputs
        final_addition = processed_concat + processed_skip

        # Final output layer
        output = self.final_fc(final_addition)
        
        return output

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomMNISTModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'custom_mnist_model.pth')
print("Model saved successfully.")
