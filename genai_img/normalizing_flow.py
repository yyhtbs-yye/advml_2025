import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the affine coupling layer
class AffineCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=256, mask_type='alternate'):
        super(AffineCouplingLayer, self).__init__()
        self.dim = dim
        
        # Create binary mask (0s and 1s)
        if mask_type == 'alternate':
            # Alternating mask: 1, 0, 1, 0, ...
            mask = torch.zeros(dim)
            mask[::2] = 1  # Set every other element to 1
        else:  # 'split'
            # First half 1s, second half 0s
            mask = torch.zeros(dim)
            mask[:dim//2] = 1
            
        # Register the mask as a buffer (not a parameter)
        self.register_buffer('mask', mask.reshape(1, -1))  # Shape: [1, dim] for broadcasting
        
        # Neural network to predict scale and translation factors
        # Input: masked dimensions
        # Output: scale and translation for non-masked dimensions
        self.nn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim * 2)  # 2*dim for scale and translation
        )
    
    def forward(self, x):
        """Forward transformation: x -> z"""
        # x shape: [batch_size, dim]
        batch_size = x.shape[0]
        
        # Apply mask to isolate inputs to the NN
        x_masked = x * self.mask
        
        # Get scale and translation from NN
        nn_out = self.nn(x_masked)
        s, t = torch.chunk(nn_out, 2, dim=1)
        
        # Apply tanh for stability and to bound scale
        s = torch.tanh(s) * 0.5  # Scale factor bounded to [-0.5, 0.5]
        
        # Apply scale and translation only to unmasked dimensions
        z = x.clone()
        z_unmasked = x * (1 - self.mask)
        z = x_masked + (z_unmasked * torch.exp(s) + t) * (1 - self.mask)
        
        # Log determinant of Jacobian
        log_det = torch.sum(s * (1 - self.mask), dim=1)
        
        return z, log_det
    
    def inverse(self, z):
        """Inverse transformation: z -> x"""
        # z shape: [batch_size, dim]
        batch_size = z.shape[0]
        
        # Apply mask to isolate inputs to the NN
        z_masked = z * self.mask
        
        # Get scale and translation from NN
        nn_out = self.nn(z_masked)
        s, t = torch.chunk(nn_out, 2, dim=1)
        
        # Apply tanh for stability and to bound scale
        s = torch.tanh(s) * 0.5  # Scale factor bounded to [-0.5, 0.5]
        
        # Apply inverse scale and translation to unmasked dimensions
        x = z.clone()
        z_unmasked = z * (1 - self.mask)
        x = z_masked + ((z_unmasked - t) * torch.exp(-s)) * (1 - self.mask)
        
        # Log determinant of inverse Jacobian (negative of forward)
        log_det = -torch.sum(s * (1 - self.mask), dim=1)
        
        return x, log_det

# Define the normalizing flow model
class NormalizingFlow(nn.Module):
    def __init__(self, dim, n_flows=4, hidden_dim=256):
        super(NormalizingFlow, self).__init__()
        self.dim = dim
        
        # Create sequence of affine coupling layers with alternating masks
        self.flows = nn.ModuleList([
            AffineCouplingLayer(
                dim, 
                hidden_dim=hidden_dim,
                mask_type='alternate' if i % 2 == 0 else 'split'
            ) for i in range(n_flows)
        ])
        
    def forward(self, x):
        """Transform from data space to latent space: x -> z"""
        batch_size = x.shape[0]
        log_det_total = torch.zeros(batch_size, device=x.device)
        
        z = x
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total += log_det
            
        return z, log_det_total
    
    def inverse(self, z):
        """Transform from latent space to data space: z -> x"""
        batch_size = z.shape[0]
        log_det_total = torch.zeros(batch_size, device=z.device)
        
        x = z
        # Apply flows in reverse order
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x)
            log_det_total += log_det
            
        return x, log_det_total
    
    def log_prob(self, x):
        """Compute log probability of x under the flow-based model"""
        z, log_det = self.forward(x)
        
        # Standard Gaussian prior
        log_prob_z = -0.5 * torch.sum(z**2 + np.log(2*np.pi), dim=1)
        
        # Change of variables formula
        log_prob_x = log_prob_z + log_det
        
        return log_prob_x
    
    def sample(self, n_samples):
        """Generate samples from the flow-based model"""
        # Sample from standard Gaussian prior
        z = torch.randn(n_samples, self.dim, device=next(self.parameters()).device)
        
        # Transform to data space
        x, _ = self.inverse(z)
        
        # For MNIST: ensure values are in [0, 1]
        x = torch.clamp(x, 0.0, 1.0)
        
        return x

# Load MNIST dataset
def load_mnist_data(n_samples=1000):
    from torchvision import datasets, transforms
    
    # Transform the images to tensors and flatten them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the 28x28 images
    ])
    
    # Load the dataset
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Select a subset of the data
    indices = torch.randperm(len(mnist_dataset))[:n_samples]
    X = torch.stack([mnist_dataset[i][0] for i in indices])
    
    return X

# Training function
def train_flow(flow, data, n_epochs=25, batch_size=128, lr=1e-3):
    optimizer = optim.Adam(flow.parameters(), lr=lr)
    n_samples = data.shape[0]
    
    losses = []
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = torch.randperm(n_samples)
        data_shuffled = data[indices]
        
        epoch_loss = 0.0
        n_batches = 0
        
        # Batch training
        for i in range(0, n_samples, batch_size):
            batch = data_shuffled[i:i+batch_size]
            
            # Compute log probability
            log_prob = flow.log_prob(batch)
            
            # Negative log-likelihood loss
            loss = -torch.mean(log_prob)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    return losses

# Create and train the model
print("Loading MNIST data...")
data = load_mnist_data(n_samples=2000)
print(f"MNIST data shape: {data.shape}")

print("Creating normalizing flow model...")
flow = NormalizingFlow(dim=784, n_flows=4, hidden_dim=256)
print("Training...")
losses = train_flow(flow, data, n_epochs=20)

# Plot the loss curve
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Negative Log-Likelihood")
plt.grid(True)
plt.show()

# Display original MNIST images and generated samples
plt.figure(figsize=(15, 6))

# Plot original data
plt.subplot(2, 4, 1)
plt.imshow(data[0].reshape(28, 28).numpy(), cmap='gray')
plt.title("Original 1")
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(data[1].reshape(28, 28).numpy(), cmap='gray')
plt.title("Original 2")
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(data[2].reshape(28, 28).numpy(), cmap='gray')
plt.title("Original 3")
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(data[3].reshape(28, 28).numpy(), cmap='gray')
plt.title("Original 4")
plt.axis('off')

# Plot samples from the model
print("Generating samples...")
with torch.no_grad():
    samples = flow.sample(4)
    
plt.subplot(2, 4, 5)
plt.imshow(samples[0].reshape(28, 28).numpy(), cmap='gray')
plt.title("Generated 1")
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(samples[1].reshape(28, 28).numpy(), cmap='gray')
plt.title("Generated 2")
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(samples[2].reshape(28, 28).numpy(), cmap='gray')
plt.title("Generated 3")
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(samples[3].reshape(28, 28).numpy(), cmap='gray')
plt.title("Generated 4")
plt.axis('off')

plt.tight_layout()
plt.show()

# Visualize interpolation in latent space
plt.figure(figsize=(15, 3))

# Sample two points from the base distribution
with torch.no_grad():
    z1 = torch.randn(1, 784)
    z2 = torch.randn(1, 784)
    
    # Create interpolations
    alphas = np.linspace(0, 1, 8)
    for i, alpha in enumerate(alphas):
        z_interp = alpha * z1 + (1 - alpha) * z2
        x_interp, _ = flow.inverse(z_interp)
        x_interp = torch.clamp(x_interp, 0.0, 1.0)
        
        plt.subplot(1, 8, i+1)
        plt.imshow(x_interp[0].reshape(28, 28).numpy(), cmap='gray')
        plt.axis('off')
        
plt.tight_layout()
plt.suptitle('Latent Space Interpolation')
plt.subplots_adjust(top=0.85)
plt.show()