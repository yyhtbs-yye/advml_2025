# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# %%
# Set random seed for reproducibility
torch.manual_seed(42)

# %%
# Hyperparameters
batch_size = 128
epochs = 20
learning_rate = 1e-3
embedding_dim = 128  # Dimension of the codebook vectors
num_embeddings = 1024  # Number of vectors in the codebook
beta = 0.25  # Commitment loss weight
kl_weight = 0.1  # Weight for the KL divergence loss
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# %%
# Data Loading
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings  # Size of the codebook
        self.embedding_dim = embedding_dim    # Dimension of each codebook vector
        self.beta = beta                      # Weight for commitment loss
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
    
    def forward(self, z):
        # z shape: [batch, channels, height, width]
        z = z.permute(0, 2, 3, 1).contiguous()  # [batch, h, w, channels]
        z_flat = z.view(-1, self.embedding_dim)  # Flatten to [batch*h*w, embedding_dim]
        
        # Compute distances to codebook vectors
        distances = (z_flat.pow(2).sum(dim=1, keepdim=True) 
                     + self.embedding.pow(2).sum(dim=1)
                     - 2 * torch.matmul(z_flat, self.embedding.t()))
        
        # Find nearest codebook vectors
        indices = torch.argmin(distances, dim=1)  # [batch*h*w]
        quantized_flat = self.embedding[indices]  # [batch*h*w, embedding_dim]
        
        # Reshape back to spatial dimensions
        quantized = quantized_flat.view_as(z)  # [batch, channels, h, w]
        
        # Commitment loss to encourage z to stay close to codebook vectors
        commitment_loss = F.mse_loss(z_flat, quantized_flat.detach())
        codebook_loss = F.mse_loss(quantized_flat, z_flat.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss
        
        # Straight-through estimator: pass gradients through quantization
        quantized = z + (quantized - z).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # [batch, channels, h, w]
        return quantized, vq_loss, indices

# %%
class VQVAE(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=512, beta=0.25):
        super(VQVAE, self).__init__()
        
        # Encoder: Downsample MNIST (28x28) to a smaller spatial size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(32, embedding_dim, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.ReLU()
        )
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, beta)
        
        # Decoder: Upsample back to 28x28
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.Sigmoid()  # Output in [0, 1] for MNIST
        )
    
    def forward(self, x):
        z = self.encoder(x)                  # Encode to latent space
        quantized, vq_loss, indices = self.vq(z)  # Quantize
        x_recon = self.decoder(quantized)    # Decode
        return x_recon, vq_loss, indices
    
    def sample(self, num_samples, device):
        # Randomly sample codebook indices
        indices = torch.randint(0, self.vq.num_embeddings, (num_samples, 7, 7), device=device)
        quantized_flat = self.vq.embedding[indices.view(-1)]  # [num_samples*7*7, embedding_dim]
        quantized = quantized_flat.view(num_samples, 7, 7, self.vq.embedding_dim)  # [num_samples, h, w, channels]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # [num_samples, channels, h, w]
        samples = self.decoder(quantized)
        return samples

# %%
model = VQVAE(embedding_dim=64, num_embeddings=512, beta=0.25).to(device)

# %%
def loss_function(recon_x, x, vq_loss):
    recon_loss = F.mse_loss(recon_x, x)  # Reconstruction loss
    total_loss = recon_loss + vq_loss    # VQ loss includes commitment and codebook losses
    return total_loss, recon_loss

# %%
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%
# Training
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, vq_loss, _ = model(data)
        loss, recon_loss = loss_function(recon_batch, data, vq_loss)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    print(f'Epoch: {epoch}, Average Loss: {train_loss / len(train_loader.dataset):.4f}')

# %%
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    samples = model.sample(16, device)  # Generate 16 samples
    samples = samples.cpu()

# Visualize
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.show()

# %%
# Save the model
torch.save(model.state_dict(), 'vqvae_mnist.pth')


# %%



