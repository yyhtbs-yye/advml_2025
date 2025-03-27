# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# %%
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

# %%
class MinibatchStdDev(nn.Module):
    def __init__(self):
        super(MinibatchStdDev, self).__init__()
    
    def forward(self, x):
        batch_size, _, height, width = x.shape
        # Calculate standard deviation across batch
        y = x - x.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2).mean(dim=0, keepdim=False) + 1e-8)
        y = y.mean().view(1, 1, 1, 1)
        y = y.repeat(batch_size, 1, height, width)
        # Append as new channel
        return torch.cat([x, y], dim=1)

# %%
class PGGenerator(nn.Module):
    def __init__(self, latent_dim=512, output_channels=1):
        super(PGGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.current_stage = 0
        self.alpha = 1.0  # Start with fully stabilized network
        
        # Initial block (7x7)
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(latent_dim, 128, 7, 1, 0),  # 7x7 output
            nn.LeakyReLU(0.2),
            PixelNorm(),
            nn.Conv2d(128, 128, 3, 1, 1),  # 7x7 maintained
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        
        # ToRGB blocks for each resolution
        self.to_rgb_blocks = nn.ModuleList([
            nn.Conv2d(128, output_channels, 1, 1, 0),  # 7x7 -> RGB
            nn.Conv2d(64, output_channels, 1, 1, 0),   # 14x14 -> RGB
            nn.Conv2d(32, output_channels, 1, 1, 0)    # 28x28 -> RGB
        ])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            # 7x7 -> 14x14
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(128, 64, 3, 1, 1),  # 14x14
                nn.LeakyReLU(0.2),
                PixelNorm(),
                nn.Conv2d(64, 64, 3, 1, 1),   # 14x14
                nn.LeakyReLU(0.2),
                PixelNorm()
            ),
            # 14x14 -> 28x28
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(64, 32, 3, 1, 1),   # 28x28
                nn.LeakyReLU(0.2),
                PixelNorm(),
                nn.Conv2d(32, 32, 3, 1, 1),   # 28x28
                nn.LeakyReLU(0.2),
                PixelNorm()
            )
        ])
    
    def forward(self, z):
        # Reshape latent vector to 2D
        x = z.view(z.size(0), self.latent_dim, 1, 1)
        
        # Initial block
        x = self.initial(x)
        
        # Return image based on current stage and alpha
        if self.current_stage == 0:
            # Stage 0: 7x7 resolution
            return self.to_rgb_blocks[0](x)
        else:
            # Process through upsampling blocks up to current stage
            for i in range(self.current_stage):
                features_prev = x
                x = self.up_blocks[i](x)
                
                # During transition phase for the current stage
                if i == self.current_stage - 1 and self.alpha < 1:
                    # Blend with upsampled lower resolution
                    y = F.interpolate(features_prev, scale_factor=2, mode='nearest')
                    y = self.to_rgb_blocks[i](y)
                    out = self.to_rgb_blocks[i+1](x)
                    return (1 - self.alpha) * y + self.alpha * out
            
            # After transition: use higher resolution directly
            return self.to_rgb_blocks[self.current_stage](x)
        
    def progress(self):
        """Progress to the next stage if not at final stage"""
        if self.current_stage < len(self.up_blocks):
            self.current_stage += 1
            self.alpha = 0.0  # Start transitioning from the previous stage
            
    def update_alpha(self, increment=0.1):
        """Update the alpha value for smooth transition"""
        self.alpha = min(1.0, self.alpha + increment)

# %%
# Progressive Discriminator
class PGDiscriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(PGDiscriminator, self).__init__()
        self.input_channels = input_channels
        self.current_stage = 0
        self.alpha = 1.0  # Start with fully stabilized network
        
        # FromRGB blocks for each resolution
        self.from_rgb_blocks = nn.ModuleList([
            nn.Conv2d(input_channels, 128, 1, 1, 0),  # RGB -> 7x7 features
            nn.Conv2d(input_channels, 64, 1, 1, 0),   # RGB -> 14x14 features
            nn.Conv2d(input_channels, 32, 1, 1, 0)    # RGB -> 28x28 features
        ])
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            # 14x14 -> 7x7
            nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),   # 14x14
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 3, 1, 1),  # 14x14
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)  # 7x7
            ),
            # 28x28 -> 14x14
            nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),   # 28x28
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 3, 1, 1),   # 28x28
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)  # 14x14
            ),
        ])
        
        # Final block (7x7 -> decision)
        self.final = nn.Sequential(
            MinibatchStdDev(),
            nn.Conv2d(128 + 1, 128, 3, 1, 1),  # +1 for minibatch std
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Process based on current stage
        if self.current_stage == 0:
            # Stage 0: 7x7 resolution
            features = self.from_rgb_blocks[0](x)
            return self.final(features)
        else:
            # Start from the highest resolution
            current_res_idx = self.current_stage
            features = self.from_rgb_blocks[current_res_idx](x)
            
            # Process through downsampling blocks from highest to lowest
            for i in range(current_res_idx-1, -1, -1):
                if i == current_res_idx-1 and self.alpha < 1:
                    # During transition: blend with downsampled input
                    x_down = F.avg_pool2d(x, 2)
                    y = self.from_rgb_blocks[current_res_idx-1](x_down)
                    features = self.down_blocks[current_res_idx-1](features)
                    features = (1 - self.alpha) * y + self.alpha * features
                else:
                    # After transition: downsample features directly
                    features = self.down_blocks[i](features)
            
            # Final decision
            return self.final(features)
    
    def progress(self):
        """Progress to the next stage if not at final stage"""
        if self.current_stage < len(self.down_blocks):
            self.current_stage += 1
            self.alpha = 0.0  # Start transitioning
            
    def update_alpha(self, increment=0.1):
        """Update the alpha value for smooth transition"""
        self.alpha = min(1.0, self.alpha + increment)


# %%
# PGGAN wrapper class
class PGGAN(nn.Module):
    def __init__(self, latent_dim=512, input_channels=1):
        super(PGGAN, self).__init__()
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        
        self.generator = PGGenerator(latent_dim, input_channels)
        self.discriminator = PGDiscriminator(input_channels)
        self.current_stage = 0
        self.resolutions = [7, 14, 28]  # Specific resolutions for MNIST
        
    def progress(self):
        """Progress both generator and discriminator to next stage"""
        if self.current_stage < len(self.resolutions) - 1:
            self.current_stage += 1
            self.generator.progress()
            self.discriminator.progress()
            print(f"Progressed to stage {self.current_stage}, resolution {self.resolutions[self.current_stage]}x{self.resolutions[self.current_stage]}")
            
    def update_alpha(self, increment=0.1):
        """Update alpha for smooth transition in both networks"""
        self.generator.update_alpha(increment)
        self.discriminator.update_alpha(increment)
    
    def get_current_resolution(self):
        """Get current resolution based on stage"""
        return self.resolutions[self.current_stage]
    
    def sample(self, n_samples=16, device='cpu'):
        """Generate samples at current resolution"""
        # Generate random noise
        z = torch.randn(n_samples, self.latent_dim).to(device)
        # Generate fake images
        with torch.no_grad():
            return self.generator(z)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# %%
# Define transforms for progressive resolutions
transform_dict = {
    7: transforms.Compose([
        transforms.Resize(7),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    14: transforms.Compose([
        transforms.Resize(14),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    28: transforms.Compose([
        transforms.ToTensor(),  # Original MNIST size
        transforms.Normalize([0.5], [0.5])
    ])
}


# %%
# Training parameters
latent_dim = 512
batch_sizes = {7: 128, 14: 64, 28: 32}  # Smaller batch size for higher resolution
iterations_per_stage = 500
sample_interval = 500
alpha_update_interval = 50  # Update alpha every N iterations
transition_iterations = 2000  # Number of iterations for transition (alpha from 0 to 1)

# %%
# Initialize PGGAN
pggan = PGGAN(latent_dim=latent_dim, input_channels=1)
pggan.to(device)


# %%
# Loss function
adversarial_loss = nn.BCELoss()

# %%
# Training loop with progressive growing
# Optimizers
optimizer_G = optim.Adam(pggan.generator.parameters(), lr=0.0001, betas=(0.0, 0.99))
optimizer_D = optim.Adam(pggan.discriminator.parameters(), lr=0.0001, betas=(0.0, 0.99))


# %%
# Training for each stage
for stage in range(0, 3):  # Three stages: 7x7, 14x14, and 28x28
    # Set current stage if not already set
    if pggan.current_stage != stage:
        pggan.current_stage = stage
        pggan.generator.current_stage = stage
        pggan.discriminator.current_stage = stage
    
    resolution = pggan.get_current_resolution()
    print(f"Starting training at resolution {resolution}x{resolution}")
    
    # Create dataset for current resolution
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_dict[resolution]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_sizes[resolution], 
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    # Create infinite data loader
    data_iter = iter(train_loader)
    
    # Training loop for current stage
    for iteration in range(iterations_per_stage):
        # Get batch (restart if we've gone through the dataset)
        try:
            real_imgs, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            real_imgs, _ = next(data_iter)
            
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        
        # Create labels
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        
        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()
        
        # Generate noise
        z = torch.randn(batch_size, latent_dim).to(device)
        
        # Generate fake images
        gen_imgs = pggan.generator(z)
        
        # Try to fool the discriminator
        g_loss = adversarial_loss(pggan.discriminator(gen_imgs), valid)
        
        g_loss.backward()
        optimizer_G.step()
        
        # -----------------
        # Train Discriminator
        # -----------------
        optimizer_D.zero_grad()
        
        # Measure discriminator's ability to classify real images
        real_loss = adversarial_loss(pggan.discriminator(real_imgs), valid)
        
        # Measure discriminator's ability to classify fake images
        fake_loss = adversarial_loss(pggan.discriminator(gen_imgs.detach()), fake)
        
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        optimizer_D.step()
        
        # Update alpha value during transition phase
        if stage > 0 and iteration < transition_iterations and iteration % alpha_update_interval == 0:
            alpha_increment = 1.0 / (transition_iterations / alpha_update_interval)
            pggan.update_alpha(alpha_increment)
        
        # Print progress
        if iteration % 100 == 0:
            print(f"Stage {stage}, Resolution {resolution}x{resolution}, Iteration {iteration}/{iterations_per_stage}, Alpha: {pggan.generator.alpha:.3f}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        
        # Save generated samples at intervals
        if iteration % sample_interval == 0:
            # Generate samples
            with torch.no_grad():
                samples = pggan.sample(16, device)
                # Denormalize samples
                samples = (samples * 0.5) + 0.5
                
                # Plot samples
                plt.figure(figsize=(4, 4))
                for j in range(16):
                    plt.subplot(4, 4, j + 1)
                    plt.imshow(samples[j].cpu().squeeze().numpy(), cmap='gray')
                    plt.axis('off')
                plt.tight_layout()
                plt.show()
    
    # Progress to next stage (if not the final stage)
    if stage < 2:  # We have 3 stages (0, 1, 2)
        pggan.progress()
        # Reset optimizers for next stage with adjusted learning rate
        optimizer_G = optim.Adam(pggan.generator.parameters(), lr=0.001 * (0.5 ** stage), betas=(0.0, 0.99))
        optimizer_D = optim.Adam(pggan.discriminator.parameters(), lr=0.001 * (0.5 ** stage), betas=(0.0, 0.99))


# %%

with torch.no_grad():
    pggan.generator.eval()
    samples = pggan.sample(16, device)
    # Denormalize samples
    samples = (samples * 0.5) + 0.5
    
    # Plot samples
    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(samples[i].cpu().squeeze().numpy(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('pggan_final_samples.png')
    plt.close()


# %%
# Save model
torch.save({
    'generator_state_dict': pggan.generator.state_dict(),
    'discriminator_state_dict': pggan.discriminator.state_dict(),
    'current_stage': pggan.current_stage
}, 'pggan_model.pth')


# %%
# Function to load and use the model
def load_and_generate(model_path, num_samples=16, device='cpu'):
    # Initialize a new model
    pggan = PGGAN(latent_dim=512, input_channels=1).to(device)
    
    # Load the saved model
    checkpoint = torch.load(model_path, map_location=device)
    pggan.generator.load_state_dict(checkpoint['generator_state_dict'])
    pggan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    pggan.current_stage = checkpoint['current_stage']
    pggan.generator.current_stage = pggan.current_stage
    pggan.discriminator.current_stage = pggan.current_stage
    
    # Generate samples
    with torch.no_grad():
        pggan.generator.eval()
        samples = pggan.sample(num_samples, device)
        samples = (samples * 0.5) + 0.5
    
    return samples

# %%



