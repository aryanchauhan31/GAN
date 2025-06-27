import torch
import torch.nn as nn
from deepspeed.ops.adam import DeepSpeedCPUAdam
import deepspeed
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# ------------------ Generator ------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=128, num_classes=10, feature_map_size=64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, noise_dim)

        self.proj = nn.Sequential(
            nn.Linear(noise_dim * 2, feature_map_size * 8 * 4 * 4),
            nn.BatchNorm1d(feature_map_size * 8 * 4 * 4),
            nn.ReLU(True)
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_size * 2, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, inputs):
        noise, labels = inputs
        label_embedding = self.label_emb(labels)
        x = torch.cat([noise, label_embedding], dim=1)
        x = self.proj(x)
        x = x.view(x.size(0), -1, 4, 4)
        return self.net(x)

# ------------------ Critic ------------------
class Critic(nn.Module):
    def __init__(self, num_classes=10, feature_map_size=64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 28 * 28)

        self.net = nn.Sequential(
            nn.Conv2d(2, feature_map_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 3, 2, 1),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_size * 4, 1, 4, 1, 0)
        )

    def forward(self, inputs):
        img, labels = inputs
        img = torch.nn.functional.interpolate(img, size=(28, 28), mode="bilinear", align_corners=False)
        label_map = self.label_emb(labels).view(-1, 1, 28, 28)
        x = torch.cat([img, label_map], dim=1)
        return self.net(x).view(-1)

# ------------------ Gradient Penalty ------------------
def gradient_penalty(critic, real, fake, labels, device):
    batch_size = real.size(0)
    real = torch.nn.functional.interpolate(real, size=(28, 28), mode="bilinear", align_corners=False)
    fake = torch.nn.functional.interpolate(fake, size=(28, 28), mode="bilinear", align_corners=False)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    mixed_scores = critic((interpolated, labels))
    gradient = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient = gradient.view(batch_size, -1)
    gp = ((gradient.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# ------------------ Training ------------------
def train():
    noise_dim = 128
    batch_size = 64
    lr = 1e-4
    n_epochs = 50
    critic_iters = 5
    lambda_gp = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root=".", train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    gen = Generator(noise_dim=noise_dim).to(device)
    critic = Critic().to(device)

    opt_gen = DeepSpeedCPUAdam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_critic = DeepSpeedCPUAdam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

    gen_engine, _, _, _ = deepspeed.initialize(model=gen, optimizer=opt_gen, config="ds_config.json")
    critic_engine, _, _, _ = deepspeed.initialize(model=critic, optimizer=opt_critic, config="ds_config.json")

    for epoch in range(n_epochs):
        for real, labels in loader:
            real, labels = real.to(device), labels.to(device)
            cur_batch_size = real.size(0)

            for _ in range(critic_iters):
                noise = torch.randn(cur_batch_size, noise_dim, device=device).half()
                fake = gen_engine((noise, labels)).detach()
                critic_real = critic_engine((real, labels))
                critic_fake = critic_engine((fake, labels))
                gp = gradient_penalty(critic_engine, real, fake, labels, device)
                loss_critic = -(critic_real.mean() - critic_fake.mean()) + lambda_gp * gp
                critic_engine.optimizer.zero_grad()
                critic_engine.backward(loss_critic)
                critic_engine.step()

            noise = torch.randn(cur_batch_size, noise_dim, device=device).half()
            fake = gen_engine((noise, labels))
            loss_gen = -critic_engine((fake, labels)).mean()
            gen_engine.optimizer.zero_grad()
            gen_engine.backward(loss_gen)
            gen_engine.step()

        print(f"Epoch [{epoch+1}/{n_epochs}]  Loss D: {loss_critic.item():.4f}  Loss G: {loss_gen.item():.4f}")

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fixed_labels = torch.arange(0, 10, device=device)
                fixed_noise = torch.randn(10, noise_dim, device=device).half()
                fakes = gen_engine((fixed_noise, fixed_labels))
                fakes = (fakes + 1) / 2
                grid = make_grid(fakes, nrow=5, normalize=True)
                plt.imshow(grid.permute(1, 2, 0).cpu())
                plt.title(f"Epoch {epoch+1}")
                plt.axis("off")
                plt.show()

if __name__ == "__main__":
    train()
