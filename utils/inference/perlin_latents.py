import torch
import numpy as np
from noise import pnoise1

class PerlinNoiseGenerator:
    def __init__(self, seed: int = 42, step_size: float = 1.0):
        self.seed = seed
        self.t = 0 
        self.step_size = step_size

    def step(self) -> None:
        self.t += self.step_size

    def generate_perlin_noise(self, z_flat: torch.Tensor) -> torch.Tensor:
        return torch.tensor([pnoise1(self.t + float(i), self.seed) for i in z_flat.tolist()]).float().to(z_flat.device)

    def perlin_latent(self, z, amt, device):
        if not (0 <= amt <= 1):
            raise ValueError("amt must be between 0 and 1")
        original_type = type(z)
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32, device=device)
        elif isinstance(z, list):
            z = torch.tensor(z, dtype=torch.float32, device=device)
        elif not isinstance(z, torch.Tensor):
            raise TypeError("Unsupported type for z")

        z_flat = z.flatten()
        perlin_noise = self.generate_perlin_noise(z_flat)
        z_with_noise = amt * perlin_noise + (1 - amt) * z_flat

        if original_type == np.ndarray:
            return z_with_noise.cpu().numpy().reshape(z.shape)
        elif original_type == list:
            return z_with_noise.cpu().tolist()
        else:
            return z_with_noise.view(z.size())


if __name__ == "__main__":
    generator = PerlinNoiseGenerator(seed=42)
    for data_type in [torch.tensor, np.array, list]:
        z = data_type([0.1, 0.2, 0.3, 0.4, 0.5])
        z_with_noise = generator.perlin_latent(z, 0.5)
        assert len(z_with_noise) == len(z)
        generator.step()
        print(f"PerlinNoiseGenerator test passed, data_type: {data_type.__name__}")

    # for i in range(100):
    #     z = torch.randn(2)
    #     z_with_noise = generator.perlin_latent(z, 0.5)
    #     print(z_with_noise)
    #     generator.step()