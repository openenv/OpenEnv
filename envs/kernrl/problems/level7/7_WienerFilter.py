"""
Wiener Filter (Frequency Domain Deconvolution)

Deconvolution filter that estimates original signal from blurred/noisy observation.
Optimal linear filter in the MSE sense for additive noise.

H_wiener = H* / (|H|^2 + SNR^-1)

where H is the blur kernel's frequency response.

Optimization opportunities:
- Fused FFT + filter + IFFT
- Shared memory for frequency components
- Real-to-complex optimizations
- Batched processing
"""

import torch
import torch.fft
import torch.nn as nn


class Model(nn.Module):
    """
    Wiener deconvolution filter.

    Given a blurred image and blur kernel, estimates the original image.
    """

    def __init__(self, kernel_size: int = 15, noise_var: float = 0.01):
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.noise_var = noise_var

        # Gaussian blur kernel (typical PSF)
        x = torch.arange(kernel_size).float() - kernel_size // 2
        y = torch.arange(kernel_size).float() - kernel_size // 2
        X, Y = torch.meshgrid(x, y, indexing="ij")
        sigma = kernel_size / 6
        kernel = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        self.register_buffer("blur_kernel", kernel)

    def forward(self, blurred: torch.Tensor) -> torch.Tensor:
        """
        Apply Wiener deconvolution.

        Args:
            blurred: (H, W) blurred image

        Returns:
            restored: (H, W) deconvolved image
        """
        H, W = blurred.shape

        # Pad kernel to image size
        kernel_padded = torch.zeros(H, W, device=blurred.device)
        kh, kw = self.blur_kernel.shape
        kernel_padded[:kh, :kw] = self.blur_kernel
        # Center the kernel (circular shift)
        kernel_padded = torch.roll(kernel_padded, (-kh // 2, -kw // 2), dims=(0, 1))

        # FFT of blurred image and kernel
        G = torch.fft.fft2(blurred)
        H_freq = torch.fft.fft2(kernel_padded)

        # Wiener filter: H* / (|H|^2 + noise_var)
        H_conj = torch.conj(H_freq)
        H_mag_sq = torch.abs(H_freq) ** 2

        # Estimate signal variance (simple: use image variance)
        signal_var = blurred.var()
        snr_inv = self.noise_var / (signal_var + 1e-10)

        # Wiener filter
        W = H_conj / (H_mag_sq + snr_inv)

        # Apply filter and inverse FFT
        F_est = G * W
        restored = torch.fft.ifft2(F_est).real

        return restored


# Problem configuration
image_height = 1024
image_width = 1024


def get_inputs():
    # Simulated blurred image
    image = torch.rand(image_height, image_width)
    return [image]


def get_init_inputs():
    return [15, 0.01]  # kernel_size, noise_var
