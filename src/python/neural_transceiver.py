import torch
import torch.nn as nn


class CaduceusAE(nn.Module):
    def __init__(self):
        super(CaduceusAE, self).__init__()
        # Encoder: 16-one-hot -> 2D IQ points
        self.encoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        # Decoder: 2D IQ points -> 16-class prediction
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    @staticmethod
    def _to_complex(iq_points):
        return torch.complex(iq_points[:, 0], iq_points[:, 1])

    @staticmethod
    def _from_complex(complex_signal):
        return torch.stack((complex_signal.real, complex_signal.imag), dim=1)

    def _apply_channel(self, iq_points, noise_std, channel_type):
        noise_std_t = torch.tensor(noise_std, device=iq_points.device, dtype=iq_points.dtype)
        complex_iq = self._to_complex(iq_points)

        if channel_type == "awgn":
            faded = complex_iq
        elif channel_type == "rayleigh":
            h_real = torch.randn(complex_iq.shape, device=iq_points.device, dtype=iq_points.dtype)
            h_imag = torch.randn(complex_iq.shape, device=iq_points.device, dtype=iq_points.dtype)
            h = torch.complex(h_real, h_imag) / torch.sqrt(torch.tensor(2.0, device=iq_points.device, dtype=iq_points.dtype))
            faded = h * complex_iq
        elif channel_type == "rician":
            # Rician fading with K-factor=3 (dominant LOS + scattered component)
            k_factor = torch.tensor(3.0, device=iq_points.device, dtype=iq_points.dtype)
            los_scale = torch.sqrt(k_factor / (k_factor + 1.0))
            scatter_scale = torch.sqrt(1.0 / (k_factor + 1.0))
            scatter_real = torch.randn(complex_iq.shape, device=iq_points.device, dtype=iq_points.dtype)
            scatter_imag = torch.randn(complex_iq.shape, device=iq_points.device, dtype=iq_points.dtype)
            scatter = torch.complex(scatter_real, scatter_imag) / torch.sqrt(torch.tensor(2.0, device=iq_points.device, dtype=iq_points.dtype))
            h = torch.complex(los_scale, torch.zeros_like(los_scale)) + scatter_scale * scatter
            faded = h * complex_iq
        else:
            raise ValueError(f"Unsupported channel_type '{channel_type}'. Use awgn, rayleigh, or rician.")

        noise_real = torch.randn(complex_iq.shape, device=iq_points.device, dtype=iq_points.dtype) * noise_std_t
        noise_imag = torch.randn(complex_iq.shape, device=iq_points.device, dtype=iq_points.dtype) * noise_std_t
        noisy = faded + torch.complex(noise_real, noise_imag)
        return self._from_complex(noisy)

    def forward(self, x, noise_std, channel_type="awgn"):
        iq_points = self.encoder(x)
        # Power normalization
        iq_points = iq_points / torch.sqrt(torch.mean(iq_points**2))

        labels = torch.argmax(x, dim=1)
        power_scale = 1.0

        # First pass used to estimate BER for CADUCEUS-WAVE adaptive controller.
        noisy_iq_probe = self._apply_channel(iq_points, noise_std, channel_type)
        decoded_probe = self.decoder(noisy_iq_probe)
        predictions_probe = torch.argmax(decoded_probe, dim=1)
        ber_estimate = (predictions_probe != labels).float().mean().item()

        if ber_estimate > 0.1:
            power_scale = 1.2
        elif ber_estimate < 0.01:
            power_scale = 0.9

        adapted_iq = iq_points * power_scale
        noisy_iq = self._apply_channel(adapted_iq, noise_std, channel_type)
        # Receiver-side power normalization keeps decoder operating at expected scale.
        rx_iq = noisy_iq / power_scale
        decoded_bits = self.decoder(rx_iq)
        return decoded_bits, adapted_iq