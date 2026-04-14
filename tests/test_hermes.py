import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_SRC = os.path.join(PROJECT_ROOT, "src", "python")
if PYTHON_SRC not in sys.path:
    sys.path.insert(0, PYTHON_SRC)

from neural_transceiver import CaduceusAE


def _device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for HERMES GPU-first tests.")
    return torch.device("cuda")


def _qpsk_ber_awgn(snr_db, symbols=10_000, device=None):
    noise_std = 10 ** (-snr_db / 20.0)
    tx_bits = torch.randint(0, 2, (symbols, 2), device=device, dtype=torch.int64)
    tx_i = tx_bits[:, 0].float() * 2 - 1
    tx_q = tx_bits[:, 1].float() * 2 - 1
    tx = torch.complex(tx_i, tx_q) / math.sqrt(2.0)
    noise = torch.complex(torch.randn(symbols, device=device), torch.randn(symbols, device=device)) * noise_std
    rx = tx + noise
    rx_bits = torch.stack(((rx.real > 0).long(), (rx.imag > 0).long()), dim=1)
    return (rx_bits != tx_bits).float().mean().item()


def _caduceus_ber(model, snr_db, device, symbols=10_000):
    noise_std = 10 ** (-snr_db / 20.0)
    with torch.no_grad():
        labels = torch.randint(0, 16, (symbols,), device=device)
        inputs = F.one_hot(labels, num_classes=16).float()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, _ = model(inputs, noise_std, channel_type="awgn")
        preds = torch.argmax(logits.float(), dim=1)
    return (preds != labels).float().mean().item()


def test_caduceus_forward_awgn():
    device = _device()
    model = CaduceusAE().to(device).eval()
    x = F.one_hot(torch.randint(0, 16, (1024,), device=device), num_classes=16).float()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits, iq = model(x, 0.2, channel_type="awgn")
    assert logits.shape == (1024, 16)
    assert iq.shape == (1024, 2)


def test_caduceus_forward_rayleigh():
    device = _device()
    model = CaduceusAE().to(device).eval()
    x = F.one_hot(torch.randint(0, 16, (1024,), device=device), num_classes=16).float()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits, iq = model(x, 0.2, channel_type="rayleigh")
    assert logits.shape == (1024, 16)
    assert iq.shape == (1024, 2)


def test_ber_decreases_with_snr():
    device = _device()
    model = CaduceusAE().to(device).eval()
    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "caduceus_model.pth"), map_location=device, weights_only=True))
    torch.manual_seed(42)
    ber_0 = _caduceus_ber(model, 0, device, symbols=8000)
    ber_20 = _caduceus_ber(model, 20, device, symbols=8000)
    assert ber_20 < ber_0


def test_caduceus_better_than_qpsk_at_high_snr():
    device = _device()
    model = CaduceusAE().to(device).eval()
    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "caduceus_model.pth"), map_location=device, weights_only=True))
    torch.manual_seed(3)
    cad_ber_20 = _caduceus_ber(model, 20, device, symbols=12000)
    qpsk_ber_20 = _qpsk_ber_awgn(20, symbols=12000, device=device)
    assert cad_ber_20 <= qpsk_ber_20


def test_shannon_capacity():
    snr_db_values = torch.arange(1, 21, dtype=torch.float32)
    capacities = torch.log2(1 + (10 ** (snr_db_values / 10.0)))
    assert torch.all(capacities > 0).item()


def test_adaptive_controller():
    device = _device()
    model = CaduceusAE().to(device).eval()

    # Freeze channel for deterministic ratio checks.
    model._apply_channel = lambda iq, noise_std, channel_type: iq

    class ZeroDecoder(torch.nn.Module):
        def forward(self, iq):
            logits = torch.zeros(iq.shape[0], 16, device=iq.device)
            logits[:, 0] = 1.0
            return logits

    model.decoder = ZeroDecoder().to(device)
    labels_high_ber = torch.randint(1, 16, (512,), device=device)
    x_high_ber = F.one_hot(labels_high_ber, num_classes=16).float()
    base = model.encoder(x_high_ber)
    base = base / torch.sqrt(torch.mean(base**2))
    _, iq_high = model(x_high_ber, 0.0, channel_type="awgn")
    assert torch.isclose((iq_high.abs().mean() / base.abs().mean()), torch.tensor(1.2, device=device), atol=0.05)

    labels_low_ber = torch.zeros(512, dtype=torch.int64, device=device)
    x_low_ber = F.one_hot(labels_low_ber, num_classes=16).float()
    base_low = model.encoder(x_low_ber)
    base_low = base_low / torch.sqrt(torch.mean(base_low**2))
    _, iq_low = model(x_low_ber, 0.0, channel_type="awgn")
    assert torch.isclose((iq_low.abs().mean() / base_low.abs().mean()), torch.tensor(0.9, device=device), atol=0.05)


def test_model_loads_from_checkpoint():
    device = _device()
    model = CaduceusAE().to(device)
    state_dict = torch.load(os.path.join(PROJECT_ROOT, "caduceus_model.pth"), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    assert True
