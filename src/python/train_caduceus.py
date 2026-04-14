import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from neural_transceiver import CaduceusAE

def train_hermes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaduceusAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Διασφάλιση ότι υπάρχει ο φάκελος
    os.makedirs("results/constellations", exist_ok=True)

    epochs = 5000
    batch_size = 1024
    noise_std = 0.3 

    print(f"Training HERMES on {device}...")
    for epoch in range(epochs):
        labels = torch.randint(0, 16, (batch_size,), device=device)
        inputs = torch.nn.functional.one_hot(labels, num_classes=16).float()

        outputs, _ = model(inputs, noise_std)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    # ΣΩΣΙΜΟ ΜΟΝΤΕΛΟΥ
    torch.save(model.state_dict(), "caduceus_model.pth")
    print("Model saved to caduceus_model.pth")

    # Οπτικοποίηση & Αποθήκευση
    model.eval()
    with torch.no_grad():
        all_labels = torch.arange(0, 16, device=device)
        all_inputs = torch.nn.functional.one_hot(all_labels, num_classes=16).float()
        _, iq_learned = model(all_inputs, 0)
        
        plt.figure(figsize=(8,8))
        plt.scatter(iq_learned[:, 0].cpu(), iq_learned[:, 1].cpu(), c='red', s=100)
        for i in range(16):
            plt.text(iq_learned[i, 0].cpu(), iq_learned[i, 1].cpu(), str(i), fontsize=12)
        plt.title("CADUCEUS-WAVE: AI-Learned Constellation")
        plt.grid(True)
        
        # Save before showing
        plt.savefig("results/constellations/learned_constellation.png")
        print("Plot saved to results/constellations/learned_constellation.png")
        plt.show()

if __name__ == "__main__":
    train_hermes()