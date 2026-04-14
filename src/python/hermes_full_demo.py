import torch
import torch.nn as nn
from neural_transceiver import CaduceusAE

def run_hermes_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaduceusAE().to(device)
    
    # Φόρτωση του εκπαιδευμένου "εγκεφάλου"
    try:
        model.load_state_dict(torch.load("caduceus_model.pth"))
        model.eval()
        print("--- HERMES System Online ---")
        print(f"Using Trained CADUCEUS-WAVE Engine on {device}")
    except:
        print("Error: Trained model 'caduceus_model.pth' not found!")
        return

    # Προσομοίωση αποστολής δεδομένων
    # Θα στείλουμε 10.000 σύμβολα (40.000 bits)
    num_samples = 10000
    noise_level = 0.2 # Ένας ρεαλιστικός θόρυβος
    
    # Παραγωγή τυχαίων δεδομένων (0-15)
    tx_data = torch.randint(0, 16, (num_samples,), device=device)
    tx_onehot = torch.nn.functional.one_hot(tx_data, num_classes=16).float()

    with torch.no_grad():
        # Το AI κωδικοποιεί, προσθέτει θόρυβο και αποκωδικοποιεί
        decoded_output, iq_points = model(tx_onehot, noise_level)
        rx_data = torch.argmax(decoded_output, dim=1)
        
        # Υπολογισμός Accuracy
        correct = (rx_data == tx_data).sum().item()
        accuracy = (correct / num_samples) * 100

    print(f"\nTransmission Results:")
    print(f"Total Symbols Sent: {num_samples}")
    print(f"Noise Level (Std Dev): {noise_level}")
    print(f"Successful Decodes: {correct}")
    print(f"Success Rate: {accuracy:.2f}%")
    
    if accuracy > 98:
        print("\n[STATUS] LINK STABLE: High-Fidelity Communication Established.")
    else:
        print("\n[STATUS] LINK UNSTABLE: Consider re-training with lower noise.")

if __name__ == "__main__":
    run_hermes_demo()