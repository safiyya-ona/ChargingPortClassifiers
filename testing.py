import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifier import Net
from dataset import ConnectorsTesting

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(testing_loader, model, device=DEVICE):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(testing_loader, 0)):
            image, label = data
            label = label.to(device)
            image = image.to(device)
            output = model(image)

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print(total, correct)


if __name__ == "__main__":
    network = Net()
    network.load_state_dict(torch.load("classifier1.pth"))
    network.to(DEVICE)
    testing_loader = DataLoader(ConnectorsTesting("testing.csv"))
    validate(testing_loader, network)
