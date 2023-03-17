import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms.functional import resize


from classifier import Net as Net
from classifier_transforms import Net as NetTransformed
from dataset import ConnectorsTesting

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def change_image_size(image, size=[64, 64]):
    image = resize(image, size)
    return image


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
    print(total, correct, 100 * correct / total, "correct")


if __name__ == "__main__":
    print("Testing Classifier without transformations")
    network = Net()
    network.load_state_dict(torch.load("classifier1.pth"))
    network.to(DEVICE)
    testing_loader = DataLoader(ConnectorsTesting("testing.csv"))
    validate(testing_loader, network)

    print("Testing Classifier with transformations")
    network = NetTransformed()
    network.load_state_dict(torch.load("classifier_transformed.pth"))
    network.to(DEVICE)
    testing_loader = DataLoader(ConnectorsTesting(
        "testing_transformed.csv", img_dir="testing_transformed", transform=change_image_size), batch_size=1, shuffle=False, num_workers=1)
    validate(testing_loader, network)
