from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from classifier import Net
from dataset import ConnectorsTraining

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 1
NUM_EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True


def train(data_loader, model, optimizer, loss_fn):
    loop = tqdm(data_loader)

    for batch_idx, (images, labels) in enumerate(loop):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        predictions = model(images)
        loss = loss_fn(predictions, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # updates progress bar in command line
        loop.set_postfix(loss=loss.item())


def main():
    model = Net().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    training_loader: DataLoader = DataLoader(ConnectorsTraining(
        "training.csv"), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=1)

    for epoch in range(NUM_EPOCHS):
        train(training_loader, model, optimizer, loss_fn)

    torch.save(model.state_dict(), "classifier1.pth")


if __name__ == "__main__":
    main()
