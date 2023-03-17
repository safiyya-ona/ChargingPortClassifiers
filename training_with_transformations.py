from tqdm import tqdm
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

from classifier_transforms import Net
from dataset import ConnectorsTraining

# Hyperparameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = [64, 64]
PIN_MEMORY = True


def change_image_size(image, size=IMAGE_SIZE):
    image = resize(image, size)
    return image


def train(data_loader, model, optimizer, loss_fn):
    loop = tqdm(data_loader)

    for batch_idx, (images, labels) in enumerate(loop):
        # original image
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

    return loss.item()


def main():
    model = Net().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    training_loader: DataLoader = DataLoader(ConnectorsTraining(
        "training_transformed.csv", img_dir="training_transformed", transform=change_image_size), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=1)

    # Create a logger object
    logging.basicConfig(
        filename='transformations_loss.log', level=logging.INFO)

    for epoch in range(NUM_EPOCHS):
        loss = train(training_loader, model, optimizer, loss_fn)
        logging.info("Epoch: " + str(epoch + 1) + " Loss: " + str(loss))

    torch.save(model.state_dict(), "classifier_transformed.pth")


if __name__ == "__main__":
    main()
