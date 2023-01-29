import argparse
import os

import cv2
import matplotlib.pyplot as plt
from tqdm import trange

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
from torchvision import transforms as T

from constants import *
from network import UpresNet


class ImageDataset(Dataset):
    def __init__(self, dir, augment=True):
        self.dir = dir
        self.augment = augment

        self.files = []
        for f in os.listdir(dir):
            if f.endswith((".png", ".jpg")):
                self.files.append(f)

        ratio = OUT_SIZE[1] / OUT_SIZE[0]
        self.augmentation = T.Compose([
            T.RandomResizedCrop(OUT_SIZE, scale=(0.3, 1), ratio=(ratio*0.9, ratio*1.1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(0.1, 0.1, 0.1, 0.1),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = cv2.imread(os.path.join(self.dir, f))
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error:
            print(f)
            raise
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img / 255.0

        if self.augment:
            img = self.augmentation(img)

        return img


def show_samples(dataset):
    """
    Use matplotlib to show some samples from the dataset.
    """
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(dataset[i].permute(1, 2, 0))
        plt.axis("off")
    plt.show()


def train(model, train_data, test_data, epochs=10, bs=8):
    loader_args = {
        "batch_size": bs,
        "shuffle": True,
        "num_workers": 7,
    }
    train_loader = DataLoader(train_data, **loader_args)
    test_loader = DataLoader(test_data, **loader_args)

    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for epoch in (pbar := trange(epochs)):
        pbar.set_description("Training")
        model.train()
        for i, batch in enumerate(train_loader):
            truth = batch.to(device)
            lowres = F.interpolate(truth, size=IN_SIZE, mode="bicubic", align_corners=False)
            pred = model(lowres)

            loss = criteria(truth, pred)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f"Epoch {epoch+1}/{epochs} | Batch {i}/{len(train_loader)} | Loss {loss.item():.4f}")

        pbar.set_description("Testing")
        model.eval()
        with torch.no_grad():
            running_total = 0
            for i, batch in enumerate(test_loader):
                truth = batch.to(device)
                lowres = F.interpolate(truth, size=IN_SIZE, mode="bicubic", align_corners=False)
                pred = model(lowres)

                loss = criteria(truth, pred)
                running_total += loss.item()
            losses.append(running_total / len(test_loader))

    return losses


def save(path, model, losses):
    if len(os.listdir(path)) == 0:
        num = 0
    else:
        num = max(map(int, os.listdir(path))) + 1

    path = os.path.join(path, f"{num:03d}")
    os.makedirs(path)

    torch.save(model.state_dict(), os.path.join(path, "model.pt"))

    plt.plot(losses)
    plt.savefig(os.path.join(path, "losses.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    dataset = ImageDataset(args.data)
    split_fac = 0.9
    train_len = int(len(dataset)*split_fac)
    test_len = len(dataset)-train_len
    train_data, test_data = random_split(dataset, [train_len, test_len])

    model = UpresNet().to(device)
    losses = train(model, train_data, test_data, epochs=args.epochs)

    save(args.results, model, losses)


if __name__ == "__main__":
    main()
