import argparse
import os

import cv2
import matplotlib.pyplot as plt
from tqdm import trange

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
from torchvision import transforms as T

from constants import *
from network import UpresNet, Discriminator


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


def init_weights(m):
    name = m.__class__.__name__
    if name.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif name.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def train(generator, disc, train_data, test_data, epochs=10, bs=8):
    """
    :param disc_interval: Number of generator iterations to each disc iter.
    """
    loader_args = {
        "batch_size": bs,
        "shuffle": True,
        "num_workers": 7,
    }
    train_loader = DataLoader(train_data, **loader_args)
    test_loader = DataLoader(test_data, **loader_args)

    criteria = torch.nn.BCELoss()
    optim_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
    optim_d = torch.optim.Adam(disc.parameters(), lr=1e-3)

    losses = []
    # Proportion of generator wins.
    accuracies = []
    for epoch in (pbar := trange(epochs)):
        pbar.set_description("Training")
        generator.train()
        disc.train()
        optim_g.zero_grad()
        optim_d.zero_grad()
        total_gen_loss = 0
        total_disc_loss = 0

        for i, batch in enumerate(train_loader):
            batch = batch.to(device)

            disc.zero_grad()

            # Train disc on real
            pred = disc(batch).view(-1)
            truth = torch.ones((batch.size(0),), device=device)
            disc_loss = criteria(pred, truth)
            disc_loss.backward()
            curr_disc_loss = disc_loss.item()
            total_disc_loss += disc_loss.item()

            # Train disc on fake
            lowres = F.interpolate(batch, size=IN_SIZE, mode="bicubic", align_corners=False)
            fake = generator(lowres)
            truth = torch.zeros((batch.size(0),), device=device)
            pred = disc(fake.detach()).view(-1)
            disc_loss = criteria(pred, truth)
            disc_loss.backward()
            curr_disc_loss += disc_loss.item()
            total_disc_loss += disc_loss.item()

            # Slow down discriminator training
            #if i % 2 == 0:
            optim_d.step()

            # Train generator
            generator.zero_grad()
            fake = generator(lowres)
            pred = disc(fake).view(-1)
            truth = torch.ones((batch.size(0),), device=device)
            gen_loss = criteria(pred, truth)
            #gen_loss = F.mse_loss(fake, batch)
            gen_loss.backward()
            optim_g.step()
            total_gen_loss += gen_loss.item()

            pbar.set_description(f"Epoch {epoch+1}/{epochs} | Batch {i}/{len(train_loader)} | LossG {gen_loss.item():.4f} | LossD {curr_disc_loss:.4f}")

        total_gen_loss /= len(train_loader)
        total_disc_loss /= len(train_loader)
        losses.append((total_gen_loss, total_disc_loss))

        pbar.set_description("Testing")
        generator.eval()
        disc.eval()
        with torch.no_grad():
            gen_win = 0
            total = 0
            for i, batch in enumerate(test_loader):
                batch = batch.to(device)
                lowres = F.interpolate(batch, size=IN_SIZE, mode="bicubic", align_corners=False)
                fake = generator(lowres)
                pred = disc(fake)

                gen_win += (pred > 0.5).sum().item()
                total += pred.numel()

            accuracies.append(gen_win / total)

    return losses, accuracies


def save(path, generator, disc, losses, accuracies):
    if len(os.listdir(path)) == 0:
        num = 0
    else:
        num = 0
        for f in os.listdir(path):
            if f.isdigit():
                num = max(num, int(f) + 1)

    # Make symlink to latest
    if os.path.exists(os.path.join(path, "latest")):
        os.remove(os.path.join(path, "latest"))
    os.symlink(f"{num:03d}", os.path.join(path, "latest"))

    path = os.path.join(path, f"{num:03d}")
    os.makedirs(path)

    torch.save(generator.state_dict(), os.path.join(path, "gen.pt"))
    torch.save(disc.state_dict(), os.path.join(path, "disc.pt"))

    plt.cla()
    plt.plot(losses)
    plt.savefig(os.path.join(path, "losses.png"))
    plt.cla()
    plt.plot(accuracies)
    plt.savefig(os.path.join(path, "accuracies.png"))

    with open(os.path.join(path, "losses.txt"), "w") as f:
        for loss in losses:
            f.write(f"{loss[0]:.4f} {loss[1]:.4f}\n")

    with open(os.path.join(path, "accuracies.txt"), "w") as f:
        for acc in accuracies:
            f.write(f"{acc:.4f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--resume", help="Path to folder containing gen.pt and disc.pt", default="")
    args = parser.parse_args()

    dataset = ImageDataset(args.data)
    split_fac = 0.9
    train_len = int(len(dataset)*split_fac)
    test_len = len(dataset)-train_len
    train_data, test_data = random_split(dataset, [train_len, test_len])

    generator = UpresNet().to(device)
    disc = Discriminator().to(device)
    if os.path.exists(args.resume):
        print("Resuming from", args.resume)
        generator.load_state_dict(torch.load(os.path.join(args.resume, "gen.pt")))
        disc.load_state_dict(torch.load(os.path.join(args.resume, "disc.pt")))
    else:
        print("Starting from scratch")
        generator.apply(init_weights)
        disc.apply(init_weights)

    losses, accuracies = train(generator, disc, train_data, test_data, epochs=args.epochs)

    save(args.results, generator, disc, losses, accuracies)


if __name__ == "__main__":
    main()
