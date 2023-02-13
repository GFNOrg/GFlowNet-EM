# Adapted from 
# https://github.com/bshall/VectorQuantizedVAE

import argparse
from pathlib import Path

import numpy as np

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torchvision.transforms.functional as tF

from model import VQVAE

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('image', cmap='gray')

def save_checkpoint(model, optimizer, step, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step}
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))

def shift(x):
    return x - 0.5

def binarize(x):
    return (x > 0.5).float()

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = tF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

device = torch.device("cuda:0")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume.")
    parser.add_argument("--channels", type=int, default=128, help="Number of channels in conv layers.")
    parser.add_argument("--latent-dim", type=int, default=1, help="Dimension of categorical latents.")
    parser.add_argument("--num-embeddings", type=int, default=10, help="Number of codebook embeddings size.")
    parser.add_argument("--embedding-dim", type=int, default=1, help="Dimension of codebook embeddings.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    args = parser.parse_args()

    # Load model
    model = VQVAE(args.channels, args.latent_dim, args.num_embeddings, args.embedding_dim)
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(binarize),
    ])

    test_dataset = datasets.MNIST(root='../data/', train=False, transform=transform, download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, drop_last=True)

    # VQVAE test-set log-likelihood
    model.eval()
    logpx = 0
    logpz = -4*4*np.log(args.num_embeddings)

    for batch in tqdm(test_dataloader):
        img, y = batch
        x = img.float().to(device)

        with torch.no_grad():
            rec, vq_loss, perplexity = model(img.float().to(device))

        logpx_z = -F.binary_cross_entropy(rec, x, reduction='none').sum((1,2,3))
        logpx += (logpx_z + logpz).sum() / test_dataset.__len__()

    model.train()
    print('VQ-VAE test-set NLL:', -logpx, 'nats')
    
