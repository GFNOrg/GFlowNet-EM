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



def train_vqvae(args):
    model = VQVAE(args.channels, args.latent_dim, args.num_embeddings, args.embedding_dim)
    model.to(args.device)

    model_name = "{}_{}_C_{}_N_{}_M_{}_D_{}".format(args.run, 'VQVAE', args.channels, args.latent_dim,
                                                    args.num_embeddings, args.embedding_dim)

    checkpoint_dir = Path('models/' + model_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open('models/' + model_name + '/run.txt', 'w') as f:
        f.write("Starting\n")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.resume is not None:
        print("Resume checkpoint from: {}:".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["step"]
    else:
        global_step = 0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(binarize),
    ])
    training_dataset = datasets.MNIST(root='../data/', train=True, transform=transform, download=True)
    
    test_dataset = datasets.MNIST(root='../data/', train=False, transform=transform, download=True)

    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, drop_last=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, drop_last=True)
    

    num_epochs = args.num_epochs #args.num_training_steps // len(training_dataloader) + 1
    start_epoch = 0 #global_step // len(training_dataloader) + 1

    N = 1 * 28 * 28
    KL = args.latent_dim * 4 * 4 * np.log(args.num_embeddings)

    for epoch in range(start_epoch, num_epochs + 1):
        if epoch == args.reduce_lr_on:
            # Reduce learning rate
            for p in range(len(optimizer.param_groups)):
                optimizer.param_groups[p]['lr'] = 5e-5
                print('Reduced learning rate')
            
        model.train()
        average_logp = average_vq_loss = average_elbo = average_bpd = average_perplexity = 0
        for i, (images, _) in enumerate(tqdm(training_dataloader), 1):
            images = images.to(args.device)

            rec, vq_loss, perplexity = model(images)
            logp = -F.binary_cross_entropy(rec, images, reduction='none').sum((1, 2, 3)).mean()
            loss = - logp / N + vq_loss
            elbo = (KL - logp) / N
            bpd = (KL - logp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 25000 == 0:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir)
                
            average_logp += (logp.item() - average_logp) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_elbo += (elbo.item() - average_elbo) / i
            average_bpd += (bpd.item() - average_bpd) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

        model.eval()
        average_logp = average_vq_loss = average_elbo = average_bpd = average_perplexity = 0
        for i, (images, _) in enumerate(test_dataloader, 1):
            images = images.to(args.device)

            with torch.no_grad():
                rec, vq_loss, perplexity = model(images)

            logp = -F.binary_cross_entropy(rec, images, reduction='none').sum((1, 2, 3)).mean()
            elbo = (KL - logp) / N
            bpd = KL - logp

            average_logp += (logp.item() - average_logp) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_elbo += (elbo.item() - average_elbo) / i
            average_bpd += (bpd.item() - average_bpd) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

        samples = rec
        grid = utils.make_grid(samples.float())
        if global_step % 25000 == 0:
            show(grid)
            
        
        with open('models/' + model_name + '/run.txt', 'a') as f:
            f.write("epoch:{}, logp:{:.3E}, vq loss:{:.3E}, elbo:{:.3f}, nll:{:.3f}, perplexity:{:.3f}\n"
                    .format(epoch, average_logp, average_vq_loss, average_elbo, average_bpd, average_perplexity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume.")
    parser.add_argument("--channels", type=int, default=128, help="Number of channels in conv layers.")
    parser.add_argument("--latent-dim", type=int, default=1, help="Dimension of categorical latents.")
    parser.add_argument("--num-embeddings", type=int, default=10, help="Number of codebook embeddings size.")
    parser.add_argument("--embedding-dim", type=int, default=1, help="Dimension of codebook embeddings.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs to train for.")
    parser.add_argument("--reduce_lr_on", type=int, default=250000, help="Epoch on which to reduce learning rate.")
    parser.add_argument("--device", type=str, default='cpu', help="Device to run on.")
    parser.add_argument("--run", type=str, default='0', help="Run identifier.")
    args = parser.parse_args()

    train_vqvae(args)
