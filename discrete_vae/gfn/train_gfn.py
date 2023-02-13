import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from networks import GFlowNet, LatentDictionary, Decoder

import tqdm

def binarize_static(x):
    return (x > 0.5).float()

def sample(gfn, img, p=1, rand_prob=0.05):
    '''Samples from the GFlowNet policy. Implemented for the autoregressive policy.
    Args:
        gfn (torch.nn.Module): GFlowNet network
        img (torch.Tensor): Image to condition GFlowNet sampling on
        p (float): If p > 0, raises the predicted action probabilites to the power of p. If p < 0 samples greedily from the GFlowNet
        rand_prob (float): Probability to perform random action
    Returns:
        state (torch.Tensor): Returns the state constructed by sampling actions from the GFlowNet
        logprobs (torch.Tensor): Sum of log-probabilities of the actions taken
    '''
    device = img.device
    batch_size, dict_size, lh, lw = img.shape[0], gfn.dictionary_size, gfn.lh, gfn.lw 
    steps = lh*lw

    # Initialize state with zeros
    state = torch.zeros((batch_size, dict_size*lh*lw)).float().to(device)

    logprobs = torch.zeros((batch_size,1)).float().to(device)
    for i in range(steps):
        # Predict logits over next states
        pred_logits = gfn(img, state.clone().float())

        # Mask out already completed states (AR policy)
        mask = torch.zeros((batch_size, lh*lw), device=device)
        mask[:,i:i+1] = 1
        mask = mask.view(batch_size,lh,lw).tile(dict_size,1,1,1).permute([1,0,2,3]).reshape(batch_size, dict_size*lh*lw)
                
        # Compute probabilites over next actions
        # This is equivalent to computing softmax over 'unmasked' positions
        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_probs = (pred_probs + 1e-9) / torch.sum((pred_probs + 1e-9) * mask, dim=-1, keepdims=True)
            
        # Perform actions
        if p > 0:
            pred_step = torch.multinomial(pred_probs**p * mask, 1)
        else:
            pred_step = torch.argmax(pred_probs * mask, 1, keepdims=True)
        # Random steps
        rand_step = torch.multinomial(torch.ones_like(pred_probs) * mask, 1)
        rand_update = (torch.rand((batch_size,1), device=device) < rand_prob).long()
        pred_step = (1-rand_update)*pred_step + rand_update*rand_step
        
        # Add step to state
        state.scatter_(1, pred_step, 1)
        # Add log prob of selected step
        logprobs = logprobs + (pred_probs.gather(1, pred_step)).log()
                
    assert torch.all(state.view(batch_size, dict_size, lh, lw).sum(1) == 1), 'State incomplete'

    return state, logprobs


def sleep_step(gfn, decoder, latent_dict, batch_size=128):
    '''Performs a sleep-phase step by sampling a random (state, image) pair and computing log-probabilities
    Args:
        gfn (torch.nn.Module): GFlowNet network
        decoder (torch.nn.Module): Decoder network
        latent_dict (torch.nn.Module): Discrete latent->Embeddings dictionary
        batch_size (int): Number of sleep-phase samples to draw
    Returns:
        logprobs (torch.Tensor): Sum of log-probabilities of sampled actions
        img_in (torch.Tensor): Sampled image
    '''
    device = latent_dict.dictionary.weight.device
    dict_size, embedding_dim, lh, lw = gfn.dictionary_size, gfn.embedding_dim, gfn.lh, gfn.lw
    steps = lh*lw
    
    # Sample a random latent
    # Trajectory is fixed since we use an AR policy
    rand_t = torch.stack([torch.arange(0, lh*lw, device=device) for _ in range(batch_size)])
    random_latent = torch.randint(0, dict_size, (batch_size,4*4), device=device)
    random_latent_state = F.one_hot(random_latent, dict_size).permute([0,2,1]).reshape(batch_size, -1)
    
    # Sample an image from the decoder
    state_in = torch.sum(latent_dict.dictionary.weight.view(1,dict_size,embedding_dim,1,1) * random_latent_state.reshape(batch_size, dict_size, 1, lh, lw), dim=1)
    img_in = torch.bernoulli(decoder(state_in))
    
    state = torch.zeros((batch_size, dict_size*lh*lw)).float().to(device)
    logprobs = torch.zeros((batch_size,1)).float().to(device)
    for i in range(steps):
        # Predict logits over next states
        pred_logits = gfn(img_in, state.clone().float())

        # Mask out already completed states - AR policy
        mask = torch.zeros((batch_size, lh*lw), device=device)
        mask[:,i:i+1] = 1
        mask = mask.view(batch_size,lh,lw).tile(dict_size,1,1,1).permute([1,0,2,3]).reshape(batch_size, dict_size*lh*lw)
        
        # Compute probabilites over next actions
        # This is equivalent to computing softmax over 'unmasked' positions
        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_probs = (pred_probs + 1e-9) / torch.sum((pred_probs+1e-9) * mask, dim=-1, keepdims=True)

        # Perform action - Predetermined by the sampled latent
        pred_step = random_latent[:,[i]] * lh*lw + rand_t[:,[i]]
        
        # Add step to state
        state.scatter_(1, pred_step, 1)

        # Add log prob of sampled step
        logprobs = logprobs + (pred_probs.gather(1, pred_step)).log()
        
    assert torch.all(state.view(batch_size, dict_size, lh, lw).sum(1) == 1), 'State incomplete'
    
    return logprobs, img_in



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the discrete GFlowNet encoder.')
    
    parser.add_argument('--dictionary_size', type=int, default=8, help='Number of dictionary entries (categorical latent variables)')
    parser.add_argument('--channels', type=int, default=128, help='Hidden channels for encoder/decoder')
    
    parser.add_argument('--greedy_prob', type=float, default=1, help='Probability of drawing a greedy sample from the GFlowNet encoder')
    parser.add_argument('--rand_prob', type=float, default=0, help='Probability of performing a random step')
    parser.add_argument('--sleep_prob', type=float, default=0.3, help='Probability of performing a sleep-phase exploration step during an E-step')
    parser.add_argument('--alternate_every', type=int, default=400, help='Number of steps before alternating between E and M steps')
    
    parser.add_argument('--gfn_lr', type=float, default=2e-4, help='GFlowNet learning rate')
    parser.add_argument('--gfn_reduced_lr', type=float, default=5e-5, help='GFlowNet reduced learning rate')
    parser.add_argument('--gfn_logZ_lr', type=float, default=1e-1, help='GFlowNet logZ learning rate')
    parser.add_argument('--dec_lr', type=float, default=2e-4, help='Decoder learning rate')
    
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train for')
    parser.add_argument('--reduce_lr_on', type=int, default=150, help='Epoch to reduce lr on')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes')
    
    parser.add_argument('--save_every', type=int, default=50, help='Number of epochs between saving models')
    parser.add_argument('--importance_samples', type=int, default=5000, help='Number of importance-weighted samples to estimate NLL')
    
    parser.add_argument('--data_dir', type=str, default='../data/', help='Directory to store dataset in')
    parser.add_argument('--model_dir', type=str, default='models/', help='Directory to save model at')
    parser.add_argument('--run', type=str, default='0', help='Run number')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    model_name = f'Run={args.run}_C={args.channels}_K={args.dictionary_size}_GreedyProb={args.greedy_prob}_Epochs={args.epochs}'
    if not os.path.isdir(os.path.join(args.model_dir, model_name)):
        os.makedirs(os.path.join(args.model_dir, model_name))
        
    # Load data
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0, 1),
        torchvision.transforms.Lambda(binarize_static), # Static MNIST binarization
    ])
    mnist_train = torchvision.datasets.MNIST(root=args.data_dir, train=True, transform=transforms, download=True)
    train_dataloader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    mnist_test = torchvision.datasets.MNIST(root=args.data_dir, train=False, transform=transforms, download=True)
    test_dataloader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Model definitions
    # GFlowNet encoder
    gfn = GFlowNet(channels=args.channels, dictionary_size=args.dictionary_size).to(args.device)
    gfn.train()
    gfn_opt = torch.optim.Adam([{'params': gfn.img_enc.parameters()},
                                {'params': gfn.state_enc.parameters()},
                                {'params': gfn.pred.parameters()},
                                {'params': gfn.logZ, 'lr': args.gfn_logZ_lr}], # Higher lr for logZ
                               lr=args.gfn_lr)

    # Decoder
    latent_dict = LatentDictionary(dictionary_size=args.dictionary_size).to(args.device)
    latent_dict.train()
    decoder = Decoder(channels=args.channels).to(args.device)
    decoder.train()
    dec_opt = torch.optim.Adam(list(decoder.parameters()) + list(latent_dict.parameters()), lr=args.dec_lr)

    # Train models
    print('[*] Training model')
    steps = gfn.lh*gfn.lw
    reward = torch.zeros((args.batch_size,1))
    gfn_losses = [0]
    dec_losses = [0]
    s = 0
    for e in range(args.epochs):
        print(f'Epoch [{e+1}/{args.epochs}]')
        # Reduce learning rate
        if (e+1) == args.reduce_lr_on:
            for i in range(3):
                gfn_opt.param_groups[i]['lr'] = args.gfn_reduced_lr

        batch_bar = tqdm.tqdm(train_dataloader)
        for batch in batch_bar:
            img, _ = batch
            img = img.float().to(args.device)
            batch_size = img.shape[0]
            
            # Alternate between E and M steps
            s += 1
            if s % (2*args.alternate_every) < (args.alternate_every):
                stage = 'dec'
            else:
                stage = 'gfn'

            # E-step: Train GFlowNet encoder using decoder image log-likelihoods
            if stage == 'gfn':
                # Sample latent from GFN
                state, logprobs = sample(gfn, img, p=1, rand_prob=args.rand_prob)
                # Decode latents into image
                state_in = latent_dict(state.view(batch_size, gfn.dictionary_size, 1, gfn.lh, gfn.lw))
                pred_img = decoder(state_in)

                # Reward -- image likelihood
                reward = -F.binary_cross_entropy(pred_img, img, reduction='none').sum((1,2,3)) / steps
                # TB loss
                fw_loss = (gfn.logZ.view(1,1) + logprobs.view(batch_size,1) / steps - reward.view(batch_size,1))**2
                fw_loss = fw_loss.mean()
                # Backprop
                gfn_opt.zero_grad()
                fw_loss.backward()
                gfn_opt.step()
                
                # Log losses
                gfn_losses.append(fw_loss.item())

                # Sleep phase exploration
                if np.random.rand() < args.sleep_prob:
                    # Maximize log-probabilities of randomly sampled trajectory
                    logprobs, _ = sleep_step(gfn, decoder, latent_dict, batch_size=args.batch_size)
                    
                    sleep_loss = -10 * logprobs.mean() / steps
                    gfn_opt.zero_grad()
                    sleep_loss.backward()
                    gfn_opt.step()

            # M-step: Train decoder using sampled latents
            elif stage == 'dec':
                # Greedy encoder - sample from GFlowNet greedily
                if np.random.rand() < args.greedy_prob:
                    state, logprobs = sample(gfn, img, p=-1, rand_prob=0)
                else:
                    state, logprobs = sample(gfn, img, p=1, rand_prob=0)

                # Reconstruct image from sampled latent
                state_in = latent_dict(state.view(batch_size, gfn.dictionary_size, 1, gfn.lh, gfn.lw))
                pred_img = decoder(state_in)
                # Decoder loss
                dec_loss = F.binary_cross_entropy(pred_img, img)
                dec_opt.zero_grad()
                dec_loss.backward()
                dec_opt.step()

                # Log losses
                dec_losses.append(dec_loss.item())

            # Update progress bar
            if (s+1) % 20 == 0 or s == 0:
                batch_bar.set_postfix({'GFN Loss': np.mean(gfn_losses), 'Decoder Loss': np.mean(dec_losses)})
                gfn_losses = gfn_losses[-100:]
                dec_losses = dec_losses[-100:]
                
        # Save models
        if (e+1) % args.save_every == 0:
            torch.save(gfn.state_dict(), os.path.join(args.model_dir, model_name, f'gfn_encoder_{e+1:04d}.pth'))
            torch.save(latent_dict.state_dict(), os.path.join(args.model_dir, model_name, f'latent_dict_{e+1:04d}.pth'))
            torch.save(decoder.state_dict(), os.path.join(args.model_dir, model_name, f'decoder_{e+1:04d}.pth'))

    torch.save(gfn.state_dict(), os.path.join(args.model_dir, model_name, f'gfn_encoder_{e+1:04d}.pth'))
    torch.save(latent_dict.state_dict(), os.path.join(args.model_dir, model_name, f'latent_dict_{e+1:04d}.pth'))
    torch.save(decoder.state_dict(), os.path.join(args.model_dir, model_name, f'decoder_{e+1:04d}.pth'))
    
    
    # Evaluate -- computing the negative log-likelihood
    logpx = 0
    logpz = -4*4*np.log(gfn.dictionary_size)
    M = args.importance_samples # Importance-weighted samples

    gfn.eval()
    latent_dict.eval()
    decoder.eval()
    
    print('[*] Evaluating model')
    for batch in tqdm.tqdm(test_dataloader):
        img, _ = batch

        # Estimate of log p(x) using importance sampling
        for b in range(img.shape[0]):
            x = torch.stack([img[b]] * M, dim=0).float().to(args.device)
            batch_size = x.shape[0]
            
            with torch.no_grad():
                # Sample from GFN
                state, logprobs = sample(gfn, x, p=1, rand_prob=0)
                # Reconstruct image
                state_in = latent_dict(state.view(batch_size, gfn.dictionary_size, 1, gfn.lh, gfn.lw))
                x_rec = decoder(state_in)

                logqz_x = logprobs.view(-1)
                logpx_z = -F.binary_cross_entropy(x_rec, x, reduction='none').sum((1,2,3))
                logpx_sample = logpx_z + logpz - logqz_x

                logpx += (torch.logsumexp(logpx_sample, dim=0) - np.log(M)) / mnist_test.__len__()
                
    gfn.train()
    latent_dict.train()
    decoder.train()

    # Log computed NLL
    print(f'K={gfn.dictionary_size} M={M} GFlowNet NLL:', -logpx.cpu().item(), 'nats')
    with open(os.path.join(args.model_dir, model_name, 'nll.txt'), 'w') as f:
        f.write(f'K={gfn.dictionary_size} M={M} GFlowNet NLL: {-logpx.cpu().item()} nats')
