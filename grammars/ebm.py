import torch
import argparse
import numpy as np
from tree_span_utils import *
from pcfg_base import SimpleTree
from tree_agg import aggregated_embedding, TreeModel
import data
import copy

import os


parser = argparse.ArgumentParser(description="Training EBM on PTB parse trees")
parser.add_argument("--seed", type=int, default=1111, help="random seed")
# Model
parser.add_argument("--d_model", type=int, default=16, help="model hidden dimension")
parser.add_argument(
    "--agg_type", type=str, default="simplemlp", help="Aggregation Type"
)
# Optim
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="beta 1 for adam")
parser.add_argument("--adam_beta2", type=float, default=0.99, help="beta 2 for adam")
parser.add_argument(
    "--ebm_l2_reg", type=float, default=0.0, help="L2 regularization on EBM"
)
# training
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--batch_size", type=int, default=32, help="training batch size")
parser.add_argument(
    "--valid_batch_size", type=int, default=32, help="validation batch size"
)
parser.add_argument(
    "--rotations_per_transition",
    type=int,
    default=1,
    help="number of random tree rotations in MCMC proposal step",
)
parser.add_argument(
    "--buffer_reset_ratio", type=float, default=0.5, help="reset frequency for PCD"
)

parser.add_argument("--batch_group_size", type=int, default=99999, help="")
parser.add_argument(
    "--seqlen", type=int, default=20, help="sequence length to train on"
)
parser.add_argument("--epochs", type=int, default=100, help="max epochs")

parser.add_argument("--log_every", type=int, default=32)
parser.add_argument(
    "--data", type=str, default="./data/ptb", help="location of the data corpus"
)
parser.add_argument(
    "--save_dir", type=str, default=None, help="path to save the final model"
)
parser.add_argument("--log_dir", type=str, default="./ckpts", help="path to save logs")
args = parser.parse_args()


torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.save_dir is not None:
    os.makedirs(args.save_dir, exist_ok=True)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

corpus = data.SeqCorpus(
    args.data,
    seqlen=args.seqlen,
    train_batch_size=args.batch_size,
    valid_batch_size=args.valid_batch_size,
    batch_group_size=args.batch_group_size,
    add_master_token=False,
    load_spans=True,
)

train_dataloader = corpus.train
val_dataloader = corpus.valid
test_dataloader = corpus.test

train_spans_dataloader = corpus.train_spans
val_spans_dataloader = corpus.valid_spans
test_spans_dataloader = corpus.test_spans

print("training samples:", len(train_spans_dataloader))
print("validation samples:", len(val_spans_dataloader))

ntokens_src = len(corpus.dict)

aggregator = aggregated_embedding(1, args.d_model, n_nts=0, agg_type=args.agg_type)
aggregator = aggregator.to(device)
eb_model = TreeModel(aggregator=aggregator, d_model=args.d_model, device=device)
eb_model = eb_model.to(device)


opt = torch.optim.Adam(
    eb_model.parameters(), args.lr, betas=(args.adam_beta1, args.adam_beta2)
)

total_params = 0
for g in opt.param_groups:
    for p in g["params"]:
        total_params += p.numel()

print("Total trainable parameters: {}".format(total_params))


def pcd_step(
    pos_samples, buffer, buffer_energies, rotations_per_transition, buffer_reset_ratio
):
    with torch.no_grad():
        mutated_buffer = [copy.deepcopy(t) for t in buffer]

        for t in mutated_buffer:
            for _ in range(rotations_per_transition):
                t.random_rotate()

        mutated_energies = eb_model(mutated_buffer)

        accept_ratio = (buffer_energies - mutated_energies).exp().clip(max=1)
        accept = accept_ratio.bernoulli()

        buffer = [
            (new if a else old) for (old, new, a) in zip(buffer, mutated_buffer, accept)
        ]
    pos_energies = eb_model(pos_samples)
    buffer_energies = eb_model(buffer)

    pos_mean_energy = pos_energies.mean()
    neg_mean_energy = buffer_energies.mean()

    loss = (pos_mean_energy - neg_mean_energy) + sum(
        (x**2).sum() for x in eb_model.parameters()
    ) * args.ebm_l2_reg
    loss.backward()

    for i in range(len(buffer)):
        if np.random.ranf() < buffer_reset_ratio:
            j = np.random.randint(len(pos_samples))
            buffer[i] = pos_samples[j]
            buffer_energies[i] = pos_energies[j]

    return buffer, buffer_energies, loss.item(), pos_energies


def span_to_tree(batch_spans):
    batch_imgtree = [spans_to_imgtree(s) for s in batch_spans]
    batch_binimgtree = [binarize_imgtree(it) for it in batch_imgtree]
    batch_trees = [imgtree_to_tree(bit) for bit in batch_binimgtree]
    return batch_trees


def train_epoch(opt, dataloader, epoch):
    total_train_loss = 0
    cum_train_loss = 0
    pos_energy = 0
    num_samples = 0
    buffer = span_to_tree(dataloader[0])
    buffer_energies = eb_model(buffer)
    for i, spans in enumerate(dataloader):
        opt.zero_grad()
        batch_trees = span_to_tree(spans)
        buffer, buffer_energies, loss, pos_energies = pcd_step(
            batch_trees,
            buffer,
            buffer_energies,
            args.rotations_per_transition,
            args.buffer_reset_ratio,
        )
        pos_energy += pos_energies.sum()
        num_samples += len(spans)
        opt.step()
        total_train_loss += loss
        cum_train_loss += loss
        if i % args.log_every == 0:
            print(
                f"Batch {i} / {len(dataloader)}, Loss:", cum_train_loss / args.log_every
            )
            cum_train_loss = 0
    return total_train_loss / len(dataloader), pos_energy / num_samples


@torch.no_grad()
def sample_ebm(buffer, num_mcmc_steps, rotations_per_transition):
    buffer_energies = eb_model(buffer)
    for _ in range(num_mcmc_steps):
        mutated_buffer = [copy.deepcopy(t) for t in buffer]

        for t in mutated_buffer:
            for _ in range(rotations_per_transition):
                t.random_rotate()

        mutated_energies = eb_model(mutated_buffer)

        accept_ratio = (buffer_energies - mutated_energies).exp().clip(max=1)
        accept = accept_ratio.bernoulli().bool()

        buffer = [
            (new if a else old) for (old, new, a) in zip(buffer, mutated_buffer, accept)
        ]
        buffer_energies = torch.where(accept, mutated_energies, buffer_energies)

    return buffer


best_val_loss = np.inf
for epoch in range(args.epochs):
    train_spans_dataloader = np.random.permutation(train_spans_dataloader)
    train_loss, train_pos_energy = train_epoch(opt, train_spans_dataloader, epoch)

    with torch.no_grad():
        val_pos_energy = 0.0
        rand_energy = 0.0
        val_num_samples = 0
        for i, spans in enumerate(val_spans_dataloader):
            batch_trees = span_to_tree(spans)
            val_pos_energy += eb_model(batch_trees).sum()
            buffer = [
                imgtree_to_tree(uniform_random_imgtree(len(tree_to_spans(t)) + 1))
                for t in batch_trees
            ]
            rand_energy += eb_model(buffer).sum()
            val_num_samples += len(spans)
        val_pos_energy /= val_num_samples
        rand_energy /= val_num_samples
        print(
            f"Training positive energy = {train_pos_energy} | Validation positive energy = {val_pos_energy} | Random energy = {rand_energy}"
        )

    buffer = [
        imgtree_to_tree(uniform_random_imgtree(np.random.randint(3, args.seqlen + 1)))
        for _ in range(args.batch_size)
    ]
    buffer = sample_ebm(buffer, 1000, 1)
    for tree in buffer:
        print(tree)

    if args.save_dir is not None:
        torch.save(
            eb_model.state_dict(), os.path.join(args.save_dir, "ebm_checkpoint_last.pt")
        )
