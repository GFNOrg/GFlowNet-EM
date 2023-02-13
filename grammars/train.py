# coding: utf-8
from builtins import NotImplementedError
import argparse
import time
import warnings
import os
import random
import math
import torch
import torch.optim as optim
import numpy as np
import scipy
from itertools import chain

try:
    from apex import amp
    import apex
except ImportError:
    warnings.warn(
        "Failed to import apex. You can still train in FP32, but you need to install apex for half-precision training."
    )

from hashlib import sha1

import wandb

import data
from gflownet_parser import *
from pcfg_neural import MLPPCFG

global global_step
global_step = 0

parser = argparse.ArgumentParser(
    description="GFlowNet for hierarchical latent structures"
)
# Arch
parser.add_argument(
    "--d_model", type=int, default=256, help="number of hidden units per layer"
)
parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
parser.add_argument("--norm_type", default="preln", choices=["postln", "preln"])
parser.add_argument(
    "--nhead",
    type=int,
    default=4,
    help="the number of heads in the encoder/decoder of the transformer model",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    help="dropout applied to layers (0 = no dropout)",
)
parser.add_argument("--gfn_arch", default="parser", choices=["parser"])
parser.add_argument(
    "--tie_tgt_embedding",
    action="store_true",
    help="whether to tie transformer target embedding",
)
parser.add_argument(
    "--share_grammar_embedding",
    action="store_true",
    help="whether to use the grammar symbol embeddings for GFN",
)
parser.add_argument(
    "--agg_type",
    default="none",
    choices=["none", "simplemlp", "skipmlp"],
    help="Aggregation Type",
)

# Optim
parser.add_argument("--lr", type=float, default=0.01, help="initial learning rate")
parser.add_argument(
    "--lr_encoder", type=float, default=None, help="initial learning rate"
)
parser.add_argument("--lr_flow", type=float, default=None, help="initial learning rate")
parser.add_argument(
    "--lr_forward", type=float, default=None, help="initial learning rate"
)
parser.add_argument(
    "--lr_backward", type=float, default=None, help="initial learning rate"
)
parser.add_argument(
    "--lr_grammar", type=float, default=None, help="initial learning rate"
)
parser.add_argument(
    "--init_mult_encoder", type=float, default=1.0, help="initialization std multiplier"
)
parser.add_argument(
    "--init_mult_flow", type=float, default=1.0, help="initialization std multiplier"
)
parser.add_argument(
    "--init_mult_forward", type=float, default=1.0, help="initialization std multiplier"
)
parser.add_argument(
    "--init_mult_backward",
    type=float,
    default=1.0,
    help="initialization std multiplier",
)
parser.add_argument("--momentum", type=float, default=0, help="momentum")
parser.add_argument("--adam_beta1", type=float, default=0.9)
parser.add_argument("--adam_beta2", type=float, default=0.99)
parser.add_argument("--adam_beta1_grammar", type=float, default=0.75)
parser.add_argument("--adam_beta2_grammar", type=float, default=0.999)
parser.add_argument("--optimizer", default="adam", choices=["sgd", "adam"])
parser.add_argument(
    "--schedule",
    default="constant",
    choices=[
        "constant",
        "linear",
        "cosine",
        "inverse_sqrt",
    ],
)
parser.add_argument(
    "--init_var",
    type=float,
    default=1,
    help="embeddings are initialized with variance emb_var/ninp",
)
parser.add_argument("--clip", type=float, default=3, help="gradient clipping")
parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
parser.add_argument(
    "--grad_acc", type=int, default=1, help="number of gradient accumulation steps"
)
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--valid_batch_size", type=int, default=64, help="batch size")
parser.add_argument(
    "--batch_group_size",
    type=int,
    default=999999,
    help="batch group size for grouped shuffling",
)

# Train
parser.add_argument("--seqlen", type=int, default=20, help="sequence length")
parser.add_argument("--epochs", type=int, default=999, help="upper epoch limit")
parser.add_argument(
    "--scheduler_epochs", type=int, default=-1, help="upper epoch limit"
)
parser.add_argument(
    "--uniform_pos_until",
    type=int,
    default=0,
    help="use uniform position decoders until a certain epoch",
)
parser.add_argument("--updates", type=int, default=999999, help="max GFlowNet updates")
parser.add_argument(
    "--max_grammar_updates", type=int, default=10000, help="max grammar updates"
)
parser.add_argument("--warmup", type=int, default=40, help="step of warmup from 0 lr")
parser.add_argument(
    "--train_grammar", type=int, default=1, help="whether to train the neural grammar"
)
parser.add_argument(
    "--tjb_backward", type=int, default=0, help="go backward from known terminal states"
)
parser.add_argument("--tjb_forward", type=int, default=1, help="go forward from s_0")
parser.add_argument(
    "--go_back_and_forward",
    type=int,
    default=1,
    help="go backward from a known terminal state and then forward",
)
parser.add_argument(
    "--sleep_mle",
    type=int,
    default=1,
    help="update logPF for tree hallucinated from grammar",
)
parser.add_argument("--bnf_gamma", type=float, default=1.0)
parser.add_argument("--smle_gamma", type=float, default=10.0)
parser.add_argument(
    "--backward_until_epochs",
    type=int,
    default=-1,
    help="stop going backward after a certain number of epochs",
)
parser.add_argument("--temperature", type=float, default=1.0, help="GFN policy temp")
parser.add_argument(
    "--temperature_pos",
    type=float,
    default=-1.0,
    help="GFN policy temp for position decoders; it falls back to --temp if negative",
)
parser.add_argument(
    "--temperature_tok",
    type=float,
    default=-1.0,
    help="GFN policy temp for token decoders; it falls back to --temp if negative",
)
parser.add_argument("--uniform_pb", action="store_true")
parser.add_argument("--modular_energy", action="store_true")
parser.add_argument("--flow_estimator", action="store_true")
parser.add_argument(
    "--grammar_pretrain_epochs",
    default=0,
    type=int,
    help="Number of epochs to pretrain grammar with exact sampling",
)
parser.add_argument(
    "--pf_uniform_eps_tok",
    type=float,
    default=0.0,
    help="P_F is uniform with probability eps",
)
parser.add_argument(
    "--pf_uniform_eps_pos",
    type=float,
    default=0.0,
    help="P_F is uniform with probability eps",
)
parser.add_argument("--subtb_lambda", type=float, default=1.0)
parser.add_argument("--symbol_dropout", type=float, default=0.0)
parser.add_argument("--mc_em", action="store_true")

# Reward and Eval
parser.add_argument(
    "--reward_temperature", type=float, default=1, help="reward = reward**pow"
)
parser.add_argument("--reward_scale", type=float, default=1.0, help="reward multiplier")

# IO
parser.add_argument(
    "--data", type=str, default="./data/ptb", help="location of the data corpus"
)
parser.add_argument(
    "--log_interval", type=int, default=64, metavar="N", help="report interval"
)
parser.add_argument(
    "--grammar_log_interval",
    type=int,
    default=100,
    metavar="N",
    help="grammar report interval",
)
parser.add_argument(
    "--save_dir", type=str, default=None, help="path to save the final model"
)
parser.add_argument(
    "--resume_dir", type=str, default=None, help="path to resume training"
)
parser.add_argument("--log_dir", type=str, default="./ckpts", help="path to save logs")
parser.add_argument("--wandb_tag", type=str, default="none", help="wandb tag")
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--curriculum_len", type=int, default=0)
parser.add_argument("--curriculum_step", type=int, default=1)

# Decoding
parser.add_argument(
    "--decode_temperature",
    type=float,
    default=1.0,
    help="temperature used when decoding by sampling",
)
# parser.add_argument('--decode_mode', choices=['greedy', 'sample'],
#                    help='decoding mode')

# grammar
parser.add_argument("--grammar_type", default="ncfg", choices=["cfg", "ncfg"])
parser.add_argument(
    "--fixed_grammar_num",
    type=int,
    default=0,
    help="choice of fixed grammar for debugging",
)
parser.add_argument(
    "--grammar_param",
    type=str,
    default="mlp_neural",
    choices=["fixed", "mlp_neural"],
    help="choice of grammar parametrization",
)
parser.add_argument(
    "--extra_nts", type=int, default=90, help="extra nts for the learned grammar"
)
parser.add_argument("--num_pts", type=int, default=60, help="number of pts for grammar")
parser.add_argument("--grammar_mlp_dim", type=int, default=256)
parser.add_argument("--grammar_optimizer", default=None, choices=["sgd", "adam", None])

# GFN-PCFG tracking trick
parser.add_argument("--grammar_update_tb_threshold_max", default=10e20, type=float)
parser.add_argument("--grammar_update_tb_threshold_min", default=10e20, type=float)
parser.add_argument("--grammar_update_tb_threshold_horizon", default=10e20, type=float)
parser.add_argument("--threshold_beta", default=0.9, type=float)
parser.add_argument("--mcmc_steps", default=1, type=int)
parser.add_argument("--use_off_policy_mcmc", action="store_true")
parser.add_argument("--bnf_starting_steps", default=10, type=int)
parser.add_argument("--temp_cond_prob", default=0.0, type=float)
parser.add_argument("--temp_cond_min", default=0.0, type=float)
parser.add_argument("--temp_cond_max", default=0.0, type=float)
parser.add_argument("--ebm_reward", type=str, default=None)
parser.add_argument("--ebm_reward_temp_start", type=float, default=1.0)
parser.add_argument("--ebm_reward_temp_end", type=float, default=1.0)
parser.add_argument("--ebm_reward_temp_horizon", type=float, default=1.0)
parser.add_argument(
    "--ebm_reward_temp_schedule_type", choices=["linear", "exp"], default="linear"
)

# Misc
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument(
    "--precision", type=str, default="float", help="float | double | half"
)
parser.add_argument("--reset_lr_schedule", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--only_pretrain_grammar", action="store_true")
parser.add_argument("--restore_grammar_only", action="store_true")
parser.add_argument("--compute_grammar_spans", action="store_true", default=False)
parser.add_argument("--use_spans_f1", action="store_true")
parser.add_argument("--train_gfn", type=int, default=1)
parser.add_argument(
    "--parser_type",
    default="gfn",
    choices=["gfn", "marginalization", "sample_from_posterior"],
)

args = parser.parse_args()

# initializing cox
args_dict = args.__dict__
args_to_ignore = [
    "log_dir",
    "save_dir",
    "resume_dir",
    "tb_dir" "cuda",
    "log_interval",
    "decode_temperature",
    "decode_mode",
]
exp_id = sha1(
    repr(
        sorted(
            frozenset(filter(lambda x: x[0] not in args_to_ignore, args_dict.items()))
        )
    ).encode("ASCII")
).hexdigest()


# wandb
wandb.init(
    project="GFN-Parser-PT-marginalization",
    entity=f"{os.environ.get('WANDB_USERNAME', default='edwardhu')}",
    config=args_dict,
    tags=[args.wandb_tag],
    id=exp_id,
    mode="offline" if args.use_wandb else "disabled",
)
wandb.define_metric("train_loss", summary="min", step_metric="epoch")
wandb.define_metric("val_marginal_nll", summary="min", step_metric="grammar_step")
wandb.define_metric("val_r_pb_over_pf", summary="max", step_metric="epoch")
wandb.define_metric("val_posterior_ent", summary="min", step_metric="grammar_step")
wandb.define_metric("val_sampler_nll", summary="min", step_metric="epoch")
wandb.define_metric("val_sample_sent_f1", summary="max", step_metric="grammar_step")
wandb.define_metric("val_sample_corpus_f1", summary="max", step_metric="grammar_step")
wandb.define_metric("per_step_F_DB", summary="min", step_metric="step")
wandb.define_metric("per_step_BF_DB", summary="min", step_metric="step")
wandb.define_metric("per_step_loss", summary="min", step_metric="step")
wandb.define_metric("per_step_R_F", summary="min", step_metric="step")
wandb.define_metric("per_step_R_BnF", summary="min", step_metric="step")

parse_table = wandb.Table(
    columns=["PCFG steps", "logz_hat", "true_ll", "sample parse"], data=[]
)
tag_dist_table = wandb.Table(
    columns=["PCFG steps"] + [f"Q{i}" for i in range(args.extra_nts - args.num_pts)],
    data=[],
)

# manipulate args
args.temperature_pos = (
    args.temperature if args.temperature_pos < 0 else args.temperature_pos
)
args.temperature_tok = (
    args.temperature if args.temperature_tok < 0 else args.temperature_tok
)

print(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.save_dir is not None:
    os.makedirs(os.path.join(args.save_dir, exp_id), exist_ok=True)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################
corpus = data.SeqCorpus(
    args.data,
    seqlen=args.seqlen,
    train_batch_size=args.batch_size,
    valid_batch_size=args.valid_batch_size,
    batch_group_size=args.batch_group_size,
    add_master_token=False,
    load_spans=args.use_spans_f1,
)

train_dataloader = corpus.train
val_dataloader = corpus.valid
test_dataloader = corpus.test

if args.use_spans_f1:
    # train_spans_dataloader = corpus.train
    val_spans_dataloader = corpus.valid_spans
    test_spans_dataloader = corpus.test_spans

print("training samples:", len(train_dataloader))
print("validation samples:", len(val_dataloader))

###############################################################################
# Build the model
###############################################################################


def setprec(t):
    if args.precision == "half":
        # do nothing since this is handled by AMP
        return t
    elif args.precision == "float":
        return t.float()
    elif args.precision == "double":
        return t.double()
    else:
        raise ValueError(f"invalid precision string {args.precision}")


ntokens_src = len(corpus.dict)


controller = parser_controller(
    device,
    args={
        "n_vocab": ntokens_src,
        "reward_temperature": args.reward_temperature,
        "reward_scale": args.reward_scale,
        "vocab_dict": corpus.dict.idx2word,
        "fixed_grammar_num": args.fixed_grammar_num,
        "grammar_type": args.grammar_type,
        "grammar_param": args.grammar_param,
        "only_unary_from_s": True,
        "extra_nts": args.extra_nts,
        "num_pts": args.num_pts,
        "mlp_dim": args.grammar_mlp_dim,
        "ebm_reward": args.ebm_reward,
        "ebm_d_model": 16,
        "ebm_agg_type": "simplemlp",
    },
)

state = tree_state(
    device,
    args={
        "n_vocab": ntokens_src,
        "n_nts": len(controller.nts_list),
        "num_pts": args.num_pts,
        "seqlen": args.seqlen + 2,
        "vocab_dict": corpus.dict.idx2word,
        "start_sym": controller.grammar.start,
        "nt_dict": controller.nts_list,
    },
)

all_toks = list(corpus.dict.idx2word) + list(controller.nts_list)
id_to_token_overall = {}
for i, tok in enumerate(all_toks):
    id_to_token_overall[i] = tok
overall_tokenizer = lambda x: id_to_token_overall[x]

print(f"vocab size: {ntokens_src}")
print(f"num of nonterminal symbols: {len(controller.nts_list)}")


n_nts = len(controller.nts_list)

model_flow = GFlowNet_Z(d_model=args.d_model)
if args.agg_type == "none":
    model_shared_embedding = GFlowNet_shared_embedding(
        n_vocab=ntokens_src,
        n_nts=n_nts,
        d_model=args.d_model,
        seqlen=(args.seqlen + 2),
        grammar_emb=controller.grammar.emb_input
        if args.share_grammar_embedding
        else None,
    )
else:
    model_shared_embedding = GFlowNet_shared_embedding_with_aggregation(
        n_vocab=ntokens_src,
        n_nts=n_nts,
        d_model=args.d_model,
        seqlen=(args.seqlen + 2),
        agg_type=args.agg_type,
    )
model_encoder = GFlowNet_encoder(
    n_vocab=ntokens_src,
    n_nts=n_nts,
    d_model=args.d_model,
    nhead=args.nhead,
    dim_feedforward=4 * args.d_model,
    seqlen=(args.seqlen + 2),
    nlayers=args.nlayers,
    dropout=args.dropout,
    batch_first=True,
    norm_first=args.norm_type == "preln",
    shared_embedding=model_shared_embedding,
)
model_forward = GFlowNet_forward(
    n_nts=n_nts,
    d_model=args.d_model,
    shared_embedding=model_shared_embedding if args.agg_type == "none" else None,
    tie_tgt_embedding=False,
    preterminal_mask=controller.pt_mask,
)  # args.tie_tgt_embedding)
model_backward = GFlowNet_backward(
    n_vocab=ntokens_src, n_nts=n_nts, d_model=args.d_model
)

from mup import set_base_shapes

try:
    from torchdistx.deferred_init import deferred_init

    __no_torchdistx__ = False
except:
    __no_torchdistx__ = True

if __no_torchdistx__:
    set_base_shapes(
        model_encoder,
        GFlowNet_encoder(
            n_vocab=ntokens_src,
            n_nts=n_nts,
            d_model=128,
            nhead=args.nhead,
            dim_feedforward=128,
            seqlen=(args.seqlen + 2),
            nlayers=args.nlayers,
            dropout=args.dropout,
            batch_first=True,
            norm_first=args.norm_type == "preln",
            shared_embedding=model_shared_embedding,
        ),
        delta=GFlowNet_encoder(
            n_vocab=ntokens_src,
            n_nts=n_nts,
            d_model=256,
            nhead=args.nhead,
            dim_feedforward=256,
            seqlen=(args.seqlen + 2),
            nlayers=args.nlayers,
            dropout=args.dropout,
            batch_first=True,
            norm_first=args.norm_type == "preln",
            shared_embedding=model_shared_embedding,
        ),
    )
    set_base_shapes(model_flow, GFlowNet_Z(d_model=128), delta=GFlowNet_Z(d_model=256))
    set_base_shapes(
        model_forward,
        GFlowNet_forward(
            n_nts=n_nts,
            d_model=128,
            shared_embedding=model_shared_embedding
            if args.agg_type == "none"
            else None,
            tie_tgt_embedding=False,
            preterminal_mask=controller.pt_mask,
        ),
        delta=GFlowNet_forward(
            n_nts=n_nts,
            d_model=256,
            shared_embedding=model_shared_embedding
            if args.agg_type == "none"
            else None,
            tie_tgt_embedding=False,
            preterminal_mask=controller.pt_mask,
        ),
    )
    set_base_shapes(
        model_backward,
        GFlowNet_backward(n_vocab=ntokens_src, n_nts=n_nts, d_model=128),
        delta=GFlowNet_backward(n_vocab=ntokens_src, n_nts=n_nts, d_model=256),
    )
    if args.grammar_param == "mlp_neural":
        set_base_shapes(
            controller.grammar,
            MLPPCFG(
                n_nts,
                ntokens_src,
                ntokens_src,
                np.concatenate(
                    (np.zeros(n_nts - args.num_pts), np.ones(args.num_pts))
                ).astype(bool),
                True,
                128,
                grammar_type=args.grammar_type,
            ),
            delta=MLPPCFG(
                n_nts,
                ntokens_src,
                ntokens_src,
                np.concatenate(
                    (np.zeros(n_nts - args.num_pts), np.ones(args.num_pts))
                ).astype(bool),
                True,
                512,
                grammar_type=args.grammar_type,
            ),
        )
else:
    set_base_shapes(
        model_encoder,
        deferred_init(
            GFlowNet_encoder,
            n_vocab=ntokens_src,
            n_nts=n_nts,
            d_model=128,
            nhead=args.nhead,
            dim_feedforward=128,
            seqlen=(args.seqlen + 2),
            nlayers=args.nlayers,
            dropout=args.dropout,
            batch_first=True,
            norm_first=args.norm_type == "preln",
            shared_embedding=model_shared_embedding,
        ),
        delta=deferred_init(
            GFlowNet_encoder,
            n_vocab=ntokens_src,
            n_nts=n_nts,
            d_model=256,
            nhead=args.nhead,
            dim_feedforward=256,
            seqlen=(args.seqlen + 2),
            nlayers=args.nlayers,
            dropout=args.dropout,
            batch_first=True,
            norm_first=args.norm_type == "preln",
            shared_embedding=model_shared_embedding,
        ),
    )
    set_base_shapes(
        model_flow,
        deferred_init(GFlowNet_Z, d_model=128),
        delta=deferred_init(GFlowNet_Z, d_model=256),
    )
    set_base_shapes(
        model_forward,
        deferred_init(
            GFlowNet_forward,
            n_nts=n_nts,
            d_model=128,
            shared_embedding=model_shared_embedding
            if args.agg_type == "none"
            else None,
            tie_tgt_embedding=False,
            preterminal_mask=controller.pt_mask,
        ),
        delta=deferred_init(
            GFlowNet_forward,
            n_nts=n_nts,
            d_model=256,
            shared_embedding=model_shared_embedding
            if args.agg_type == "none"
            else None,
            tie_tgt_embedding=False,
            preterminal_mask=controller.pt_mask,
        ),
    )
    set_base_shapes(
        model_backward,
        deferred_init(GFlowNet_backward, n_vocab=ntokens_src, n_nts=n_nts, d_model=128),
        delta=deferred_init(
            GFlowNet_backward, n_vocab=ntokens_src, n_nts=n_nts, d_model=256
        ),
    )
    if args.grammar_param == "mlp_neural":
        set_base_shapes(
            controller.grammar,
            deferred_init(
                MLPPCFG,
                n_nts,
                ntokens_src,
                ntokens_src,
                np.concatenate(
                    (np.zeros(n_nts - args.num_pts), np.ones(args.num_pts))
                ).astype(bool),
                True,
                128,
                grammar_type=args.grammar_type,
            ),
            delta=deferred_init(
                MLPPCFG,
                n_nts,
                ntokens_src,
                ntokens_src,
                np.concatenate(
                    (np.zeros(n_nts - args.num_pts), np.ones(args.num_pts))
                ).astype(bool),
                True,
                512,
                grammar_type=args.grammar_type,
            ),
        )

model_flow = setprec(model_flow).to(device)
model_encoder = setprec(model_encoder).to(device)
model_shared_embedding = setprec(model_shared_embedding).to(device)
model_forward = setprec(model_forward).to(device)
model_backward = setprec(model_backward).to(device)
if args.train_grammar == 1:
    controller.grammar = setprec(controller.grammar).to(device)

# adjust GFN init std
for p in model_encoder.parameters():
    p.data *= args.init_mult_encoder
for p in model_forward.parameters():
    p.data *= args.init_mult_forward
for p in model_backward.parameters():
    p.data *= args.init_mult_backward
for p in model_flow.parameters():
    p.data *= args.init_mult_flow

"""
import numpy as np
pretrained_emb = np.load('ptb.emb.npy')
pretrained_vocab = np.load('ptb.vocab.npy')
pretrained_ind = torch.tensor([np.where(pretrained_vocab==word)[0].item() if word in pretrained_vocab else -1 for word in corpus.dict.idx2word], dtype=torch.long)
pretrained_emb = torch.from_numpy(pretrained_emb)
pretrained_emb = torch.cat([pretrained_emb, pretrained_emb.mean(dim=0, keepdim=True)], dim=0)
pretrained_emb_mat = pretrained_emb[pretrained_ind]
model_shared_embedding.embedding_tgt.weight.data[:pretrained_emb_mat.size(0), :] = pretrained_emb_mat.to(model_shared_embedding.embedding_tgt.weight.data.dtype).to(device)
"""
###############################################################################
# Training code
###############################################################################


def get_ebm_temp(grammar_step):
    if grammar_step >= args.ebm_reward_temp_horizon:
        return args.ebm_reward_temp_end
    if args.ebm_reward_temp_schedule_type == "linear":
        return args.ebm_reward_temp_start + (
            args.ebm_reward_temp_end - args.ebm_reward_temp_start
        ) * (grammar_step / args.ebm_reward_temp_horizon)
    elif args.ebm_reward_temp_schedule_type == "exp":
        c = (grammar_step / args.ebm_reward_temp_horizon) * math.log(
            args.ebm_reward_temp_end / args.ebm_reward_temp_start
        )
        # print(args.ebm_reward_temp_start * math.exp(c))
        return args.ebm_reward_temp_start * math.exp(c)
    else:
        raise NotImplementedError


def compare_spans(pred_span, label_span):
    pred_span_set = set(sorted(set(pred_span), key=lambda x: x[1])[:-1])
    label_span_set = set(label_span[:-1])
    tp = 0
    fp = 0
    fn = 0
    for span in pred_span_set:
        if span in label_span_set:
            tp += 1
        else:
            fp += 1
    for span in label_span_set:
        if span not in pred_span_set:
            fn += 1

    overlap = pred_span_set.intersection(label_span_set)
    prec = float(len(overlap)) / (len(pred_span_set) + 1e-8)
    reca = float(len(overlap)) / (len(label_span_set) + 1e-8)
    if len(label_span_set) == 0:
        reca = 1.0
        if len(pred_span_set) == 0:
            prec = 1.0
    f1 = 2 * prec * reca / (prec + reca + 1e-8)

    return f1, (tp, fp, fn)


@torch.no_grad()
def decode_and_evaluate(original_seq, gold_spans):
    model_flow.eval()
    model_encoder.eval()
    model_backward.eval()
    model_forward.eval()
    forward_seq = [seq.clone() for seq in original_seq]

    def calc_stats_from_torch_struct(sample):
        sent_f1s = []
        corpus_f1s = [0, 0, 0]
        charts = sample[3].detach().cpu().numpy()
        tags = charts.argmax(-1).reshape(-1)
        if gold_spans is not None:
            for i in range(len(charts)):
                spans = [(s, s + t + 1) for s, t in zip(*charts[i].nonzero()[1::-1])]
                gold_span = gold_spans[i]
                sent_f1, corpus_f1 = compare_spans(spans, gold_span)
                sent_f1s.append(sent_f1)
                corpus_f1s = [corpus_f1s[i] + corpus_f1[i] for i in range(3)]
        return (sent_f1s, corpus_f1s), tags

    def calc_stats_from_gfn(forward_seq):
        sent_f1s = []
        corpus_f1s = [0, 0, 0]
        if gold_spans is not None:
            for i, seq in enumerate(forward_seq):
                # convert to (start, end) and ignore PT tags
                spans = [
                    (span[0], span[0] + span[1] - 1)
                    for span in seq.trees[0].all_spans(0)[0]
                    if span[1] != 1
                ]
                gold_span = gold_spans[i]
                sent_f1, corpus_f1 = compare_spans(spans, gold_span)
                sent_f1s.append(sent_f1)
                corpus_f1s = [corpus_f1s[i] + corpus_f1[i] for i in range(3)]
        return (sent_f1s, corpus_f1s)

    seqs = [
        torch.tensor([root.data for root in state._state], device=device)
        for state in forward_seq
    ]
    lengths = [len(s) for s in seqs]
    batch_seq = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)

    if args.parser_type == "gfn":
        bsz = len(forward_seq)
        encoded_tokens, pad_mask = model_encoder(
            forward_seq,
            seq_type="all_root",
            temp_cond=0 if args.temp_cond_prob > 0 else None,
        )
        if args.tjb_backward == 1 or args.tjb_forward >= 1:
            estimated_ll = model_flow(encoded_tokens, pad_mask=pad_mask)
        else:
            estimated_ll = encoded_tokens.new_zeros(len(encoded_tokens))
        forward_step = 0
        forward_logp = torch.zeros((bsz,), device=device)
        backward_logp = torch.zeros((bsz,), device=device)
        while True:
            F_logits = model_forward(encoded_tokens)
            sample_forward_result = controller.sample_forward(
                F_logits, forward_seq, greedy=False
            )
            forward_seq = sample_forward_result["new_states"]
            forward_logp += sample_forward_result["policy_log_pf"]
            backward_actions = sample_forward_result["backward_actions"]
            forward_step += 1
            encoded_tokens, _ = model_encoder(
                forward_seq,
                seq_type="all_root",
                temp_cond=0 if args.temp_cond_prob > 0 else None,
            )
            B_logits = model_backward(encoded_tokens, uniform_pos=args.uniform_pb)
            backward_logp += controller.batch_calc_backward_prob(
                B_logits, forward_seq, B_actions=backward_actions
            )[0]
            if all([s.is_terminated for s in forward_seq]):
                break

        if args.mc_em:
            for mcmc_i in range(args.mcmc_steps):
                starting_seq = [seq.clone() for seq in forward_seq]
                starting_ll = controller.calc_log_reward(
                    starting_seq, temp_cond=0 if args.temp_cond_prob > 0 else None
                )
                for i in range(len(forward_seq)):
                    if mcmc_i % 2 == 0:
                        forward_seq[i]._state[0].random_rotate()
                    else:
                        forward_seq[i]._state[0].random_change_symbol(
                            [
                                x + ntokens_src
                                for x in range(args.extra_nts - args.num_pts)
                            ]
                        )
                    forward_seq[i]._state[0].clear_ll()
                forward_seq_ll = controller.calc_log_reward(
                    forward_seq, temp_cond=0 if args.temp_cond_prob > 0 else None
                )
                mcmc_accept_prob = forward_seq_ll - starting_ll
                mcmc_outcomes = (
                    mcmc_accept_prob
                    < torch.rand(starting_ll.shape, device=starting_ll.device).log()
                )
                new_seq = []
                for i, ele in enumerate(mcmc_outcomes):
                    if ele.item():
                        new_seq.append(starting_seq[i])
                    else:
                        new_seq.append(forward_seq[i])
                forward_seq = new_seq

        # print an example parse
        sample_parse = (
            f"{estimated_ll[0]:2.3f}",
            "\t",
            forward_seq[0]._state[0].print(tostr=overall_tokenizer),
        )
        # sample_parse = None
        sampler_ll = controller.calc_log_reward(forward_seq)
        parse_table.add_data(
            f"{global_grammar_updates}",
            f"{estimated_ll[0]:2.3f}",
            f"{sampler_ll[0]:2.3f}",
            forward_seq[0]._state[0].print(tostr=overall_tokenizer),
        )
        # spans and tags
        tags = []
        for i, seq in enumerate(forward_seq):
            # convert to (start, end) and ignore PT tags
            tags += seq.trees[0].all_tags()
        tags = np.array(tags) - ntokens_src
        sampled_parse_f1s = calc_stats_from_gfn(forward_seq)
    else:
        # baselines
        if args.grammar_type == "ncfg":
            raise NotImplementedError
        else:
            sample, sampler_ll = controller.grammar.sample_batch(batch_seq, lengths)
            sampled_parse_f1s, tags = calc_stats_from_torch_struct(sample)
            estimated_ll = sampler_ll.new_zeros(sampler_ll.shape)
            forward_logp = sampler_ll.new_zeros(sampler_ll.shape)
            backward_logp = sampler_ll.new_zeros(sampler_ll.shape)
            sample_parse = None

    # calculate the F1 for the most likely parse from the grammar
    if args.grammar_type == "ncfg":
        max_parse_f1s = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])  # no max parse for NCFG
    else:
        sample, sampler_ll = controller.grammar.maxll_batch(batch_seq, lengths)
        max_parse_f1s, _ = calc_stats_from_torch_struct(sample)

    return (
        sampler_ll,
        estimated_ll,
        forward_logp,
        backward_logp,
        (sampled_parse_f1s, max_parse_f1s),
        tags[tags.nonzero()],
        sample_parse,
    )


@torch.no_grad()
def valid():
    sampler_ll = []
    estimated_ll = []
    forward_logp = []
    backward_logp = []
    pcfg_max_ll = []
    posterior_ent = []
    sample_sent_f1 = []
    sample_corpus_f1 = [0.0, 0.0, 0.0]
    tag_bincount = None
    lengths = []
    sample_parses = []
    iterable = (
        zip(val_dataloader, val_spans_dataloader)
        if args.use_spans_f1
        else val_dataloader
    )
    # for PCFG
    marginal_ll = []
    pcfg_max_ll = []
    posterior_ent = []
    max_sent_f1 = []
    max_corpus_f1 = [0.0, 0.0, 0.0]
    for item in iterable:
        if args.use_spans_f1:
            src, gold_span = item
        else:
            src = item
            gold_span = None
        # calculate marginal LL and pcfg max LL under the grammar
        forward_seq = [state.from_iterable(t) for t in src]
        seqs = [
            torch.tensor([root.data for root in state._state], device=device)
            for state in forward_seq
        ]
        batch_lengths = [len(s) for s in seqs]
        if args.grammar_type == "cfg":
            batch_seq = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
            batch_marginal_ll, batch_pcfg_max_ll, batch_posterior_ent, _ = (
                controller.grammar.compute_marginal_batch(
                    batch_seq, batch_lengths, args.compute_grammar_spans
                )
                if args.grammar_param != "fixed"
                else torch.tensor(
                    [controller.grammar.compute_marginal(seq) for seq in seqs]
                )
            )
            batch_marginal_ll = batch_marginal_ll.data
            batch_pcfg_max_ll = batch_pcfg_max_ll.data
            batch_posterior_ent = batch_posterior_ent.data
        else:
            batch_marginal_ll, batch_pcfg_max_ll, batch_posterior_ent = (
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                torch.tensor([0.0]),
            )
        # calculate sample LL from the sampler
        (
            batch_sampler_ll,
            batch_estimated_ll,
            batch_forward_logp,
            batch_backward_logp,
            f1_stats,
            batch_tags,
            sample_parse,
        ) = decode_and_evaluate(forward_seq, gold_span)
        sample_parses.append(sample_parse)
        if args.use_spans_f1:
            sample_sent_f1.extend(f1_stats[0][0])
            max_sent_f1.extend(f1_stats[1][0])
            sample_corpus_f1 = [
                sample_corpus_f1[i] + f1_stats[0][1][i] for i in range(3)
            ]
            max_corpus_f1 = [max_corpus_f1[i] + f1_stats[1][1][i] for i in range(3)]
        marginal_ll.append(batch_marginal_ll)
        pcfg_max_ll.append(batch_pcfg_max_ll)
        posterior_ent.append(batch_posterior_ent)
        sampler_ll.append(batch_sampler_ll)
        estimated_ll.append(batch_estimated_ll)
        forward_logp.append(batch_forward_logp)
        backward_logp.append(batch_backward_logp)
        if tag_bincount is None:
            tag_bincount = np.bincount(batch_tags, minlength=args.extra_nts)
        else:
            tag_bincount += np.bincount(batch_tags, minlength=args.extra_nts)
        # accounting for </s>
        for i in range(len(batch_lengths)):
            batch_lengths[i] += 1
        lengths += batch_lengths

    marginal_ll = torch.concat(marginal_ll, dim=0)
    pcfg_max_ll = torch.concat(pcfg_max_ll, dim=0)
    sampler_ll = torch.concat(sampler_ll, dim=0)
    estimated_ll = torch.concat(estimated_ll, dim=0)
    forward_logp = torch.concat(forward_logp, dim=0)
    backward_logp = torch.concat(backward_logp, dim=0)
    posterior_ent = torch.concat(posterior_ent, dim=0)
    if args.grammar_type == "cfg":
        spearman_corr = scipy.stats.spearmanr(marginal_ll.cpu(), estimated_ll.cpu())[0]
    else:
        spearman_corr = 0.0

    def calc_corpus_f1(stats):
        if args.use_spans_f1:
            tp, fp, fn = stats
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (
            100 * (2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0)
            if args.use_spans_f1
            else 0
        )

    sample_corpus_f1 = calc_corpus_f1(sample_corpus_f1)
    max_corpus_f1 = calc_corpus_f1(max_corpus_f1)
    sample_sent_f1_mean = np.mean(sample_sent_f1) * 100 if args.use_spans_f1 else 0
    max_sent_f1_mean = np.mean(max_sent_f1) * 100 if args.use_spans_f1 else 0
    ll_stats = {
        "sampler_ll": sampler_ll,
        "marginal_ll": marginal_ll,
        "r_pb_over_pf": sampler_ll + backward_logp - forward_logp,
        "pcfg_max_ll": pcfg_max_ll,
        "estimated_ll": estimated_ll,
        "posterior_ent": posterior_ent,
        "sample_sent_f1": sample_sent_f1_mean,
        "sample_corpus_f1": sample_corpus_f1,
        "max_sent_f1": max_sent_f1_mean,
        "max_corpus_f1": max_corpus_f1,
        "spearman_corr": spearman_corr,
        "tag_bincount": tag_bincount[: args.extra_nts - args.num_pts],
        "sample_parses": sample_parses,
    }
    # log tag counts
    tag_dist_table.add_data(
        f"{global_grammar_updates}",
        *[f"{x}" for x in tag_bincount[: args.extra_nts - args.num_pts].tolist()],
    )
    return ll_stats, lengths


def filter_len(tensor_loader, max_len):
    bsz = tensor_loader.size(1)
    new_tensor_loader = tensor_loader.view(-1, tensor_loader.size(-1)).clone()
    new_tensor_loader = new_tensor_loader[(new_tensor_loader != 0).sum(-1) <= max_len]
    new_tensor_loader = new_tensor_loader[: new_tensor_loader.size(0) // bsz * bsz]
    return new_tensor_loader.view(-1, bsz, new_tensor_loader.size(-1))


def validate_and_log():
    val_ll_stats, val_lengths = valid()

    val_sample_sent_f1 = val_ll_stats["sample_sent_f1"]
    val_sample_corpus_f1 = val_ll_stats["sample_corpus_f1"]
    val_max_sent_f1 = val_ll_stats["max_sent_f1"]
    val_max_corpus_f1 = val_ll_stats["max_corpus_f1"]

    val_marginal_nll = -val_ll_stats["marginal_ll"].sum().item() / sum(val_lengths)
    val_sampler_nll = -val_ll_stats["sampler_ll"].sum().item() / sum(val_lengths)
    val_estimated_nll = -val_ll_stats["estimated_ll"].sum().item() / sum(val_lengths)
    val_posterior_ent = val_ll_stats["posterior_ent"].sum().item() / len(val_lengths)
    val_r_pb_over_pf = val_ll_stats["r_pb_over_pf"].mean()

    wandb.log(
        {
            "val_marginal_nll": val_marginal_nll,
            "grammar_step": global_grammar_updates,
            "val_sampler_nll": val_sampler_nll,
            "val_estimated_nll": val_estimated_nll,
            "val_sample_sent_f1": val_sample_sent_f1,
            "val_sample_corpus_f1": val_sample_corpus_f1,
            "val_r_pb_over_pf": val_r_pb_over_pf,
            "val_max_sent_f1": val_max_sent_f1,
            "val_max_corpus_f1": val_max_corpus_f1,
            "val_posterior_ent": val_posterior_ent,
            "val_tag_dist_table": tag_dist_table,
            "sample_parse_table": parse_table,
            "val_tag_dist": val_ll_stats["tag_bincount"],
            "sample_parses": val_ll_stats["sample_parses"],
            "grammar_type": args.grammar_type,
        }
    )


F_db_loss_smoothed = torch.tensor(0.0)
global_grammar_updates = 0


# torch.autograd.set_detect_anomaly(True)
def train(epoch, optimizers, schedulers, uniform_pos=False, max_len=9999):
    global F_db_loss_smoothed
    global global_grammar_updates
    model_flow.train()
    model_encoder.train()
    model_backward.train()
    model_forward.train()
    epoch_loss = 0.0
    forward_steps_list = []
    backward_steps_list = []
    num_correct = 0
    total_num_correct = 0
    total_num = 0
    Z = []
    R_F = []
    R_BF = []
    len_ratio = []
    start_time = time.time()
    first_loss = None
    global global_step
    total_PCFG_loss = 0  # scalar for logging
    total_LLLB_loss = 0
    total_B_db_loss = 0
    total_F_db_loss = 0
    total_BF_db_loss = 0
    total_sleep_mle_loss = 0
    total_PCFG_updates = 0

    F_db_loss_min = 999

    update_grammar = (
        False or not (args.train_gfn) or args.grammar_update_tb_threshold_max == 10e10
    )

    controller.grammar.cache_params()

    for opt in optimizers:
        opt.zero_grad()
    """
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=2,
            repeat=1),
        with_stack=True) as profiler:
        #on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./tensorboard/{exp_id}_1'),
    """
    for sent_id, src in enumerate(filter_len(train_dataloader, max_len), 1):
        # trim zeros
        # print(sent_id)
        src = src[:, src.sum(dim=0) != 0]
        # warmup
        if global_step < args.warmup and args.train_gfn:
            for g in optimizers[0].param_groups:
                g["lr"] = g["initial_lr"] * (global_step / args.warmup)

        if global_grammar_updates < args.warmup and args.train_grammar:
            for g in optimizers[-1].param_groups:
                g["lr"] = g["initial_lr"] * (global_grammar_updates / args.warmup)

        # split the batch
        split_src = src.view(args.grad_acc, src.size(0) // args.grad_acc, src.size(1))
        for acc_step in range(args.grad_acc):
            forward_seq = [state.from_iterable(t) for t in split_src[acc_step]]
            original_seqs = [
                torch.tensor(state.leaf_seq, device=device) for state in forward_seq
            ]
            bsz = len(forward_seq)

            PCFG_loss = 0
            F_db_loss = torch.tensor(0.0)
            B_db_loss = 0
            LLLB_loss = 0
            BF_db_loss = 0
            sleep_mle_loss = 0
            forward_steps = 0
            backward_steps = 0

            # temperature conditioning
            if args.temp_cond_prob > 0:
                if random.random() < args.temp_cond_prob:
                    temp_cond = random.random() * (
                        math.log(args.temp_cond_max) - math.log(args.temp_cond_min)
                    ) + math.log(args.temp_cond_min)
                else:
                    temp_cond = 0.0
            else:
                temp_cond = None

            if args.tjb_forward:
                # =============================================================
                # ================== Going forward from s0 ===================
                # =============================================================
                action_seq_lengths = [s.size(0) - 1 for s in original_seqs]
                F_logP_act_B = torch.zeros(
                    (bsz, max(action_seq_lengths)), device=device
                )
                F_logP_act_F = torch.zeros(
                    (bsz, max(action_seq_lengths)), device=device
                )
                F_logflows = torch.zeros(
                    (bsz, max(action_seq_lengths) + 1), device=device
                )
                partial_rewards = torch.zeros((bsz,), device=device)

                encoded_tokens, pad_mask = model_encoder(
                    forward_seq, seq_type="all_root", temp_cond=temp_cond
                )

                batch_token_dropout_mask = torch.full(
                    (bsz, args.extra_nts), args.symbol_dropout, device=device
                ).bernoulli()
                while (
                    batch_token_dropout_mask[:, 1 : -args.num_pts].sum(1)
                    == args.extra_nts - args.num_pts - 1
                ).any():
                    batch_token_dropout_mask = torch.full(
                        (bsz, args.extra_nts), args.symbol_dropout, device=device
                    ).bernoulli()

                while True:
                    # we could speed this up: the partial log reward is always increased by the score of the newly taken action, no need to recompute each time
                    # partial_log_reward = controller.calc_partial_log_reward(forward_seq)
                    F_logflows[:, forward_steps] = partial_rewards / np.exp(
                        temp_cond if temp_cond is not None else 0.0
                    ) + model_flow(encoded_tokens, pad_mask=pad_mask)
                    F_logits = model_forward(encoded_tokens, uniform_pos=uniform_pos)
                    sample_forward_result = controller.sample_forward(
                        F_logits,
                        forward_seq,
                        greedy=False,
                        temperature_pos=args.temperature_pos,
                        temperature_tok=args.temperature_tok,
                        uniform_eps_pos=args.pf_uniform_eps_pos,
                        uniform_eps_tok=args.pf_uniform_eps_tok,
                        batch_token_dropout_mask=batch_token_dropout_mask,
                    )
                    forward_seq = sample_forward_result["new_states"]
                    logP_F = sample_forward_result["policy_log_pf"]
                    B_actions = sample_forward_result["backward_actions"]
                    reward_term = sample_forward_result["reward_term"]

                    encoded_tokens, pad_mask = model_encoder(
                        forward_seq, seq_type="all_root", temp_cond=temp_cond
                    )
                    B_logits = model_backward(
                        encoded_tokens, uniform_pos=args.uniform_pb
                    )
                    logP_B = controller.batch_calc_backward_prob(
                        B_logits, forward_seq, B_actions=B_actions
                    )[0]
                    F_logP_act_B[:, forward_steps] = logP_B
                    F_logP_act_F[:, forward_steps] = logP_F
                    partial_rewards += reward_term
                    if all([s.is_terminated for s in forward_seq]):
                        break
                    forward_steps += 1
                    # start over

                Z.append(F_logflows[:, 0].mean().item())

                forward_ll = controller.calc_log_reward(
                    forward_seq,
                    temp_cond=temp_cond,
                    ebm_reward_temp=get_ebm_temp(global_grammar_updates),
                )
                F_logP_traj_F = torch.zeros((bsz,), device=device)
                F_logP_traj_B = torch.zeros((bsz,), device=device)
                for i, l, ll in zip(range(bsz), action_seq_lengths, forward_ll):
                    F_logflows[i, l] = ll
                    F_logP_traj_F[i] = F_logP_act_F[i, :l].sum()
                    F_logP_traj_B[i] = F_logP_act_B[i, :l].sum()

                if args.tjb_forward == 2:  # hvi
                    reward = (forward_ll + F_logP_traj_B - F_logP_traj_F).detach()
                    loss_pb = -F_logP_traj_B
                    loss_pf = -F_logP_traj_F * (reward - reward.mean())
                    F_db_loss = loss_pb + loss_pf
                elif args.tjb_forward == 1 and args.subtb_lambda < 0:  # TB
                    F_db_loss = (
                        forward_ll + F_logP_traj_B - F_logflows[:, 0] - F_logP_traj_F
                    ) ** 2
                else:
                    lengths = torch.LongTensor(action_seq_lengths).to(device)
                    weights = torch.zeros((bsz,), device=device)
                    F_db_loss = torch.zeros_like(weights)
                    for i in range(max(action_seq_lengths)):
                        for j in range(i + 1, max(action_seq_lengths) + 1):
                            weight = args.subtb_lambda ** (j - i - 1)
                            mask = lengths >= j
                            weights[mask] += weight
                            F_db_loss[mask] += (
                                weight
                                * (
                                    F_logflows[mask, i]
                                    + F_logP_act_F[mask, i:j].sum(1)
                                    - F_logflows[mask, j]
                                    - F_logP_act_B[mask, i:j].sum(1)
                                )
                                ** 2
                            )
                    F_db_loss = F_db_loss / weights

                F_db_loss = F_db_loss.mean() / args.grad_acc

                F_db_loss.backward()
                R_F.append(controller.calc_log_reward(forward_seq).mean().item())
                # print(controller.calc_log_reward(forward_seq)[0])

            mcmc_steps = (
                args.mcmc_steps
                if update_grammar and (temp_cond is None or temp_cond == 0)
                else 1
            )

            back_and_forth_seq = forward_seq
            if args.go_back_and_forward:
                for mcmc_i in range(mcmc_steps):
                    # =============================================================
                    # ================== Going back and forth =====================
                    # =============================================================
                    BF_logP_traj_BB = torch.zeros(bsz, device=device)
                    BF_logP_traj_FF = torch.zeros(bsz, device=device)
                    BF_logP_traj_BF = torch.zeros(bsz, device=device)
                    BF_logP_traj_FB = torch.zeros(bsz, device=device)

                    BF_logP_true_BB = torch.zeros(bsz, device=device)
                    BF_logP_true_FF = torch.zeros(bsz, device=device)
                    BF_logP_true_BF = torch.zeros(bsz, device=device)
                    BF_logP_true_FB = torch.zeros(bsz, device=device)
                    for seq in back_and_forth_seq:
                        seq._state[0].reset_emb()
                    starting_seq = [seq.clone() for seq in back_and_forth_seq]
                    if mcmc_i == mcmc_steps - 1:
                        encoded_tokens, _ = model_encoder(
                            back_and_forth_seq, seq_type="all_root", temp_cond=temp_cond
                        )
                        for i in range(
                            random.randint(
                                1, min(args.bnf_starting_steps + epoch, args.seqlen)
                            )
                        ):
                            if all([seq.is_s0 for seq in back_and_forth_seq]):
                                break
                            B_logits = model_backward(
                                encoded_tokens, uniform_pos=args.uniform_pb
                            )
                            sample_backward_result = controller.sample_backward(
                                B_logits,
                                back_and_forth_seq,
                                greedy=False,
                                temperature_pos=args.temperature_pos
                                if args.use_off_policy_mcmc
                                else 1.0,
                            )
                            back_and_forth_seq = sample_backward_result["new_states"]
                            logP_B = sample_backward_result["policy_log_pb"]
                            true_logP_B = sample_backward_result["true_log_pb"]
                            F_actions = sample_backward_result["forward_actions"]
                            encoded_tokens, _ = model_encoder(
                                back_and_forth_seq,
                                seq_type="all_root",
                                temp_cond=temp_cond,
                            )
                            F_logits = model_forward(
                                encoded_tokens, uniform_pos=uniform_pos
                            )
                            logP_F, true_logP_F = controller.batch_calc_forward_prob(
                                F_logits,
                                back_and_forth_seq,
                                F_actions=F_actions,
                                temperature_pos=args.temperature_pos
                                if args.use_off_policy_mcmc
                                else 1.0,
                                temperature_tok=args.temperature_tok
                                if args.use_off_policy_mcmc
                                else 1.0,
                            )
                            BF_logP_traj_BB += logP_B
                            BF_logP_traj_BF += logP_F
                            BF_logP_true_BB += true_logP_B
                            BF_logP_true_BF += true_logP_F

                        forward_steps = 0
                        while True:
                            F_logits = model_forward(
                                encoded_tokens, uniform_pos=uniform_pos
                            )
                            sample_forward_result = controller.sample_forward(
                                F_logits,
                                back_and_forth_seq,
                                greedy=False,
                                temperature_pos=args.temperature_pos
                                if args.use_off_policy_mcmc
                                else 1.0,
                                temperature_tok=args.temperature_tok
                                if args.use_off_policy_mcmc
                                else 1.0,
                            )
                            back_and_forth_seq = sample_forward_result["new_states"]
                            logP_F = sample_forward_result["policy_log_pf"]
                            true_logP_F = sample_forward_result["true_log_pf"]
                            B_actions = sample_forward_result["backward_actions"]

                            encoded_tokens, _ = model_encoder(
                                back_and_forth_seq,
                                seq_type="all_root",
                                temp_cond=temp_cond,
                            )
                            B_logits = model_backward(
                                encoded_tokens, uniform_pos=args.uniform_pb
                            )
                            logP_B, true_logP_B = controller.batch_calc_backward_prob(
                                B_logits,
                                back_and_forth_seq,
                                B_actions=B_actions,
                                temperature_pos=args.temperature_pos
                                if args.use_off_policy_mcmc
                                else 1.0,
                            )
                            BF_logP_traj_FB += logP_B
                            BF_logP_traj_FF += logP_F
                            BF_logP_true_FF += true_logP_F
                            BF_logP_true_FB += true_logP_B
                            forward_steps += 1
                            if all([s.is_terminated for s in back_and_forth_seq]):
                                break
                            # start over
                        back_and_forth_ll = controller.calc_log_reward(
                            back_and_forth_seq,
                            temp_cond=temp_cond,
                            ebm_reward_temp=get_ebm_temp(global_grammar_updates),
                        )
                        starting_ll = controller.calc_log_reward(
                            starting_seq,
                            temp_cond=temp_cond,
                            ebm_reward_temp=get_ebm_temp(global_grammar_updates),
                        )
                        BF_db_loss += (
                            starting_ll
                            + BF_logP_traj_BB
                            + BF_logP_traj_FF
                            - back_and_forth_ll
                            - BF_logP_traj_FB
                            - BF_logP_traj_BF
                        ) ** 2
                        BF_db_loss = args.bnf_gamma * BF_db_loss.mean() / args.grad_acc
                        BF_db_loss.backward()
                    else:
                        with torch.no_grad():
                            encoded_tokens, _ = model_encoder(
                                back_and_forth_seq,
                                seq_type="all_root",
                                temp_cond=temp_cond,
                            )
                            # for i in range(random.randint(1, min(epoch+1, args.seqlen-1))):
                            for i in range(random.randint(1, args.seqlen)):
                                if all([seq.is_s0 for seq in back_and_forth_seq]):
                                    break
                                B_logits = model_backward(
                                    encoded_tokens, uniform_pos=args.uniform_pb
                                )
                                sample_backward_result = controller.sample_backward(
                                    B_logits,
                                    back_and_forth_seq,
                                    greedy=False,
                                    temperature_pos=args.temperature_pos
                                    if args.use_off_policy_mcmc
                                    else 1.0,
                                )
                                back_and_forth_seq = sample_backward_result[
                                    "new_states"
                                ]
                                logP_B = sample_backward_result["policy_log_pb"]
                                true_logP_B = sample_backward_result["true_log_pb"]
                                F_actions = sample_backward_result["forward_actions"]
                                encoded_tokens, _ = model_encoder(
                                    back_and_forth_seq,
                                    seq_type="all_root",
                                    temp_cond=temp_cond,
                                )
                                F_logits = model_forward(
                                    encoded_tokens, uniform_pos=uniform_pos
                                )
                                (
                                    logP_F,
                                    true_logP_F,
                                ) = controller.batch_calc_forward_prob(
                                    F_logits,
                                    back_and_forth_seq,
                                    F_actions=F_actions,
                                    temperature_pos=args.temperature_pos
                                    if args.use_off_policy_mcmc
                                    else 1.0,
                                    temperature_tok=args.temperature_tok
                                    if args.use_off_policy_mcmc
                                    else 1.0,
                                )

                                BF_logP_traj_BB += logP_B
                                BF_logP_traj_BF += logP_F
                                BF_logP_true_BB += true_logP_B
                                BF_logP_true_BF += true_logP_F

                            forward_steps = 0
                            while True:
                                F_logits = model_forward(
                                    encoded_tokens, uniform_pos=uniform_pos
                                )
                                sample_forward_result = controller.sample_forward(
                                    F_logits,
                                    back_and_forth_seq,
                                    greedy=False,
                                    temperature_pos=args.temperature_pos
                                    if args.use_off_policy_mcmc
                                    else 1.0,
                                    temperature_tok=args.temperature_tok
                                    if args.use_off_policy_mcmc
                                    else 1.0,
                                )
                                back_and_forth_seq = sample_forward_result["new_states"]
                                logP_F = sample_forward_result["policy_log_pf"]
                                true_logP_F = sample_forward_result["true_log_pf"]
                                B_actions = sample_forward_result["backward_actions"]

                                encoded_tokens, _ = model_encoder(
                                    back_and_forth_seq,
                                    seq_type="all_root",
                                    temp_cond=temp_cond,
                                )
                                B_logits = model_backward(
                                    encoded_tokens, uniform_pos=args.uniform_pb
                                )
                                (
                                    logP_B,
                                    true_logP_B,
                                ) = controller.batch_calc_backward_prob(
                                    B_logits,
                                    back_and_forth_seq,
                                    B_actions=B_actions,
                                    temperature_pos=args.temperature_pos
                                    if args.use_off_policy_mcmc
                                    else 1.0,
                                )
                                BF_logP_traj_FB += logP_B
                                BF_logP_traj_FF += logP_F
                                BF_logP_true_FF += true_logP_F
                                BF_logP_true_FB += true_logP_B
                                forward_steps += 1
                                if all([s.is_terminated for s in back_and_forth_seq]):
                                    break
                            back_and_forth_ll = controller.calc_log_reward(
                                back_and_forth_seq,
                                temp_cond=temp_cond,
                                ebm_reward_temp=get_ebm_temp(global_grammar_updates),
                            )
                            starting_ll = controller.calc_log_reward(
                                starting_seq,
                                temp_cond=temp_cond,
                                ebm_reward_temp=get_ebm_temp(global_grammar_updates),
                            )
                    mcmc_accept_prob = (
                        back_and_forth_ll
                        - starting_ll
                        - BF_logP_true_BB
                        - BF_logP_true_FF
                        + BF_logP_true_FB
                        + BF_logP_true_BF
                    )
                    mcmc_outcomes = (
                        mcmc_accept_prob
                        < torch.rand(starting_ll.shape, device=starting_ll.device).log()
                    )
                    new_seq = []
                    for i, ele in enumerate(mcmc_outcomes):
                        if ele.item():
                            new_seq.append(starting_seq[i])
                        else:
                            new_seq.append(back_and_forth_seq[i])
                    back_and_forth_seq = new_seq
                R_BF.append(
                    controller.calc_log_reward(back_and_forth_seq).mean().item()
                )

            if args.mc_em:
                for mcmc_i in range(mcmc_steps):
                    starting_seq = [seq.clone() for seq in back_and_forth_seq]
                    starting_ll = controller.calc_log_reward(
                        starting_seq,
                        temp_cond=temp_cond,
                        ebm_reward_temp=get_ebm_temp(global_grammar_updates),
                    )
                    for i in range(len(back_and_forth_seq)):
                        if mcmc_i % 2 == 0:
                            back_and_forth_seq[i]._state[0].random_rotate()
                        else:
                            back_and_forth_seq[i]._state[0].random_change_symbol(
                                [
                                    x + ntokens_src
                                    for x in range(args.extra_nts - args.num_pts)
                                ]
                            )
                        back_and_forth_seq[i]._state[0].clear_ll()
                    back_and_forth_seq_ll = controller.calc_log_reward(
                        back_and_forth_seq,
                        temp_cond=temp_cond,
                        ebm_reward_temp=get_ebm_temp(global_grammar_updates),
                    )
                    mcmc_accept_prob = back_and_forth_seq_ll - starting_ll
                    mcmc_outcomes = (
                        mcmc_accept_prob
                        < torch.rand(starting_ll.shape, device=starting_ll.device).log()
                    )
                    new_seq = []
                    for i, ele in enumerate(mcmc_outcomes):
                        if ele.item():
                            new_seq.append(starting_seq[i])
                        else:
                            new_seq.append(back_and_forth_seq[i])
                    back_and_forth_seq = new_seq

            if args.sleep_mle and (temp_cond is None or temp_cond == 0):
                new_trees = [
                    seq[0].left
                    for seq in controller.grammar.generate_batch_q_fast(
                        bsz, controller.grammar.params_cache, max_steps=args.seqlen * 2
                    )
                ]
                for i in range(len(new_trees)):
                    new_trees[i].remove_pts()
                sleep_mle_seq = [state.from_tree(tree) for tree in new_trees]

                # =============================================================
                # ========= Unparse and optimize NLL of recovering ============
                # =============================================================
                # SMLE_logP_traj_BB = torch.zeros(bsz, device=device)
                SMLE_logP_traj_BF = torch.zeros(bsz, device=device)

                encoded_tokens, _ = model_encoder(
                    sleep_mle_seq, seq_type="all_root", temp_cond=temp_cond
                )
                # for i in range(random.randint(1, min(epoch+1, args.seqlen))):
                for i in range(args.seqlen):
                    B_logits = model_backward(
                        encoded_tokens, uniform_pos=args.uniform_pb
                    )
                    sample_backward_result = controller.sample_backward(
                        B_logits, sleep_mle_seq, greedy=False, temperature_pos=1.0
                    )
                    sleep_mle_seq = sample_backward_result["new_states"]
                    logP_B = sample_backward_result["policy_log_pb"]
                    F_actions = sample_backward_result["forward_actions"]
                    try:
                        encoded_tokens, _ = model_encoder(
                            sleep_mle_seq, seq_type="all_root", temp_cond=temp_cond
                        )
                    except:
                        SMLE_logP_traj_BF *= 0.0
                        break
                    F_logits = model_forward(encoded_tokens, uniform_pos=uniform_pos)
                    logP_F = controller.batch_calc_forward_prob(
                        F_logits, sleep_mle_seq, F_actions=F_actions
                    )[0]

                    # SMLE_logP_traj_BB += logP_B
                    SMLE_logP_traj_BF += logP_F

                sleep_mle_loss = (
                    args.smle_gamma * (-SMLE_logP_traj_BF).mean() / args.grad_acc
                )
                sleep_mle_loss.backward()

            F_db_loss_smoothed = (
                F_db_loss_smoothed * args.threshold_beta
                + F_db_loss.item() * (1 - args.threshold_beta)
            )
            tb_threshold = args.grammar_update_tb_threshold_max + (
                global_grammar_updates / args.grammar_update_tb_threshold_horizon
            ) * (
                args.grammar_update_tb_threshold_min
                - args.grammar_update_tb_threshold_max
            )
            if tb_threshold < args.grammar_update_tb_threshold_min:
                tb_threshold = args.grammar_update_tb_threshold_min
            F_db_loss_min = min(F_db_loss_smoothed, F_db_loss_min)
            update_grammar_next = (
                not (args.train_gfn) or F_db_loss_smoothed < tb_threshold
            )
            if (
                args.train_grammar
                and update_grammar
                and (temp_cond is None or temp_cond == 0)
            ):  # and sent_id == 1:
                total_PCFG_updates += 1
                global_grammar_updates += 1
                if args.parser_type == "gfn":
                    for seq in back_and_forth_seq:
                        for root in seq.trees:
                            root.clear_ll()
                    logP_z_x = torch.cat(
                        [
                            t.view(1)
                            for t in (
                                controller.grammar.compute_ll_batch(
                                    sum(
                                        [
                                            [root for root in seq.trees]
                                            for seq in back_and_forth_seq
                                        ],
                                        [],
                                    )
                                )
                            )
                        ],
                        0,
                    )
                    PCFG_loss = -logP_z_x
                else:
                    lengths = [len(s) for s in original_seqs]
                    batch_seq = torch.nn.utils.rnn.pad_sequence(
                        original_seqs, batch_first=True
                    )
                    if args.parser_type == "sample_from_posterior":
                        _, ll = controller.grammar.sample_batch(batch_seq, lengths)
                        PCFG_loss = -ll.mean()
                    elif args.parser_type == "marginalization" or (
                        args.parser_type == "gfn"
                        and epoch <= args.grammar_pretrain_epochs
                    ):
                        (
                            lps,
                            max_nll,
                            _,
                            spans,
                        ) = controller.grammar.compute_marginal_batch(
                            batch_seq, lengths, return_spans=False
                        )
                        PCFG_loss = -lps

                PCFG_loss = PCFG_loss.mean() / args.grad_acc
                PCFG_loss.backward()

                # if we hit a grammar_log_interval, validate and log
                if global_grammar_updates % args.grammar_log_interval == 0:
                    validate_and_log()

            total_PCFG_loss += (
                PCFG_loss.item() if type(PCFG_loss) is torch.Tensor else PCFG_loss
            )
            total_LLLB_loss += (
                LLLB_loss.item() if type(LLLB_loss) is torch.Tensor else LLLB_loss
            )
            total_F_db_loss += (
                F_db_loss.item() if type(F_db_loss) is torch.Tensor else F_db_loss
            )
            total_B_db_loss += (
                B_db_loss.item() if type(B_db_loss) is torch.Tensor else B_db_loss
            )
            total_BF_db_loss += (
                BF_db_loss.item() if type(BF_db_loss) is torch.Tensor else BF_db_loss
            )
            total_sleep_mle_loss += (
                sleep_mle_loss.item()
                if type(sleep_mle_loss) is torch.Tensor
                else sleep_mle_loss
            )

            epoch_loss += (
                LLLB_loss + B_db_loss + F_db_loss + BF_db_loss + total_sleep_mle_loss
            ).item()
            forward_steps_list.append(forward_steps)
            backward_steps_list.append(backward_steps)

        total_num += args.batch_size

        if args.clip > 0:
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if args.precision == "half":
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizers), args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(
                    chain(
                        model_encoder.parameters(),
                        model_flow.parameters(),
                        model_forward.parameters(),
                        model_backward.parameters(),
                    ),
                    args.clip,
                    error_if_nonfinite=True,
                )
                if args.train_grammar == 1:
                    torch.nn.utils.clip_grad_norm_(
                        controller.grammar.parameters(),
                        args.clip,
                        error_if_nonfinite=True,
                    )

        for i, opt, sched in zip(range(len(optimizers)), optimizers, schedulers):
            if args.train_gfn and i == 0:
                opt.step()
                opt.zero_grad()
                if sched is not None and global_step >= args.warmup:
                    sched.step()
            # do not act if the PCFG was not updated
            if args.train_grammar and update_grammar and i == len(optimizers) - 1:
                opt.step()
                opt.zero_grad()
                if sched is not None and global_grammar_updates >= args.warmup:
                    sched.step()
                controller.grammar.cache_params()
        # profiler.step()
        # if global_step > 6:
        #    profiler.export_stacks("./torch_trace/profiler_stacks_nopt.txt", "self_cpu_time_total")
        global_step += 1

        def get_lr():
            for g in optimizers[0].param_groups:
                return g["lr"]

        if sent_id % args.log_interval == 0:
            cur_PCFG_loss = (
                (total_PCFG_loss / total_PCFG_updates)
                if total_PCFG_updates > 0
                else float("nan")
            )
            cur_LLLB_loss = total_LLLB_loss / min(args.log_interval, sent_id)
            cur_B_db_loss = total_B_db_loss / min(args.log_interval, sent_id)
            cur_F_db_loss = total_F_db_loss / min(args.log_interval, sent_id)
            cur_BF_db_loss = total_BF_db_loss / min(args.log_interval, sent_id)
            cur_sleep_mle_loss = total_sleep_mle_loss / min(args.log_interval, sent_id)
            cur_PCFG_update_freq = total_PCFG_updates / min(args.log_interval, sent_id)
            elapsed = time.time() - start_time
            print(
                f"| epoch {epoch:3d} | {sent_id:5d}/{len(train_dataloader):5d} batches "
                + f"| lr {get_lr():.6f} | s/batch {elapsed / args.log_interval:5.2f} "
                + (f"| PCFG {cur_PCFG_loss:5.2f} " if args.train_grammar else "")
                + (f"| B-TB {cur_B_db_loss:5.2f} " if args.tjb_backward else "")
                + (f"| F-TB {cur_F_db_loss:5.2f} " if args.tjb_forward else "")
                + (
                    f"| BnF-TB {cur_BF_db_loss:5.2f} "
                    if args.go_back_and_forward
                    else ""
                )
                + (f"| SMLE {cur_sleep_mle_loss:5.2f} " if args.sleep_mle else "")
                + (
                    f"| Z {np.mean(Z) / args.reward_scale:4.2f} "
                    if args.tjb_forward or args.tjb_backward
                    else ""
                )
                + (
                    f"| R_F {np.mean(R_F) / args.reward_scale:4.2f} "
                    if args.tjb_forward
                    else ""
                )
                + (
                    f"| R_BnF {np.mean(R_BF) / args.reward_scale:4.2f} "
                    if args.go_back_and_forward
                    else ""
                )
                + (
                    f"| PCFG update freq {cur_PCFG_update_freq:1.2f} | Total PCFG update {global_grammar_updates}"
                )
                + (f"| min F_db {F_db_loss_min}")
                + f"| loss {cur_LLLB_loss+cur_B_db_loss+cur_F_db_loss+cur_BF_db_loss:5.2f}"
            )
            start_time = time.time()
            if first_loss is None:
                first_loss = (
                    cur_LLLB_loss + cur_B_db_loss + cur_F_db_loss + cur_sleep_mle_loss
                )
            wandb.log(
                dict(
                    step=global_step,
                    per_step_time=elapsed / args.log_interval,
                    per_step_F_DB=cur_F_db_loss,
                    per_step_BF_DB=cur_BF_db_loss,
                    per_step_sleep_mle=cur_sleep_mle_loss,
                    per_step_loss=cur_LLLB_loss
                    + cur_B_db_loss
                    + cur_F_db_loss
                    + cur_BF_db_loss
                    + cur_sleep_mle_loss,
                    per_grammar_loss=cur_PCFG_loss,
                    per_step_R_F=np.mean(R_F),
                    per_step_R_BnF=np.mean(R_BF),
                )
            )
            total_PCFG_loss = 0.0
            total_LLLB_loss = 0.0
            total_B_db_loss = 0.0
            total_F_db_loss = 0.0
            total_BF_db_loss = 0.0
            total_sleep_mle_loss = 0.0
            total_PCFG_updates = 0
            backward_steps_list = []
            forward_steps_list = []
            Z = []
            R_F = []
            R_BF = []

        update_grammar = update_grammar_next

        if global_step >= args.updates:
            return 0, 0
        if global_grammar_updates >= args.max_grammar_updates:
            return 0, 0
    return epoch_loss / (len(train_dataloader) - 1), total_num_correct / total_num


if args.save_dir is not None:
    os.makedirs(args.save_dir, exist_ok=True)

# Loop over epochs.
lr = args.lr
best_val_sample_sent_f1 = 0


optimizers = []
if args.grammar_optimizer is None:
    args.grammar_optimizer = args.optimizer


if args.train_gfn == 1:
    GFN_param_group = [
        {
            "params": model_encoder.parameters(),
            "lr": args.lr if args.lr_encoder is None else args.lr_encoder,
            "initial_lr": args.lr if args.lr_encoder is None else args.lr_encoder,
        },
        {
            "params": model_backward.parameters(),
            "lr": args.lr if args.lr_backward is None else args.lr_backward,
            "initial_lr": args.lr if args.lr_backward is None else args.lr_backward,
        },
        {
            "params": [
                p for n, p in model_forward.named_parameters() if "embedding" not in n
            ],
            "lr": args.lr if args.lr_forward is None else args.lr_forward,
            "initial_lr": args.lr if args.lr_forward is None else args.lr_forward,
        },
        {
            "params": model_flow.parameters(),
            "lr": args.lr if args.lr_flow is None else args.lr_flow,
            "initial_lr": args.lr if args.lr_flow is None else args.lr_flow,
        },
    ]
    if args.optimizer == "sgd":
        optimizers.append(
            mup.MuSGD(GFN_param_group, lr=args.lr, momentum=args.momentum)
        )
    elif args.optimizer == "adam":
        if args.precision == "half":
            warnings.warn("half precison training hasn't been testesd")
            optimizers.append(
                apex.optimizers.FusedAdam(
                    GFN_param_group,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.wd,
                )
            )
        else:
            optimizers.append(
                mup.MuAdamW(
                    GFN_param_group,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.wd,
                )
            )

if args.train_grammar == 1:
    grammar_param_group = [
        {
            "params": controller.grammar.parameters(),
            "lr": args.lr if args.lr_grammar is None else args.lr_grammar,
            "initial_lr": args.lr if args.lr_grammar is None else args.lr_grammar,
        }
    ]
    if args.grammar_optimizer == "sgd":
        optimizers.append(
            mup.MuSGD(
                grammar_param_group,
                lr=args.lr if args.lr_grammar is None else args.lr_grammar,
                momentum=args.momentum,
            )
        )
    elif args.grammar_optimizer == "adam":
        if args.adam_beta1_grammar is None:
            args.adam_beta1_grammar = args.adam_beta1
        if args.adam_beta2_grammar is None:
            args.adam_beta2_grammar = args.adam_beta2
        if args.precision == "half":
            optimizers.append(
                apex.optimizers.FusedAdam(
                    grammar_param_group,
                    betas=(args.adam_beta1_grammar, args.adam_beta2_grammar),
                    weight_decay=args.wd,
                )
            )
        else:
            optimizers.append(
                mup.MuAdamW(
                    grammar_param_group,
                    betas=(args.adam_beta1_grammar, args.adam_beta2_grammar),
                    weight_decay=args.wd,
                )
            )

# print num of trainable params
total_params = 0
for opt in optimizers:
    for g in opt.param_groups:
        for p in g["params"]:
            total_params += p.numel()
print(f"Grand Total of Trainable Params: {total_params}")

if args.scheduler_epochs == -1:
    max_updates = min(args.updates, args.epochs * len(train_dataloader))
else:
    max_updates = min(args.updates, args.scheduler_epochs * len(train_dataloader))

print(f"decaying LR over {max_updates} updates")
schedulers = []
for opt in optimizers:
    if args.schedule == "cosine":
        schedulers.append(
            optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(max_updates))
        )
    elif args.schedule == "inverse_sqrt":
        lambda_inverse_sqrt = lambda update: min(50 / (1 + update) ** 0.5, 1)
        schedulers.append(optim.lr_scheduler.LambdaLR(opt, lambda_inverse_sqrt))
    elif args.schedule == "linear":
        lambda_linear = lambda update: (1 - update / int(max_updates))
        schedulers.append(optim.lr_scheduler.LambdaLR(opt, lambda_linear))
    elif args.schedule == "constant":
        schedulers.append(None)
    else:
        assert False, f"unknown schedule"


# half-precision black magic
if args.precision == "half":
    (
        model_encoder,
        model_flow,
        model_forward,
        model_backward,
    ), optimizers = amp.initialize(
        [model_encoder, model_flow, model_forward, model_backward],
        optimizers,
        opt_level="O1",
        verbosity=1,
    )

start_epoch = 0
if args.resume_dir:
    if os.path.exists(os.path.join(args.resume_dir, "checkpoint_last.pt")):
        checkpoint = torch.load(os.path.join(args.resume_dir, "checkpoint_last.pt"))
    elif os.path.exists(os.path.join(args.resume_dir, exp_id, "checkpoint_last.pt")):
        checkpoint = torch.load(
            os.path.join(args.resume_dir, exp_id, "checkpoint_last.pt")
        )
    else:
        warnings.warn(
            "Checkpoint not found at --resume_dir. Starting from random init..."
        )
        checkpoint = None
    if checkpoint is not None:
        if args.restore_grammar_only:
            if args.grammar_type != "fixed":
                controller.grammar.load_state_dict(checkpoint["grammar"])
        else:
            model_shared_embedding.load_state_dict(checkpoint["model_shared_embedding"])
            model_encoder.load_state_dict(checkpoint["model_encoder"])
            if args.tjb_backward == 1 or args.tjb_forward >= 1:
                model_flow.load_state_dict(checkpoint["model_flow"])
            model_forward.load_state_dict(checkpoint["model_forward"])
            model_backward.load_state_dict(checkpoint["model_backward"])
            if not args.reset_lr_schedule:
                for i, sched in enumerate(schedulers):
                    if sched is not None:
                        sched.load_state_dict(checkpoint["schedulers"][i])
                for i, opt in enumerate(optimizers):
                    opt.load_state_dict(checkpoint["optimizers"][i])
                global_step = checkpoint["global_step"]
                start_epoch = checkpoint["epoch"]
            if args.precision == "half":
                amp.load_state_dict(checkpoint["amp"])
            if args.grammar_type != "fixed":
                controller.grammar.load_state_dict(checkpoint["grammar"])
            if "best_val_nll" in checkpoint:
                best_val_sample_sent_f1 = checkpoint["val_sample_sent_f1"]
            else:
                best_val_sample_sent_f1 = checkpoint["best_val_sample_sent_f1"]
            torch.set_rng_state(checkpoint["torch_rng"])
            np.random.set_state(checkpoint["numpy_rng"])
            random.setstate(checkpoint["python_rng"])

# At any point you can hit Ctrl + C to break out of training early.
train_losses = []
val_marginal_nlls = []
val_marginal_ppls = []
logZs = []
bnf = []
ftbs = []

# controller.grammar.cache_params()
# valid()
# validate_and_log()
# assert False

try:
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start_time = time.time()
        train_dataloader = train_dataloader[torch.randperm(len(train_dataloader))]
        if args.train_gfn == 1 or args.train_grammar == 1:
            train_loss, train_acc = train(
                epoch,
                optimizers,
                schedulers,
                uniform_pos=epoch < args.uniform_pos_until,
                max_len=args.seqlen
                - args.curriculum_len
                + args.curriculum_step * epoch,
            )
        else:
            train_loss, train_acc = 0, 0

        # validation
        val_ll_stats, val_lengths = valid()

        val_estimated_ll = val_ll_stats["estimated_ll"]
        val_sample_sent_f1 = val_ll_stats["sample_sent_f1"]
        val_sample_corpus_f1 = val_ll_stats["sample_corpus_f1"]
        val_sampler_nll = -val_ll_stats["sampler_ll"].sum().item() / sum(val_lengths)
        val_max_sent_f1 = val_ll_stats["max_sent_f1"]
        val_max_corpus_f1 = val_ll_stats["max_corpus_f1"]

        val_marginal_nll = -val_ll_stats["marginal_ll"].sum().item() / sum(val_lengths)
        val_pcfg_max_nll = -val_ll_stats["pcfg_max_ll"].sum().item() / sum(val_lengths)
        val_sampler_nll = -val_ll_stats["sampler_ll"].sum().item() / sum(val_lengths)
        val_r_pb_over_pf = val_ll_stats["r_pb_over_pf"].mean()

        wandb.log(
            {
                "epoch": epoch,
                "val_marginal_nll": val_marginal_nll,
                "val_sampler_nll": val_sampler_nll,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_r_pb_over_pf": val_r_pb_over_pf,
            }
        )

        # print the tokenizer
        if args.train_grammar and args.grammar_type == "cfg":
            print("PCFG:")
            print(controller.grammar.print(overall_tokenizer))
            print("--------------------------------")

        train_losses.append(train_loss)
        val_marginal_nlls.append(val_marginal_nll)
        val_marginal_ppls.append(np.exp(val_marginal_nll))

        print(
            f"| train | loss {train_loss: .3f} "
            + f"| valid | GFN-Sample-NLL {val_sampler_nll: 1.3f} "
            + (
                f"| Sent F1 (sample) {val_sample_sent_f1: 1.3f} | Corpus F1 (sample) {val_sample_corpus_f1: 1.3f}"
                if args.use_spans_f1
                else ""
            )
        )
        if args.grammar_type == "cfg":
            print(
                f"| valid | M_NLL {val_marginal_nll: 1.3f} | M_PPL {np.exp(val_marginal_nll): 1.3f} "
                f"| PCFG Max NLL: {val_pcfg_max_nll: 1.3f} "
                + (
                    f"| Sent F1 (max) {val_max_sent_f1: 1.3f} | Corpus F1 (max) {val_max_corpus_f1: 1.3f}"
                    if args.use_spans_f1
                    else ""
                )
            )

        if args.save_dir is not None:
            checkpoint = {
                "model_shared_embedding": model_shared_embedding.state_dict(),
                "model_encoder": model_encoder.state_dict(),
                "model_flow": model_flow.state_dict(),
                "model_forward": model_forward.state_dict(),
                "model_backward": model_backward.state_dict(),
                "optimizers": [opt.state_dict() for opt in optimizers],
                "schedulers": [
                    sched.state_dict() if args.schedule != "constant" else None
                    for sched in schedulers
                ],
                "epoch": epoch,
                "best_val_sample_sent_f1": max(
                    val_sample_sent_f1, best_val_sample_sent_f1
                ),
                "global_step": global_step,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "val_estimated_ll_list": val_estimated_ll,
            }
            if args.grammar_type != "fixed":
                checkpoint["grammar"] = controller.grammar.state_dict()
            if args.precision == "half":
                checkpoint["amp"] = amp.state_dict()
            if val_sample_sent_f1 > best_val_sample_sent_f1:
                best_val_sample_sent_f1 = val_sample_sent_f1
                torch.save(
                    checkpoint,
                    os.path.join(args.save_dir, exp_id, f"checkpoint_best.pt"),
                )
            torch.save(
                checkpoint, os.path.join(args.save_dir, exp_id, "checkpoint_last.pt")
            )
            if args.plot:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(2, 1)
                ax[0].plot(train_losses)
                ax[0].set_ylabel("train_loss")
                ax[1].plot(val_sample_sent_f1)
                ax[1].set_ylabel("val_F1")
                plt.savefig(os.path.join(args.save_dir, exp_id, "latest_plot.png"))
                print(
                    f"Saved plot to {os.path.join(args.save_dir, exp_id, 'latest_plot.png')}"
                )

        if global_step >= args.updates:
            break
        if args.backward_until_epochs > -1 and epoch >= args.backward_until_epochs:
            args.go_backward = 0

except KeyboardInterrupt:
    print("-" * 89)
    print("Exiting from training early")
