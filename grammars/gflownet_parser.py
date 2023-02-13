# shared encoder
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import math
from my_transformer import TransformerEncoderLayer, TransformerEncoder, LayerNorm
from pcfg_base import SimpleTree
from pcfg_neural import get_pcfg

import mup


def create_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(-1)
    # print(seq_length)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
    return position_ids


class GFlowNet_Z(nn.Module):
    def __init__(self, d_model):
        nn.Module.__init__(self)
        self.to_flow = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            mup.MuReadout(d_model, 1, readout_zero_init=False),
        )

    def forward(self, x, pad_mask):
        x = self.to_flow(x).squeeze(-1)
        masked_x = (x.view(-1) * pad_mask.exp().view(-1)).view(x.size())
        pooled_x = masked_x.sum(1)  # / pad_mask.exp().sum(dim=-1).view(-1)
        return pooled_x


class GFlowNet_shared_embedding(nn.Module):
    def __init__(self, n_vocab, d_model, seqlen=128, n_nts=0, grammar_emb=None):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.seqlen = seqlen
        self.n_vocab = n_vocab
        self.n_nts = n_nts
        self.embedding_tgt = nn.Embedding(n_vocab + n_nts, d_model)
        self.embedding_pos = nn.Embedding(seqlen, d_model)
        self.grammar_emb = grammar_emb
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding_tgt.weight, mean=0.0, std=1 / 128**0.5)
        nn.init.normal_(self.embedding_pos.weight, mean=0.0, std=1 / 128**0.5)

    def forward(self, tgt, type="all_root"):
        if type == "all_root":
            tensor_tgt = torch.stack([x.all_root_seq for x in tgt], dim=0)
        elif type == "all_pair":
            tensor_tgt = torch.stack([x.all_pair_seq for x in tgt], dim=0)
        elif type == "splittable":
            tensor_tgt = torch.stack([x.splittable_seq for x in tgt], dim=0)
        else:
            raise NotImplementedError
        tensor_tgt = tensor_tgt[:, tensor_tgt.sum(dim=0) != 0]
        assert (
            tensor_tgt.max() < self.n_vocab + self.n_nts
        ), f"input contains token id {tensor_tgt.max()}, but max is {self.n_vocab+self.n_nts-1}"
        encoded_tgt = self.embedding_tgt(tensor_tgt)
        if self.grammar_emb is not None:
            with torch.no_grad():
                encoded_tgt += F.embedding(tensor_tgt, self.grammar_emb)
        pos_ids = create_position_ids(tensor_tgt)
        encoded_tgt += self.embedding_pos(pos_ids)
        tgt_key_padding_mask = encoded_tgt.new_zeros(tensor_tgt.shape)
        tgt_key_padding_mask[tensor_tgt == 0] = -float("inf")
        return encoded_tgt, tgt_key_padding_mask


from tree_agg import (
    trivial_unary_agg,
    trivial_binary_agg,
    skipmlp_unary_agg,
    skipmlp_binary_agg,
    simplemlp_binary_agg,
    transformer_unary_agg,
    transformer_binary_agg,
)


class GFlowNet_shared_embedding_with_aggregation(nn.Module):
    def __init__(self, n_vocab, d_model, seqlen=128, n_nts=0, agg_type="trivial"):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.seqlen = seqlen
        self.n_vocab = n_vocab
        self.n_nts = n_nts
        self.embedding_tgt = nn.Embedding(n_vocab + n_nts, d_model)
        self.embedding_pos = nn.Embedding(seqlen, d_model)
        if agg_type == "trivial":
            unary_agg, binary_agg = trivial_unary_agg, trivial_binary_agg
        elif agg_type == "skipmlp":
            unary_agg, binary_agg = skipmlp_unary_agg, skipmlp_binary_agg
        elif agg_type == "simplemlp":
            unary_agg = None
            binary_agg = simplemlp_binary_agg
        elif agg_type == "transformer":
            unary_agg, binary_agg = transformer_unary_agg, transformer_binary_agg
        if unary_agg is not None:
            self.uni_agg = unary_agg(
                d_model=d_model,
                nhead=4,
                dim_feedforward=4 * d_model,
                dropout=0,
                norm_first=True,
                nlayers=2,
            )
        else:
            self.uni_agg = lambda x, y: y
        self.bin_agg = binary_agg(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=0,
            norm_first=True,
            nlayers=2,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(
            self.embedding_tgt.weight, mean=0.0, std=1 / self.d_model**0.5
        )
        nn.init.normal_(
            self.embedding_pos.weight, mean=0.0, std=1 / self.d_model**0.5
        )

    def forward(self, tgt, type="all_root"):
        if type == "all_root":
            tensor_tgt = torch.stack([x.all_root_seq for x in tgt], dim=0)
            encoded_tgt = [
                x.get_all_root_emb_seq(
                    self.bin_agg, self.uni_agg, self.embedding_tgt.weight
                )
                for x in tgt
            ]
        elif type == "splittable":
            tensor_tgt = torch.stack([x.splittable_seq for x in tgt], dim=0)
            encoded_tgt = [
                x.get_splittable_emb_seq(
                    self.bin_agg, self.uni_agg, self.embedding_tgt.weight
                )
                for x in tgt
            ]
        else:
            raise NotImplementedError
        tensor_tgt = tensor_tgt[:, tensor_tgt.sum(dim=0) != 0]
        encoded_tgt = torch.nn.utils.rnn.pad_sequence(encoded_tgt, batch_first=True)
        pos_ids = create_position_ids(tensor_tgt)
        encoded_tgt += self.embedding_pos(pos_ids)
        tgt_key_padding_mask = encoded_tgt.new_zeros(tensor_tgt.shape)
        tgt_key_padding_mask[tensor_tgt == 0] = -float("inf")
        return encoded_tgt, tgt_key_padding_mask


class GFlowNet_encoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        norm_first,
        nlayers,
        n_nts=0,
        seqlen=128,
        batch_first=True,
        activation="relu",
        shared_embedding=None,
    ):
        nn.Module.__init__(self)
        self.d_model = d_model
        if shared_embedding is None:
            self.embedding = GFlowNet_shared_embedding(
                n_vocab, d_model, seqlen, n_nts=n_nts
            )
        else:
            self.embedding = shared_embedding
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            activation=activation,
            norm_first=norm_first,
        )
        encoder_norm = LayerNorm(d_model, eps=1e-5)
        self.model_encoder = TransformerEncoder(encoder_layer, nlayers, encoder_norm)

    def forward(self, src, seq_type="all_root", temp_cond=None):
        encoded_src, src_key_padding_mask = self.embedding(src, type=seq_type)
        memory = self.model_encoder(
            encoded_src, src_key_padding_mask=src_key_padding_mask, temp_cond=temp_cond
        )
        return memory, src_key_padding_mask


class GFlowNet_forward(nn.Module):
    def __init__(
        self,
        n_nts,
        d_model,
        shared_embedding=None,
        tie_tgt_embedding=False,
        preterminal_mask=None,
    ):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.n_nts = n_nts
        if shared_embedding is None:
            pass
        else:
            self.embedding = shared_embedding
        self.to_pos = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            mup.MuReadout(d_model, 2, readout_zero_init=False),
        )  # turning into a real token or left-inserting smtg
        self.tie_tgt_embedding = tie_tgt_embedding
        if tie_tgt_embedding:
            self.to_tok = nn.LayerNorm(d_model)
        else:
            self.to_tok = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                mup.MuReadout(d_model, n_nts, readout_zero_init=False),
            )
        self.t_to_nt_head = mup.MuReadout(
            d_model, n_nts, bias=True, readout_zero_init=False
        )
        self.preterminal_mask = preterminal_mask

    def forward(self, x, uniform_pos=False, t_to_nt=False):
        if t_to_nt:
            if self.preterminal_mask is not None:
                logits = self.t_to_nt_head(x)
                mask = logits.new_zeros(logits.shape[-1])
                mask[~self.preterminal_mask] = -100
                return (None, logits + mask)
            else:
                return (None, self.t_to_nt_head(x))
        if uniform_pos:
            pos_logits = x.new_zeros((x.size(0), x.size(1) * 2))
        else:
            pos_logits = self.to_pos(x).view(x.size(0), -1)
        tok_logits = self.to_tok(x)
        if self.tie_tgt_embedding:
            tok_logits = (
                tok_logits @ self.embedding.embedding_tgt.weight.T[:, -self.n_nts :]
            )
        if self.preterminal_mask is not None:
            mask = tok_logits.new_zeros(tok_logits.shape[-1])
            mask[self.preterminal_mask] = float("-inf")
            tok_logits = tok_logits + mask
        return (pos_logits, tok_logits)


class GFlowNet_backward(nn.Module):
    def __init__(self, d_model, n_nts, n_vocab):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.to_pos = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            mup.MuReadout(d_model, 1, readout_zero_init=False),
        )  # turning into a placeholder or being deleted
        self.to_tok = mup.MuReadout(n_nts, n_vocab, readout_zero_init=False)

    def forward(self, x, uniform_pos=False, nt_to_t=False):
        if not nt_to_t:
            if uniform_pos:
                pos_logits = x.new_zeros(x.shape[:-1])
            else:
                pos_logits = self.to_pos(x).squeeze(-1)
            return (pos_logits,)
        tok_logits = self.to_tok(x)
        return (tok_logits,)


class GFlowNet_dummy(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)


from gflownet import GFN_controller, GFN_state


class tree_state(GFN_state):
    def __init__(self, *args, **kwargs):
        GFN_state.__init__(self, *args, **kwargs)
        self.seqlen = self.args["seqlen"]
        self.n_vocab = self.args["n_vocab"]
        self.n_nts = self.args["n_nts"]
        self.start_sym = self.args["start_sym"]
        self._all_root_seq = None
        self._state = []

    def clone(self, keep_state=True):
        # create a new tree_state object and copy attributes one by one
        new_state = tree_state(device=self.device, args=self.args)
        new_state.seqlen = self.seqlen
        new_state.n_vocab = self.n_vocab
        new_state.n_nts = self.n_nts
        new_state.start_sym = self.start_sym
        if keep_state:
            new_state._state = [s.clone() for s in self._state]
        return new_state

    def from_iterable(self, iterable):
        assert len(iterable.shape) == 1
        iterable = iterable[iterable != 0]
        new_state = self.clone(keep_state=False)
        new_state.seqlen = self.seqlen if self.seqlen is not None else len(iterable)
        new_state._state = [SimpleTree(ele.item()) for ele in iterable]
        assert len(new_state.all_root_seq) == self.seqlen
        self._all_root_seq = None
        return new_state

    def from_tree(self, node):
        new_state = self.clone(keep_state=False)
        new_state._state = [node]
        assert len(new_state.all_root_seq) == self.seqlen
        self._all_root_seq = None
        return new_state

    def link_self(self, pos, ele):
        # assert not self.is_terminated
        parent = SimpleTree(ele)
        parent.hang(self._state[pos], None)
        self._state[pos] = parent
        self._all_root_seq = None
        return self

    def link_next(self, pos, ele):
        assert not self.is_terminated
        parent = SimpleTree(ele)
        parent.hang(self._state[pos], self._state[pos + 1])
        self._state[pos] = parent
        self._state = self._state[: pos + 1] + self._state[pos + 2 :]
        self._all_root_seq = None
        return self

    def split(self, pos):
        self._state = (
            self._state[:pos]
            + (
                [self._state[pos].left, self._state[pos].right]
                if self._state[pos].right is not None
                else [
                    self._state[pos].left,
                ]
            )
            + self._state[pos + 1 :]
        )
        self._all_root_seq = None
        return self

    def keep_only_root(self):
        for root in self._state:
            root.left = None
            root.right = None
        self._all_root_seq = None
        return self

    def s1_to_s0(self):
        assert self.is_s1
        self._state = [nt.left for nt in self._state]
        assert self.is_s0
        self._all_root_seq = None
        return self

    @property
    def trees(self):
        if len(self) > 1:
            return [subtree for subtree in self._state]
        return [SimpleTree(self.start_sym, left=self._state[0])]

    @property
    def leaf_seq(self):
        return [x for subtree in self._state for x in subtree.leaf_seq]

    @property
    def splittable_mask(self):
        assert len(self) <= self.seqlen
        return (
            torch.Tensor(
                [x.is_full for x in self._state] + [0] * (self.seqlen - len(self))
            )
            .bool()
            .to(self.device)
        )

    @property
    def all_root_seq(self):
        assert len(self) <= self.seqlen
        if self._all_root_seq is None:
            self._all_root_seq = torch.tensor(
                [x.data for x in self._state] + [0] * (self.seqlen - len(self)),
                dtype=torch.long,
                device=self.device,
            )
        return self._all_root_seq

    @property
    def all_pair_seq(self):
        assert len(self) <= self.seqlen
        if self.is_terminated:
            return self.all_root_seq
        return torch.tensor(
            [x.data for x in self._state][:-1] + [0] * (self.seqlen - len(self) + 1),
            dtype=torch.long,
            device=self.device,
        )

    @property
    def splittable_seq(self):
        assert len(self) <= self.seqlen
        nonterminal_nodes = [x.data for x in self._state if x.is_full]
        return torch.tensor(
            nonterminal_nodes + [0] * (self.seqlen - len(nonterminal_nodes)),
            dtype=torch.long,
            device=self.device,
        )

    def get_all_root_emb_seq(self, bin_agg, uni_agg, t_nt_emb):
        return torch.stack(
            [x.get_emb(bin_agg, uni_agg, t_nt_emb) for x in self._state], dim=0
        )

    def get_splittable_emb_seq(self, bin_agg, uni_agg, t_nt_emb):
        return torch.stack(
            [x.get_emb(bin_agg, uni_agg, t_nt_emb) for x in self._state if x.is_full],
            dim=0,
        )

    @property
    def is_terminated(self):
        return (
            len(self._state) == 1 and self._state[0].data >= self.n_vocab
        )  # and self._state[0].data == self.n_vocab

    @property
    def is_s1(self):
        return all([node.is_preterminal for node in self._state])

    @property
    def is_s0(self):
        return all([node.is_terminal for node in self._state])

    def __len__(self):
        return len(self._state)

    def __eq__(self, state):
        return len(self._state) == len(state._state)

    def __repr__(self) -> str:
        return self._state.__repr__()


def harden_mask(mask):
    return mask * 100


class parser_controller(GFN_controller):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.grammar, self.nts_list, self.pt_mask = get_pcfg(
            self.args["fixed_grammar_num"],
            vocab_dict=args["vocab_dict"],
            p_type=args["grammar_param"],
            extra_nts=args["extra_nts"],
            only_unary_from_s=args["only_unary_from_s"],
            num_pts=args["num_pts"],
            mlp_dim=args["mlp_dim"],
            grammar_type=args["grammar_type"],
            device=device,
        )
        self.args["n_nts"] = len(self.nts_list)
        # load the ebm if we use ebm_reward
        self.ebm = None
        if self.args["ebm_reward"] is not None:
            from tree_agg import aggregated_embedding, TreeModel

            aggregator = aggregated_embedding(
                1, self.args["ebm_d_model"], n_nts=0, agg_type=self.args["ebm_agg_type"]
            ).to(device)
            self.ebm = TreeModel(
                aggregator=aggregator, d_model=self.args["ebm_d_model"], device=device
            ).to(device)
            self.ebm.load_state_dict(torch.load(self.args["ebm_reward"]))

    def sample_forward(
        self,
        F_logits: tuple,
        states: list,
        greedy: bool = False,
        temperature_pos: float = 1.0,
        temperature_tok: float = 1.0,
        uniform_eps_pos: float = 0.0,
        uniform_eps_tok: float = 0.0,
        batch_token_dropout_mask=None,
    ):
        (F_pos_logits, F_tok_logits) = F_logits
        if F_pos_logits is not None:
            assert len(F_pos_logits.shape) == 2  # b x seq*2
        assert len(F_tok_logits.shape) == 3  # b x seq x n_nts
        assert type(greedy) is bool
        assert F_tok_logits.shape[2] == self.args["n_nts"]
        result_states = []
        B_pos_list = []
        if random.random() < uniform_eps_pos:
            temperature_pos = 99
        if random.random() < uniform_eps_tok:
            temperature_tok = 99

        forward_logP, true_logP, F_actions = self.batch_calc_forward_prob(
            F_logits,
            states,
            greedy=greedy,
            temperature_pos=temperature_pos,
            temperature_tok=temperature_tok,
            batch_token_dropout_mask=batch_token_dropout_mask,
        )

        reward_term = torch.zeros((len(states),), device=forward_logP.device)
        for i in range(len(states)):
            state = states[i]
            # has the trajectory already terminated?
            if state.is_terminated:
                reward_term[i] = 0.0
                result_states.append(state)
                B_pos_list.append(-1)
                continue
            # ==========================================
            # ====== Now, deal with other actions ======
            # ==========================================
            (B_pos,) = self.reverse_forward_actions(
                state, (F_actions[0][i], F_actions[1][i])
            )
            new_state, reward_term[i] = self.apply_forward_actions(
                state, (F_actions[0][i], F_actions[1][i])
            )
            result_states.append(new_state)
            B_pos_list.append(B_pos)
        B_pos_tensor = torch.tensor(B_pos_list, dtype=torch.long, device=self.device)
        return {
            "new_states": result_states,
            "policy_log_pf": forward_logP,
            "true_log_pf": true_logP,
            "backward_actions": (B_pos_tensor,),
            "reward_term": reward_term,
        }

    def batch_calc_forward_prob(
        self,
        F_logits: tuple,
        states: list,
        F_actions: tuple = None,
        greedy: bool = False,
        temperature_pos: float = 1.0,
        temperature_tok: float = 1.0,
        batch_token_dropout_mask=None,
    ):
        (F_pos_logits, F_tok_logits) = F_logits
        assert len(F_pos_logits.shape) == 2  # b x seq*2
        assert len(F_tok_logits.shape) == 3  # b x seq x ntokens
        # build the necessary masks
        stacked_states = torch.stack([x.all_root_seq for x in states], dim=0)
        # 1) mask out columns that are fully padded
        stacked_states = stacked_states[:, : F_pos_logits.size(1) // 2]
        # 2) pad mask
        F_pos_logits = F_pos_logits.view(
            F_pos_logits.size(0), F_pos_logits.size(1) // 2, 2
        )[:, :, 1]
        pad_mask = torch.cat(
            [
                (stacked_states == 0)[:, 1:],
                torch.ones((len(states), 1), dtype=torch.bool, device=self.device),
            ],
            dim=-1,
        )
        # 3) build the action type mask and do softmax over possible action locations
        action_mask = F_pos_logits.new_zeros(F_pos_logits.shape)
        action_mask[
            pad_mask
        ] = (
            -75
        )  # float('inf') # if the sequence has no more phs -inf could cause numerical issues
        logP_pos = torch.nn.functional.log_softmax((F_pos_logits + action_mask), dim=-1)
        # mask out S
        token_mask = F_tok_logits.new_zeros(F_tok_logits.shape)
        token_mask[:, :, self.grammar.start - self.grammar.num_t] += -75
        # 4) generate a probability distribution over tokens
        logP_tok = torch.nn.functional.log_softmax(F_tok_logits + token_mask, dim=-1)
        # true log prob
        true_logP_tok = torch.nn.functional.log_softmax(
            F_tok_logits / temperature_tok
            + harden_mask(token_mask)
            + (
                harden_mask(-batch_token_dropout_mask).unsqueeze(1)
                if batch_token_dropout_mask is not None
                else 0
            ),
            dim=-1,
        )
        true_logP_pos = torch.nn.functional.log_softmax(
            (F_pos_logits / temperature_pos + harden_mask(action_mask)), dim=-1
        )
        # 5) sample the actions if they aren't provided
        if F_actions is None:
            with torch.no_grad():
                P_tok_hard = torch.nn.functional.softmax(
                    F_tok_logits / temperature_tok
                    + harden_mask(token_mask)
                    + (
                        harden_mask(-batch_token_dropout_mask).unsqueeze(1)
                        if batch_token_dropout_mask is not None
                        else 0
                    ),
                    dim=-1,
                )
                P_pos_hard = torch.nn.functional.softmax(
                    (F_pos_logits / temperature_pos + harden_mask(action_mask)), dim=-1
                )
                if greedy:
                    F_pos_tensor = torch.argmax(
                        P_pos_hard, dim=-1, keepdim=True
                    ).squeeze(-1)
                    F_loc_tensor = F_pos_tensor
                    F_tok_tensor = torch.argmax(
                        P_tok_hard, dim=-1, keepdim=True
                    ).squeeze(-1)[torch.arange(P_tok_hard.shape[0]), F_loc_tensor]
                else:
                    F_pos_tensor = torch.multinomial(P_pos_hard, 1).squeeze(-1)
                    F_loc_tensor = F_pos_tensor
                    F_tok_tensor = torch.multinomial(
                        P_tok_hard.view(-1, P_tok_hard.size(-1)), 1
                    ).view(P_tok_hard.shape[:2])[
                        torch.arange(P_tok_hard.shape[0]), F_loc_tensor
                    ]
                # deal with termination
                is_terminated_tensor = torch.tensor(
                    [x.is_terminated for x in states],
                    dtype=torch.bool,
                    device=self.device,
                )
                F_pos_tensor = torch.where(is_terminated_tensor, -1, F_pos_tensor)
        else:
            F_pos_tensor = F_actions[0]
            F_tok_tensor = F_actions[1]
            F_loc_tensor = F_pos_tensor

        F_logP = logP_pos[torch.arange(logP_pos.size(0)), F_pos_tensor]
        F_logP += logP_tok[torch.arange(logP_tok.size(0)), F_loc_tensor, F_tok_tensor]
        true_F_logP = true_logP_pos[torch.arange(logP_pos.size(0)), F_pos_tensor]
        true_F_logP += true_logP_tok[
            torch.arange(logP_tok.size(0)), F_loc_tensor, F_tok_tensor
        ]
        # deal with termination
        F_logP = torch.where(F_pos_tensor == -1, torch.zeros_like(F_logP), F_logP)
        true_F_logP = torch.where(
            F_pos_tensor == -1, torch.zeros_like(true_F_logP), true_F_logP
        )
        if F_actions is None:
            return F_logP, true_F_logP, (F_pos_tensor, F_tok_tensor)
        else:
            return F_logP, true_F_logP

    @torch.no_grad()
    def apply_forward_actions(self, state: GFN_state, F_action: tuple):
        (F_pos, F_tok) = F_action
        F_pos = F_pos.item() if F_pos is not None else None
        if F_pos == -2:
            # T -> NT
            for i, _ in enumerate(state._state):
                new_parent = SimpleTree(F_tok[i].item() + self.args["n_vocab"])
                new_parent.hang(state._state[i])
                state._state[i] = new_parent
                state._all_root_seq = None
            return state, 0.0
            # TODO: when we start to tag, this should also call score_one_fast
        if F_pos == -1:
            assert state.is_terminated or state.is_s1
            return state, 0.0
        assert F_tok.item() < self.args["n_nts"]
        F_tok = F_tok.item()
        assert F_pos < len(state) - 1
        new_state = state.link_next(F_pos, F_tok + self.args["n_vocab"])
        if self.args["grammar_type"] == "cfg":
            rule_ll = self.grammar.score_one_fast(
                new_state._state[F_pos], self.grammar.params_cache
            )
        elif self.args["grammar_type"] == "ncfg":
            rule_ll = self.grammar.score_one_fast(
                new_state._state[F_pos].left, self.grammar.params_cache
            ) + self.grammar.score_one_fast(
                new_state._state[F_pos].right, self.grammar.params_cache
            )
        return new_state, rule_ll

    @torch.no_grad()
    def reverse_forward_actions(self, state: GFN_state, F_action: tuple):
        (F_pos, _) = F_action
        return (F_pos,)

    def sample_backward(
        self,
        B_logits: tuple,
        states: list,
        greedy: bool = False,
        temperature_pos: float = 1.0,
    ):
        assert type(greedy) is bool
        result_states = []
        backward_logP, true_logP, B_actions = self.batch_calc_backward_prob(
            B_logits, states, greedy=greedy, temperature_pos=temperature_pos
        )
        F_pos_tensor, F_tok_tensor = self.batch_reverse_backward_actions(
            states, B_actions[0]
        )
        for i in range(len(states)):
            state = states[i]
            if state.is_s0:
                result_states.append(state)
                continue
            new_state = self.apply_backward_actions(state, (B_actions[0][i],))
            result_states.append(new_state)
        assert len(backward_logP.shape) == 1
        return {
            "new_states": result_states,
            "policy_log_pb": backward_logP,
            "true_log_pb": true_logP,
            "forward_actions": (F_pos_tensor, F_tok_tensor),
        }

    def batch_calc_backward_prob(
        self,
        B_logits: tuple,
        states: list,
        B_actions: tuple = None,
        greedy: bool = False,
        temperature_pos: float = 1.0,
    ):
        (B_pos_logits,) = B_logits
        assert len(B_pos_logits.shape) == 2  # b x seq
        # build the necessary masks
        stacked_states = torch.stack([x.all_root_seq for x in states], dim=0)
        # 1) mask out columns that are fully padded
        column_pad_mask = stacked_states.sum(dim=0) != 0
        stacked_states = stacked_states[:, column_pad_mask]
        # 2) mask out padding and terminals
        pad_mask = torch.stack([x.splittable_mask for x in states], dim=0)
        pad_mask = pad_mask[:, column_pad_mask]
        # 3) build the action type mask and do softmax over possible action locations
        action_mask = B_pos_logits.new_zeros(B_pos_logits.shape)
        action_mask[~pad_mask] = -75
        logP_pos = torch.nn.functional.log_softmax(
            (B_pos_logits + action_mask), dim=-1
        )  # seqlen x two actions
        # true log prob
        true_logP_pos = torch.nn.functional.log_softmax(
            (B_pos_logits / temperature_pos + harden_mask(action_mask)), dim=-1
        )
        # 4) sample the actions if they aren't provided
        if B_actions is None:
            with torch.no_grad():
                P_pos_hard = torch.nn.functional.softmax(
                    (B_pos_logits / temperature_pos + harden_mask(action_mask)), dim=-1
                )  # seqlen x two actions
                if greedy:
                    B_pos_tensor = torch.argmax(P_pos_hard, dim=-1)
                else:
                    B_pos_tensor = torch.multinomial(P_pos_hard, 1).squeeze(-1)
                # deal with termination and near termination
                is_s0_tensor = torch.tensor(
                    [x.is_s0 for x in states], dtype=torch.bool, device=self.device
                )
                B_pos_tensor = torch.where(is_s0_tensor, -1, B_pos_tensor)
        else:
            B_pos_tensor = B_actions[0]
        B_logP = logP_pos[torch.arange(logP_pos.size(0)), B_pos_tensor]
        true_B_logP = true_logP_pos[torch.arange(true_logP_pos.size(0)), B_pos_tensor]
        # deal with termination
        B_logP = torch.where(B_pos_tensor == -1, torch.zeros_like(B_logP), B_logP)
        true_B_logP = torch.where(
            B_pos_tensor == -1, torch.zeros_like(true_B_logP), true_B_logP
        )
        if B_actions is None:
            return B_logP, true_B_logP, (B_pos_tensor,)
        else:
            return B_logP, true_B_logP

    @torch.no_grad()
    def apply_backward_actions(self, state: GFN_state, B_action: tuple):
        (B_pos,) = B_action
        if B_pos == -1:
            assert state.is_terminated or state.is_s0
            return state
        new_state = state.split(B_pos)
        return new_state

    @torch.no_grad()
    def reverse_backward_actions(self, state: GFN_state, B_action: tuple):
        (B_pos,) = B_action
        P_pos = B_pos
        return (P_pos, state.all_root_seq[B_pos] - self.args["n_vocab"])

    @torch.no_grad()
    def batch_reverse_backward_actions(
        self, states: GFN_state, B_actions: torch.Tensor
    ):
        F_pos_tensor = B_actions
        F_tok_tensor = torch.nn.utils.rnn.pad_sequence(
            [state.all_root_seq for state in states], batch_first=True
        )[torch.arange(F_pos_tensor.size(0)), F_pos_tensor]
        is_s0_tensor = torch.tensor(
            [x.is_s0 for x in states], dtype=torch.bool, device=self.device
        )
        F_pos_tensor = torch.where(is_s0_tensor, -1, F_pos_tensor)
        F_tok_tensor = torch.where(
            is_s0_tensor, -1, F_tok_tensor - self.args["n_vocab"]
        )
        return (F_pos_tensor, F_tok_tensor)

    @torch.no_grad()
    def calc_reward(self, to_calc):
        return self.calc_log_reward(to_calc).exp()

    @torch.no_grad()
    def calc_log_reward(self, to_calc, temp_cond=None, ebm_reward_temp=999999.0):
        new_ll = torch.Tensor(
            self.grammar.compute_ll_batch([s.trees[0] for s in to_calc])
        ).to(to_calc[0].device)
        # ebm stuff
        if self.ebm is not None:
            ebm_energy = self.ebm([s.trees[0] for s in to_calc]) / ebm_reward_temp
            new_ll -= ebm_energy.squeeze(-1)  # negate and add the energy

        if temp_cond is None:
            return new_ll
        return new_ll / math.exp(temp_cond)

    @torch.no_grad()
    def calc_partial_log_reward(self, to_calc, temp_cond=None):
        trees = sum([s.trees for s in to_calc], [])
        lengths = [len(s.trees) for s in to_calc]
        new_ll = torch.Tensor(self.grammar.compute_ll_batch(trees)).to(
            to_calc[0].device
        )
        cumlengths = [0] + list(torch.LongTensor(lengths).cumsum(0).numpy())
        new_ll = torch.Tensor(
            [new_ll[i:j].sum() for i, j in zip(cumlengths[:-1], cumlengths[1:])]
        ).to(to_calc[0].device)
        return new_ll / math.exp(temp_cond if temp_cond is not None else 0.0)
