import torch
from pcfg_base import *
from nltk import read_grammar, Nonterminal
import numpy as np
import re
from torch_struct import SentCFG, SampledSemiring
from utils import *
import mup


class MLPPCFG(BaseNeuralPCFG):
    def __init__(
        self,
        num_nt,
        num_t,
        start,
        preterminal_mask,
        only_unary_from_s,
        embedding_dim,
        grammar_type,
    ):
        super(MLPPCFG, self).__init__(
            num_nt, num_t, start, preterminal_mask, only_unary_from_s
        )

        self.grammar_type = grammar_type
        self.ruletype_logits = torch.nn.Parameter(torch.randn((num_nt,)))
        self.emb_input = torch.nn.Parameter(
            torch.randn((num_nt + num_t, embedding_dim))
        )
        if self.grammar_type == "ncfg":
            self.emb_input_parent = torch.nn.Parameter(
                torch.randn(
                    (self.num_nt - self.preterminal_mask.sum().item(), embedding_dim)
                )
            )
        self.emb_output_unary = mup.MuReadout(
            embedding_dim, num_nt + num_t, readout_zero_init=False
        )
        self.emb_output_binary = mup.MuReadout(
            embedding_dim, num_nt**2, readout_zero_init=False
        )

        def make_mlp():
            return torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_dim, embedding_dim),
                torch.nn.ReLU(),
            )

        self.g11, self.g12 = make_mlp(), make_mlp()
        self.g21, self.g22 = make_mlp(), make_mlp()

        self.pts = self.preterminal_mask.sum().item()
        self.nts = self.num_nt - self.pts

    def get_context(self, node):
        if self.grammar_type == "ncfg":
            if node.parent is None:
                return self.num_t
            return node.parent.data
        return None

    def get_raw_ruletype_logit(self, nt):
        return self.ruletype_logits[nt]

    def get_raw_unary_rule_logits(self, nt, context=None):
        x = self.emb_input[nt]
        if context is not None:
            x = x + self.emb_input_parent[context]
        if nt == self.start - self.num_t:
            f1x = x + self.g22(x + self.g21(x))
        else:
            f1x = x + self.g12(x + self.g11(x))
        return self.emb_output_unary(f1x)

    def get_raw_binary_rule_logits(self, nt, context=None):
        x = self.emb_input[nt]
        if context is not None:
            x = x + self.emb_input_parent[context]
        return self.emb_output_binary(x).view(self.num_nt, self.num_nt)

    def cache_params(self):
        self.params_cache = [
            p[0]
            for p in self.get_params(
                torch.arange(self.num_t).unsqueeze(0).to(self.ruletype_logits.device)
            )
        ]

    def get_params(self, inp):
        if self.grammar_type == "ncfg":

            def terms(words):
                b, n = inp.shape[:2]
                term_logits = (
                    torch.cat(
                        [
                            self.get_unary_rule_logits(i, j).unsqueeze(0)
                            for i in range(self.nts, self.num_nt)
                            for j in range(self.nts)
                        ]
                    )
                    .view(self.pts, self.nts, -1)
                    .to(words.device)
                )
                rule_logits = torch.cat(
                    [
                        self.get_ruletype_logit(i).unsqueeze(0)
                        for i in range(self.nts, self.num_nt)
                    ]
                ).to(words.device)
                term_prob = (
                    (
                        term_logits[:, :, : self.num_t].log_softmax(-1)
                        + torch.nn.functional.logsigmoid(-1 * rule_logits)
                        .unsqueeze(1)
                        .unsqueeze(2)
                        .expand([self.pts, self.nts, self.num_t])
                    )
                    .unsqueeze(0)
                    .expand(b, self.pts, self.nts, self.num_t)
                    .permute(0, 3, 1, 2)
                )
                # indices = words.unsqueeze(2).unsqueeze(3).expand(b, n, self.pts, self.nts).unsqueeze(4)
                # term_prob = torch.gather(term_prob, 4, indices).squeeze(4)
                return term_prob

            def rules(b):
                # nonterm_emb = self.emb_input[1:self.nts]
                nonterm_logits = (
                    torch.cat(
                        [
                            self.get_binary_rule_logits(i, j).unsqueeze(0)
                            for i in range(1, self.nts)
                            for j in range(self.nts)
                        ]
                    )
                    .view(self.nts - 1, self.nts, -1)
                    .to(inp.device)
                )
                rule_logits = torch.cat(
                    [
                        self.get_ruletype_logit(i).unsqueeze(0)
                        for i in range(1, self.nts)
                    ]
                ).to(inp.device)
                return (
                    (
                        nonterm_logits.view(self.nts - 1, self.nts, -1).log_softmax(-1)
                        + torch.nn.functional.logsigmoid(rule_logits)
                        .unsqueeze(1)
                        .unsqueeze(2)
                        .expand([self.nts - 1, self.nts, self.num_nt**2])
                    )
                    .view(1, self.nts - 1, self.nts, self.num_nt, self.num_nt)
                    .expand(b, self.nts - 1, self.nts, self.num_nt, self.num_nt)[
                        :, :, :, 1:, 1:
                    ]
                )

            def roots(b):
                root_logits = (
                    self.get_unary_rule_logits(0, 0).unsqueeze(0).to(inp.device)
                )
                rule_logits = self.get_ruletype_logit(0).view(-1, 1).to(inp.device)
                return (
                    (
                        root_logits[
                            :, self.num_t + 1 : self.num_t + self.nts
                        ].log_softmax(-1)
                        + torch.nn.functional.logsigmoid(-1 * rule_logits)
                        .expand([1, self.nts - 1])
                        .to(self.ruletype_logits.device)
                    )
                    .view(1, self.nts - 1)
                    .expand(b, self.nts - 1)
                )

        elif self.grammar_type == "cfg":

            def terms(words):
                b, n = inp.shape[:2]
                term_logits = torch.cat(
                    [
                        self.get_unary_rule_logits(i).unsqueeze(0)
                        for i in range(self.nts, self.num_nt)
                    ]
                ).to(words.device)
                rule_logits = torch.cat(
                    [
                        self.get_ruletype_logit(i).unsqueeze(0)
                        for i in range(self.nts, self.num_nt)
                    ]
                ).to(words.device)
                term_prob = (
                    (
                        term_logits[:, : self.num_t].log_softmax(-1)
                        + torch.nn.functional.logsigmoid(-1 * rule_logits)
                        .unsqueeze(1)
                        .expand([self.pts, self.num_t])
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(b, n, self.pts, self.num_t)
                )
                indices = words.unsqueeze(2).expand(b, n, self.pts).unsqueeze(3)
                term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
                return term_prob

            def rules(b):
                # nonterm_emb = self.emb_input[1:self.nts]
                nonterm_logits = torch.cat(
                    [
                        self.get_binary_rule_logits(i).unsqueeze(0)
                        for i in range(1, self.nts)
                    ]
                ).to(inp.device)
                rule_logits = torch.cat(
                    [
                        self.get_ruletype_logit(i).unsqueeze(0)
                        for i in range(1, self.nts)
                    ]
                ).to(inp.device)
                return (
                    (
                        nonterm_logits.view(self.nts - 1, -1).log_softmax(-1)
                        + torch.nn.functional.logsigmoid(rule_logits)
                        .unsqueeze(1)
                        .expand([self.nts - 1, self.num_nt**2])
                    )
                    .view(1, self.nts - 1, self.num_nt, self.num_nt)
                    .expand(b, self.nts - 1, self.num_nt, self.num_nt)[:, :, 1:, 1:]
                )

            def roots(b):
                root_logits = self.get_unary_rule_logits(0).unsqueeze(0).to(inp.device)
                rule_logits = self.get_ruletype_logit(0).view(-1, 1).to(inp.device)
                return (
                    (
                        root_logits[
                            :, self.num_t + 1 : self.num_t + self.nts
                        ].log_softmax(-1)
                        + torch.nn.functional.logsigmoid(-1 * rule_logits)
                        .expand([1, self.nts - 1])
                        .to(self.ruletype_logits.device)
                    )
                    .view(1, self.nts - 1)
                    .expand(b, self.nts - 1)
                )

        return terms(inp), rules(inp.shape[0]), roots(inp.shape[0])

    def compute_marginal_batch(self, b_sentence, lengths, return_spans):
        if self.grammar_type == "ncfg":
            raise NotImplementedError("marginal doesn't work with noncf")
        else:
            # b_sentence: batch of sentences = [b, n]
            # lengths: length of sentences in batch = [b]
            params = self.get_params(b_sentence)
            # import pdb;pdb.set_trace();
            dist = SentCFG(params, lengths=lengths)
            argmax = dist.argmax
            spans = argmax[-1]
            tags = argmax[0]
            num_nts = self.num_nt - self.preterminal_mask.sum()
            if return_spans:
                nt_spans, _ = extract_parses(
                    spans, lengths, b_sentence, self.num_t, inc=1
                )
                for b in range(len(nt_spans)):
                    nt_spans[b].extend(
                        [
                            (i, 1, tags[b][i].argmax().item() + num_nts)
                            for i in range(lengths[b])
                        ]
                    )
            else:
                nt_spans = None
            # return dist.partition, dist.max, dist.entropy, nt_spans
            return dist.partition, dist.max, torch.Tensor([0]), nt_spans

    def compute_ll_batch(self, trees):
        # params = self.get_params(torch.arange(self.num_t).unsqueeze(0).to(self.ruletype_logits.device))
        # import pdb;pdb.set_trace()
        return [self.score_full_fast(t, self.params_cache) for t in trees]

    def sample_children_fast(self, node, params, context=None):
        assert node >= self.num_t

        type_logit = self.get_ruletype_logit(node - self.num_t)
        type = torch.bernoulli(type_logit.sigmoid()).item()
        type_ll = torch.nn.functional.logsigmoid(type_logit * (1 if type else -1))

        if type == 0:  # generate one child
            # t_logits = self.get_unary_rule_logits(node - self.num_t).log_softmax(0)
            if node == self.num_t:  # node is root
                t_logits = params[2]
            else:
                if self.grammar_type == "ncfg":
                    t_logits = params[0][
                        :, node - self.num_t - self.nts, context - self.num_t
                    ]
                else:
                    t_logits = params[0][:, node - self.num_t - self.nts]
            child = t_logits.exp().multinomial(1)
            child_ll = t_logits.gather(0, child).squeeze(0)

            if node == self.num_t:
                child += self.num_t + 1

            return [child.item()], type_ll + child_ll

        else:  # generate two children
            if self.grammar_type == "ncfg":
                nt_logits = params[1][
                    node - self.num_t - 1, context - self.num_t
                ].reshape(-1)
            else:
                nt_logits = params[1][node - self.num_t - 1].reshape(-1)
            children = nt_logits.exp().multinomial(1)
            children_ll = nt_logits.gather(0, children).squeeze(0)
            return [
                1 + self.num_t + children.item() // (self.num_nt - 1),
                1 + self.num_t + children.item() % (self.num_nt - 1),
            ], type_ll + children_ll

    def ll_children_fast(self, node, children, params, context=None):
        assert node >= self.num_t

        if len(children) == 0:
            return 0

        if len(children) == 1:
            if node == self.num_t:  # node is root
                return params[2][children[0] - self.num_t - 1]
            else:
                raise NotImplementedError  # this should not happen with marginalization over PTs or with gold tags!
        else:
            if self.grammar_type == "ncfg":
                rules = params[1][node - self.num_t - 1, context - self.num_t]
            else:
                rules = params[1][node - self.num_t - 1]

            if self.grammar_type == "ncfg":
                if children[0] > self.num_t and children[1] > self.num_t:
                    return rules[
                        children[0] - self.num_t - 1, children[1] - self.num_t - 1
                    ]
                elif children[0] > self.num_t and children[1] < self.num_t:
                    return (
                        rules[children[0] - self.num_t - 1, self.nts - 1 :]
                        + params[0][children[1], ..., node - self.num_t]
                    ).logsumexp(0)
                elif children[0] < self.num_t and children[1] > self.num_t:
                    return (
                        rules[self.nts - 1 :, children[1] - self.num_t - 1]
                        + params[0][children[0], ..., node - self.num_t]
                    ).logsumexp(0)
                elif children[0] < self.num_t and children[1] < self.num_t:
                    return (
                        rules[self.nts - 1 :, self.nts - 1 :]
                        + params[0][children[0], ..., node - self.num_t].unsqueeze(1)
                        + params[0][children[1], ..., node - self.num_t]
                    ).logsumexp((0, 1))
                else:  # root is a child :-(
                    return torch.tensor(-10000.0).to(rules.device)
            else:
                if children[0] > self.num_t and children[1] > self.num_t:
                    return rules[
                        children[0] - self.num_t - 1, children[1] - self.num_t - 1
                    ]
                elif children[0] > self.num_t and children[1] < self.num_t:
                    return (
                        rules[children[0] - self.num_t - 1, self.nts - 1 :]
                        + params[0][children[1]]
                    ).logsumexp(0)
                elif children[0] < self.num_t and children[1] > self.num_t:
                    return (
                        rules[self.nts - 1 :, children[1] - self.num_t - 1]
                        + params[0][children[0]]
                    ).logsumexp(0)
                elif children[0] < self.num_t and children[1] < self.num_t:
                    return (
                        rules[self.nts - 1 :, self.nts - 1 :]
                        + params[0][children[0]].unsqueeze(1)
                        + params[0][children[1]]
                    ).logsumexp((0, 1))
                else:  # root is a child :-(
                    return torch.tensor(-10000.0).to(rules.device)

    def score_one_fast(self, node, params):
        if node is None:
            return 0
        if node.ll is None:
            node.ll = self.ll_children_fast(
                node.data,
                [child.data for child in node.children()],
                params,
                self.get_context(node),
            )
        # old_ll = self.ll_children(node.data, [ child.data for child in node.children() ])
        # assert new_ll == old_ll:
        return node.ll

    def score_full_fast(self, node, params):
        ll = 0.0
        if node.data >= self.num_t:  # node is nonterminal
            ll += self.score_one_fast(node, params)
            for child in node.children():
                ll += self.score_full_fast(child, params)
        return ll

    def generate_batch_q_fast(self, n, params, max_steps=99999):
        res = []
        for _ in range(n):
            gen = SimpleTree(self.start)
            ll = self.expand_full_q_fast(gen, params, max_steps=max_steps)
            res.append((gen, ll))
        return res

    def expand_full_q_fast(self, node, params, max_steps=99999):
        ll = 0.0
        node_counter = 1
        to_process = [node]
        while len(to_process) > 0:
            cur_node = to_process[0]
            if cur_node.data >= self.num_t:  # node is nonterminal
                ll += self.expand_one_fast(cur_node, params)
                for child in cur_node.children():
                    to_process.append(child)
                    node_counter += 1
                node_counter -= 1
            to_process = to_process[1:]
            if node_counter >= max_steps:
                break
        return ll

    def expand_one_fast(self, node, params):
        children, ll = self.sample_children_fast(
            node.data, params, self.get_context(node)
        )
        node.hang(*[SimpleTree(child) for child in children])
        # ll2 = self.score_one_fast(node, params)
        # assert ll.item()==ll2.item()
        return ll

    def compute_marginal_slow(self, sentence):
        if self.grammar_type == "ncfg":
            raise NotImplementedError("torch-struct doesn't work with noncf")
        else:
            if self.preterminal_mask is None:
                raise NotImplementedError(
                    "Marginal computation only supported with preterminal mask."
                )

            sentence_len = len(sentence)
            beta = (
                torch.ones(
                    (sentence_len, sentence_len, self.num_nt),
                    device=self.ruletype_logits.device,
                )
                + LOGINF
            )
            for i in range(sentence_len):
                for r in range(self.num_nt):
                    idx = sentence[i]
                    idx = torch.LongTensor([idx]).to(self.ruletype_logits.device)
                    beta[i, 0, r] = torch.nn.functional.logsigmoid(
                        self.get_ruletype_logit(r) * -1
                    ) + self.get_unary_rule_logits(r).log_softmax(0).gather(
                        0, idx
                    ).squeeze(
                        0
                    )
            for j in range(1, sentence_len):
                for i in range(sentence_len - j):
                    for nt in range(self.num_nt):
                        acc = torch.tensor(0.0).to(self.ruletype_logits.device)
                        acc += beta[i, j, nt]
                        for l in range(j):
                            for k1 in range(self.num_nt):
                                for k2 in range(self.num_nt):
                                    idx = k1 * self.num_nt + k2
                                    idx = torch.LongTensor([idx]).to(
                                        self.ruletype_logits.device
                                    )
                                    rule_ll = torch.nn.functional.logsigmoid(
                                        self.get_ruletype_logit(nt)
                                    ) + self.get_binary_rule_logits(nt).view(
                                        -1
                                    ).log_softmax(
                                        0
                                    ).gather(
                                        0, idx
                                    ).squeeze(
                                        0
                                    )
                                    # beta[i, j, nt] = torch.logaddexp(beta[i, j, nt], rule_ll + beta[i, l, k1] + beta[i+l+1, j-l-1, k2])
                                    acc = torch.logaddexp(
                                        acc,
                                        rule_ll
                                        + beta[i, l, k1]
                                        + beta[i + l + 1, j - l - 1, k2],
                                    )
                        beta[i, j, nt] = acc

            if self.only_unary_from_s:
                logits_from_s = self.get_unary_rule_logits(
                    self.start - self.num_t
                ).log_softmax(0)[self.num_t :]
                return (logits_from_s + beta[0, sentence_len - 1]).logsumexp(0)
            else:
                return beta[0, sentence_len - 1, self.start - self.num_t]

    @torch.enable_grad()
    def sample_batch(self, b_sentence, lengths):
        if self.grammar_type == "ncfg":
            raise NotImplementedError("torch-struct doesn't work with noncf")
        else:
            params = self.get_params(b_sentence)
            dist = SentCFG(params, lengths=lengths)
            sample = dist._struct(SampledSemiring).marginals(
                dist.log_potentials, lengths=dist.lengths
            )

            terms, rules, roots = dist.log_potentials[:3]
            m_term, m_rule, m_root = (part.detach() for part in sample[:3])
            ll = (
                (terms * m_term).sum((1, 2))
                + (rules * m_rule).sum((1, 2, 3))
                + (roots * m_root).sum(1)
            )
            return sample, ll

    @torch.enable_grad()
    def maxll_batch(self, b_sentence, lengths):
        if self.grammar_type == "ncfg":
            raise NotImplementedError("torch-struct doesn't work with noncf")
        else:
            params = self.get_params(b_sentence)
            dist = SentCFG(params, lengths=lengths)
            return dist.argmax, dist.max


class TabularPCFG(BaseNeuralPCFG):
    def __init__(
        self, num_nt, num_t, start, preterminal_mask=None, only_unary_from_s=False
    ):
        super(TabularPCFG, self).__init__(
            num_nt, num_t, start, preterminal_mask, only_unary_from_s
        )

        self.ruletype_logits = torch.nn.Parameter(torch.randn((num_nt,)) * 1e-4)
        self.binary_rule_logits = torch.nn.Parameter(
            torch.randn((num_nt, num_nt, num_nt)) * 1e-4
        )
        self.unary_rule_logits = torch.nn.Parameter(
            torch.randn(num_nt, num_t + num_nt) * 1e-4
        )

    def get_raw_ruletype_logit(self, nt):
        return self.ruletype_logits[nt]

    def get_raw_unary_rule_logits(self, nt):
        return self.unary_rule_logits[nt]

    def get_raw_binary_rule_logits(self, nt):
        return self.binary_rule_logits[nt]

    def from_fixed(other: FixedPCFG, only_unary_from_s=False, enable_pts=False):
        this = TabularPCFG(
            other.num_nt, other.num_t, other.start, only_unary_from_s=only_unary_from_s
        )
        pts = np.zeros(other.num_nt)
        for nt in range(other.num_t, other.num_t + other.num_nt):
            rules = [a[1:] for a in other.lhs[nt]]

            unary_probs = torch.zeros((this.num_t + this.num_nt,))
            binary_probs = torch.zeros((this.num_nt, this.num_nt))
            is_pt = True
            for prod, p in rules:
                if len(prod) == 1:  # unary
                    unary_probs[prod[0]] = p
                    if prod[0] >= other.num_t:
                        is_pt = False
                elif len(prod) == 2:  # binary
                    binary_probs[prod[0] - this.num_t, prod[1] - this.num_t] = p
                    is_pt = False

            if is_pt:
                pts[nt] = 1

            p_binary = binary_probs.sum()
            if p_binary == 0:
                binary_rule_logit = LOGINF
            elif p_binary == 1:
                binary_rule_logit = -LOGINF
            else:
                binary_rule_logit = np.log(p_binary / (1 - p_binary))

            this.ruletype_logits.data[nt - this.num_t] = binary_rule_logit
            this.unary_rule_logits.data[nt - this.num_t] = (
                (unary_probs / (unary_probs.sum() + 1e-20)).log().clamp(min=LOGINF)
            )
            this.binary_rule_logits.data[nt - this.num_t] = (
                (binary_probs / (binary_probs.sum() + 1e-20)).log().clamp(min=LOGINF)
            )

        if enable_pts:
            pt_mask = np.array(pts, dtype=np.bool).reshape(other.num_nt, -1)
            this.preterminal_mask = pt_mask
            this.ruletype_logits.data[pt_mask] = -200  # preterminals only produce unary
            this.unary_rule_logits.data[
                pt_mask, other.num_nt :
            ] = -200  # ...and only generate terminals
            this.ruletype_logits.data[~pt_mask] = 200

        return this


_STANDARD_NONTERM_RE = re.compile(r"( [\w/][\w/^<>-]* ) \s*", re.VERBOSE)


def standard_nonterm_parser(string, pos):
    m = _STANDARD_NONTERM_RE.match(string, pos)
    if not m:
        raise ValueError("Expected a nonterminal, found: " + string[pos:])
    return (Nonterminal(m.group(1)), m.end())


def get_rules_syms(prods, term_vocab, start):
    nts = {}
    rules_ = {}
    nts[str(start)] = len(term_vocab)
    for prod in prods:
        lhs = str(prod.lhs())
        if lhs not in nts:
            nts[lhs] = len(term_vocab) + len(nts.keys())
        rhs = prod.rhs()
        tf_rhs = []
        for s in rhs:
            if isinstance(s, str):
                tf_rhs.append(term_vocab.index(s))
            else:
                if str(s) not in nts:
                    nts[str(s)] = len(term_vocab) + len(nts.keys())
                tf_rhs.append(nts[str(s)])
        if nts[lhs] not in rules_:
            rules_[nts[lhs]] = []
        rules_[nts[lhs]].append([nts[lhs], tf_rhs, prod.prob()])
    return rules_, list(nts.keys())


def get_pcfg(
    num,
    vocab_dict,
    p_type="fixed",
    extra_nts=0,
    only_unary_from_s=False,
    num_pts=0,
    mlp_dim=8,
    device="cpu",
    grammar_type="cfg",
):
    if num == 0:
        desc = None
        nts = []

    elif num == 1:
        desc = """
            S -> NP VP [1.0]
            NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]
            Det -> 'the' [0.8] | 'my' [0.2]
            N -> 'man' [0.5] | 'telescope' [0.5]
            VP -> VP PP [0.2] | V NP [0.8]
            V -> 'ate' [0.35] | 'saw' [0.65]
            PP -> P NP [1.0]
            P -> 'with' [0.61] | 'under' [0.39]
            """
    elif num == 2:
        desc = """
            A -> B C [1.0]
            B -> 'b' [1.0]
            C -> 'c' [1.0]
            """
    elif num == 3:
        desc = """
            A -> B C [0.5] | D E [0.5]
            B -> 'b' [1.0]
            C -> 'c' [1.0]
            D -> 'd' [1.0]
            E -> 'e' [1.0]
            """
    elif num == 4:
        desc = """
            S -> A   [1.0]
            A -> B C [1.0]
            B -> X B [0.5] | X X [0.5]
            C -> Y C [0.5] | Y Y [0.5]
            X -> 'x' [1.0]
            Y -> 'y' [1.0]
            """
    elif num == 5:
        desc = """
            S    -> NP VP         [1.0]
            VP   -> V NP          [.59]
            VP   -> V             [.40]
            VP   -> VP PP         [.01]
            NP   -> Det N         [.41]
            NP   -> Name          [.28]
            NP   -> NP PP         [.31]
            PP   -> P NP          [1.0]
            V    -> 'saw'         [.21]
            V    -> 'ate'         [.51]
            V    -> 'ran'         [.28]
            N    -> 'boy'         [.11]
            N    -> 'cookie'      [.12]
            N    -> 'table'       [.13]
            N    -> 'telescope'   [.14]
            N    -> 'hill'        [.5]
            Name -> 'Jack'        [.52]
            Name -> 'Bob'         [.48]
            P    -> 'with'        [.61]
            P    -> 'under'       [.39]
            Det  -> 'the'         [.41]
            Det  -> 'a'           [.31]
            Det  -> 'my'          [.28]
            """
    elif num == 6:
        from nltk.corpus import treebank
        from nltk import Nonterminal, ProbabilisticProduction
        import nltk

        nltk.download("treebank")
        productions = []
        S = Nonterminal("S")
        for t in treebank.fileids()[:2]:
            for x in treebank.parsed_sents(t):
                x.collapse_unary(collapsePOS=False)
                x.chomsky_normal_form(horzMarkov=2)
                productions += x.productions()
        pcount = {}

        # LHS-count: counts the number of times a given lhs occurs
        lcount = {}

        for prod in productions:
            lcount[prod.lhs()] = lcount.get(prod.lhs(), 0) + 1
            pcount[prod] = pcount.get(prod, 0) + 1

        prods = [
            ProbabilisticProduction(p.lhs(), p.rhs(), prob=pcount[p] / lcount[p.lhs()])
            for p in pcount
        ]

        rules, nts = get_rules_syms(prods, vocab_dict, S)
        return FixedPCFG(len(nts), len(vocab_dict), len(vocab_dict), rules), nts

    if desc is not None:
        start, productions = read_grammar(
            desc, standard_nonterm_parser, probabilistic=True
        )
        rules, nts = get_rules_syms(productions, vocab_dict, start)

    for i in range(extra_nts):
        nts.append(f"Q{i}")
    if p_type == "fixed":
        return FixedPCFG(len(nts), len(vocab_dict), len(vocab_dict), rules), nts, None
    elif p_type == "tabular_neural":
        pt_mask = None
        if num_pts > 0:
            pt_mask = np.concatenate(
                (np.zeros(len(nts) - num_pts), np.ones(num_pts))
            ).astype(np.bool)
        return (
            TabularPCFG(
                len(nts),
                len(vocab_dict),
                len(vocab_dict),
                only_unary_from_s=only_unary_from_s,
                preterminal_mask=pt_mask,
            ).to(device),
            nts,
            pt_mask,
        )
    elif p_type == "mlp_neural":
        pt_mask = None
        if num_pts > 0:
            pt_mask = np.concatenate(
                (np.zeros(len(nts) - num_pts), np.ones(num_pts))
            ).astype(np.bool)
        return (
            MLPPCFG(
                len(nts),
                len(vocab_dict),
                len(vocab_dict),
                pt_mask,
                only_unary_from_s,
                mlp_dim,
                grammar_type=grammar_type,
            ).to(device),
            nts,
            pt_mask,
        )
    elif p_type == "neural_from_fixed":
        fixed_pcfg = FixedPCFG(len(nts), len(vocab_dict), len(vocab_dict), rules)
        tabular_pcfg = TabularPCFG.from_fixed(
            fixed_pcfg, only_unary_from_s=only_unary_from_s, enable_pts=(num_pts == 0)
        )
        return tabular_pcfg, nts, tabular_pcfg.preterminal_mask
    else:
        raise NotImplementedError
