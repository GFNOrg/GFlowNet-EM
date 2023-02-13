import torch
import random
import numpy as np
from math import log
from scipy.special import logsumexp

LOGINF = -20


class SimpleTree:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.parent = None
        self.hang(left, right)
        self.ll = None
        self._emb = None
        self._ebm_emb = None

    def clone(self):
        new_tree = SimpleTree(
            self.data,
            self.left.clone() if self.left is not None else None,
            self.right.clone() if self.right is not None else None,
        )
        new_tree._emb = self._emb.clone() if self._emb is not None else None
        return new_tree

    def hang(self, left_child, right_child=None):
        self.left = left_child
        self.right = right_child
        if self.left is not None:
            self.left.parent = self
        if self.right is not None:
            self.right.parent = self

    def children(self):
        if self.left is not None:
            yield self.left
        if self.right is not None:
            yield self.right

    def print(self, tostr=str) -> str:
        data = self.data.item() if not isinstance(self.data, int) else self.data
        if self.left is not None:
            return f'{{{tostr(data)} {" ".join(map(lambda x: x.print(tostr), self.children()))}}}'
        else:
            return f"{tostr(data)}"

    def __repr__(self) -> str:
        return self.print()

    @property
    def leaf_seq(self):
        if self.is_terminal:
            return [self.data]
        elif self.is_full:
            return self.left.leaf_seq + self.right.leaf_seq
        else:
            return self.left.leaf_seq

    @property
    def height(self):
        return (
            max(
                0 if self.left is None else self.left.height,
                0 if self.right is None else self.right.height,
            )
            + 1
        )

    @property
    def is_full(self):
        return self.left is not None and self.right is not None

    @property
    def is_terminal(self):
        return self.left is None and self.right is None

    @property
    def is_preterminal(self):
        if self.right is not None:
            return False
        if self.left is not None and self.left.is_terminal:
            return True
        return False

    @property
    def has_single_nt_child(self):
        return not self.is_full and not self.is_terminal and not self.is_preterminal

    @property
    def count_left_branch_tree(self):
        if self.is_terminal:
            return 1
        if self.right is not None and not self.right.is_terminal:
            return 0
        return self.left.count_left_branch_tree + 1

    def all_spans(self, start):
        # terminals are not in the span
        if self.left is None and self.right is None:
            return [], 1

        states, length = [], 0
        if self.left is not None:
            states_left, length_left = self.left.all_spans(start)
            states += states_left
            length += length_left
        if self.right is not None:
            states_right, length_right = self.right.all_spans(start + length)
            states += states_right
            length += length_right

        states.append((start, length, self.data))
        return states, length

    def all_tags(self):
        tags = [self.data]
        if self.is_terminal:
            return []

        tags += self.left.all_tags()
        if self.right is not None:
            tags += self.right.all_tags()

        return tags

    def cache_size(self):
        self._size = 1 if self.is_terminal else 0
        for child in self.children():
            child.cache_size()
            self._size += child._size

    def random_rotate(self):
        self.cache_size()
        if self._size == 2:
            return
        self.random_rotate_rec()

    def random_rotate_rec(self):
        assert self._size > 2

        points = np.zeros((4,))

        points[0] = self.left._size - 2
        points[1] = self.right._size - 2
        points[2:] = 1 + np.clip(points[:2], None, 0)
        points[:2] = np.clip(points[:2], 0, None)

        sample = np.random.choice(
            np.array(
                4,
            ),
            p=points / points.sum(),
        )

        if sample == 0:
            self.left.random_rotate_rec()
        elif sample == 1:
            self.right.random_rotate_rec()
        elif sample == 2:
            new_left = self.left.left
            new_right = SimpleTree(self.data, self.left.right, self.right)
            self.data = self.left.data
            self.left = new_left
            self.right = new_right
            self.left.parent = self
            self.right.parent = self
        elif sample == 3:
            new_left = SimpleTree(self.data, self.left, self.right.left)
            self.data = self.right.data
            new_right = self.right.right
            self.left = new_left
            self.right = new_right
            self.left.parent = self
            self.right.parent = self

    def random_change_symbol(self, tagset):
        self.cache_size()
        if self._size == 1:
            return
        self.random_change_symbol_rec(tagset)

    def random_change_symbol_rec(self, tagset):
        assert self._size > 1

        points = np.zeros((3,))

        points[0] = self.left._size - 1
        points[1] = 1
        points[2] = self.right._size - 1

        sample = np.random.choice(
            np.array(
                3,
            ),
            p=points / points.sum(),
        )

        if sample == 0:
            self.left.random_change_symbol_rec(tagset)
        elif sample == 2:
            self.right.random_change_symbol_rec(tagset)
        elif sample == 1:
            self.data = np.random.choice(tagset)

    def get_emb(self, bin_agg, uni_agg, t_nt_emb, ignore_data=False, ebm_cache=False):
        # print(ignore_cache, self.data)
        if not ebm_cache and self._emb is not None:
            return self._emb
        elif ebm_cache and self._ebm_emb is not None:
            return self._ebm_emb
        index = 0 if ignore_data else self.data
        if self.is_terminal:
            emb = t_nt_emb[index]
        elif self.is_preterminal:
            emb = uni_agg(
                t_nt_emb[index],
                self.left.get_emb(
                    bin_agg,
                    uni_agg,
                    t_nt_emb,
                    ignore_data=ignore_data,
                    ebm_cache=ebm_cache,
                ),
            )
        else:
            emb = bin_agg(
                t_nt_emb[index],
                self.left.get_emb(
                    bin_agg,
                    uni_agg,
                    t_nt_emb,
                    ignore_data=ignore_data,
                    ebm_cache=ebm_cache,
                ),
                self.right.get_emb(
                    bin_agg,
                    uni_agg,
                    t_nt_emb,
                    ignore_data=ignore_data,
                    ebm_cache=ebm_cache,
                ),
            )
        if ebm_cache:
            self._ebm_emb = emb
        else:
            self._emb = emb
        return emb

    def reset_emb(self):
        self._emb = None
        if self.is_preterminal:
            self.left.reset_emb()
        elif not self.is_terminal:
            self.left.reset_emb()
            self.right.reset_emb()

    def detach_emb(self):
        self._emb = self._emb.detach()
        if self.is_preterminal:
            self.left.detach_emb()
        elif not self.is_terminal:
            self.left.detach_emb()
            self.right.detach_emb()

    def remove_pts(self):
        if self.left is not None and self.left.is_preterminal:
            self.left = self.left.left
        if self.right is not None and self.right.is_preterminal:
            self.right = self.right.left

        if self.left is not None:
            self.left.remove_pts()
        if self.right is not None:
            self.right.remove_pts()

    def clear_ll(self):
        if self.left is not None:
            self.left.clear_ll()
        if self.right is not None:
            self.right.clear_ll()
        self.ll = None


class FixedPCFG:
    def __init__(self, num_nt, num_t, start, rules):
        self.num_nt = num_nt
        self.num_t = num_t
        self.start = start
        # rules are lists [lhs, rhs, prob]
        self.lhs = rules

    def sample_children(self, node):
        assert node >= self.num_t

        filtered_rules = self.lhs[node]
        probabilities = [rule[2] for rule in filtered_rules]
        rule = random.choices(filtered_rules, weights=probabilities)[0]
        return rule[1], np.log(rule[2])

    def ll_children(self, node, children):
        assert node >= self.num_t
        is_nt_nt = len(children) == 2
        if not isinstance(node, int):
            node = node.item()
        for i in range(len(children)):
            if not isinstance(children[i], int):
                children[i] = children[i].item()
        filtered_rules = self.lhs[node]
        for rule in filtered_rules:
            if is_nt_nt and len(rule[1]) == 2:
                if rule[1][0] == children[0] and rule[1][1] == children[1]:
                    return np.log(rule[2])
            elif not is_nt_nt and len(rule[1]) == 1:
                if rule[1][0] == children[0]:
                    return np.log(rule[2])
        return LOGINF

    def expand_one(self, node):
        children, ll = self.sample_children(node.data)
        node.hang(*[SimpleTree(child) for child in children])
        return ll

    def score_one(self, node):
        return self.ll_children(node.data, [child.data for child in node.children()])

    def expand_full(self, node):
        ll = 0.0
        if node.data >= self.num_t:  # node is nonterminal
            ll += self.expand_one(node)
            for child in node.children():
                ll += self.expand_full(child)
        return ll

    def score_full(self, node):
        ll = 0.0
        if node.data >= self.num_t:  # node is nonterminal
            ll += self.score_one(node)
            for child in node.children():
                ll += self.score_full(child)
        return ll

    def generate_batch(self, n, seqlen=32):
        res = []
        for _ in range(n):
            gen = SimpleTree(self.start)
            ll = self.expand_full(gen)
            while len(gen.leaf_seq) > seqlen:
                gen = SimpleTree(self.start)
                ll = self.expand_full(gen)
            res.append((gen, ll))
        return res

    def compute_marginal(self, sentence):
        # sentence = list of terminals
        sentence_len = len(sentence)
        beta = np.zeros((sentence_len, sentence_len, self.num_nt)) + LOGINF
        # Prob CYK Algorithm
        for i in range(sentence_len):
            for key in self.lhs:
                for rule in self.lhs[key]:
                    if len(rule[1]) == 1 and rule[1][0] == sentence[i]:
                        beta[i, 0, rule[0] - self.num_t] = log(rule[2])
                    # print(prod)
        for j in range(1, sentence_len):
            for i in range(sentence_len - j):
                for l in range(j):
                    for key in self.lhs:
                        for rule in self.lhs[key]:
                            if len(rule[1]) == 2:
                                k = rule[0] - self.num_t
                                k1 = rule[1][0] - self.num_t
                                k2 = rule[1][1] - self.num_t
                                # print(rule, np.logaddexp(beta[i, j, k], log(rule[2]) + beta[i, l, k1] + beta[i+l+1, j-l-1, k2]))
                                beta[i, j, k] = np.logaddexp(
                                    beta[i, j, k],
                                    log(rule[2])
                                    + beta[i, l, k1]
                                    + beta[i + l + 1, j - l - 1, k2],
                                )
        # import pdb; pdb.set_trace();
        return beta[
            0, sentence_len - 1, 1
        ]  # 1 since we have an extra token for the root to make the root unary


class BaseNeuralPCFG(torch.nn.Module):
    def __init__(self, num_nt, num_t, start, preterminal_mask, only_unary_from_s):
        super(BaseNeuralPCFG, self).__init__()

        self.num_nt = num_nt
        self.num_t = num_t
        self.start = start

        assert start >= num_t

        assert (preterminal_mask is None) or (
            (preterminal_mask.dtype == np.bool)
            and (preterminal_mask.shape == (num_nt,))
            and (~preterminal_mask[start - num_t])
        )

        self.preterminal_mask = preterminal_mask
        self.only_unary_from_s = only_unary_from_s

    def get_ruletype_logit(self, nt):
        if (nt == self.start - self.num_t) and self.only_unary_from_s:
            return torch.tensor(-200.0)
        if self.preterminal_mask is not None:
            return (
                torch.tensor(-200.0)
                if self.preterminal_mask[nt]
                else torch.tensor(200.0)
            )
        return self.get_raw_ruletype_logit(nt)

    def get_unary_rule_logits(self, nt, context=None):
        x = self.get_raw_unary_rule_logits(nt, context)

        mask = torch.zeros_like(x).bool()

        # disallow generating start
        mask[self.start] = True

        # disallow generating special
        mask[:2] = True

        # if nt is preterminal, we generate only terminals
        if (self.preterminal_mask is not None) and self.preterminal_mask[nt]:
            mask[self.num_t :] = True

        # if nt is start and flag is set, we generate only nonterminals
        if (nt == self.start - self.num_t) and self.only_unary_from_s:
            mask[: self.num_t] = True
            if self.preterminal_mask is not None:
                mask[self.num_t :][self.preterminal_mask] = True

        # disallow generation of self
        mask[self.num_t + nt] = True

        return torch.where(mask, torch.full_like(x, -200), x)

    def get_binary_rule_logits(self, nt, context=None):
        x = self.get_raw_binary_rule_logits(nt, context)

        mask = torch.zeros_like(x).bool()

        mask[:, self.start - self.num_t] = True
        mask[self.start - self.num_t, :] = True

        return torch.where(mask, torch.full_like(x, -200), x)

    def sample_children(self, node, context=None):
        assert node >= self.num_t

        type_logit = self.get_ruletype_logit(node - self.num_t)
        type = torch.bernoulli(type_logit.sigmoid()).item()
        type_ll = torch.nn.functional.logsigmoid(type_logit * (1 if type else -1))

        if type == 0:  # generate one child
            t_logits = self.get_unary_rule_logits(
                node - self.num_t, context
            ).log_softmax(0)
            child = t_logits.exp().multinomial(1)
            child_ll = t_logits.gather(0, child).squeeze(0)

            return [child.item()], type_ll + child_ll

        else:  # generate two children
            nt_logits = (
                self.get_binary_rule_logits(node - self.num_t, context)
                .view(-1)
                .log_softmax(0)
            )
            children = nt_logits.exp().multinomial(1)
            children_ll = nt_logits.gather(0, children).squeeze(0)

            return [
                self.num_t + children.item() // self.num_nt,
                self.num_t + children.item() % self.num_nt,
            ], type_ll + children_ll

    def ll_children(self, node, children, context=None):
        assert node >= self.num_t

        if len(children) == 0:
            return 0

        type_logit = self.get_ruletype_logit(node - self.num_t)
        type = 1 if len(children) == 2 else 0
        type_ll = torch.nn.functional.logsigmoid(type_logit * (1 if type else -1))

        if type == 0:  # score one child
            t_logits = self.get_unary_rule_logits(
                node - self.num_t, context
            ).log_softmax(0)
            child = torch.LongTensor([children[0]]).to(t_logits.device)
            child_ll = t_logits.gather(0, child).squeeze(0)

            return type_ll + child_ll

        else:  # score two children
            nt_logits = (
                self.get_binary_rule_logits(node - self.num_t, context)
                .view(-1)
                .log_softmax(0)
            )
            idx = (children[0] - self.num_t) * self.num_nt + (children[1] - self.num_t)
            idx = torch.LongTensor([idx]).to(nt_logits.device)
            children_ll = nt_logits.gather(0, idx).squeeze(0)

            return type_ll + children_ll

    def expand_one(self, node):
        children, ll = self.sample_children(node.data, self.get_context(node))
        node.hang(*[SimpleTree(child) for child in children])
        return ll

    def score_one(self, node):
        return self.ll_children(
            node.data, [child.data for child in node.children()], self.get_context(node)
        )

    def expand_full(self, node):
        ll = 0.0
        if node.data >= self.num_t:  # node is nonterminal
            ll += self.expand_one(node)
            for child in node.children():
                ll += self.expand_full(child)
        return ll

    def expand_full_q(self, node, max_steps=99999):
        ll = 0.0
        node_counter = 1
        to_process = [node]
        while len(to_process) > 0:
            cur_node = to_process[0]
            if cur_node.data >= self.num_t:  # node is nonterminal
                ll += self.expand_one(cur_node)
                for child in cur_node.children():
                    to_process.append(child)
                    node_counter += 1
                node_counter -= 1
            to_process = to_process[1:]
            if node_counter >= max_steps:
                break
        return ll

    def expand_full_ll_only(self, node):
        ll = 0.0
        children = [node]
        while len(children) > 0:
            child = children[0]
            if child.data <= self.num_t:
                children = children[1:]
                continue
            new_children, cur_ll = self.sample_children(child.data)
            ll += cur_ll
            children = children[1:]
            if ll > -100:
                children += [SimpleTree(c) for c in new_children]
        return ll

    def score_full(self, node):
        ll = 0.0
        if node.data >= self.num_t:  # node is nonterminal
            ll += self.score_one(node)
            for child in node.children():
                ll += self.score_full(child)
        return ll

    def generate_batch(self, n, seqlen=32):
        res = []
        for _ in range(n):
            gen = SimpleTree(self.start)
            ll = self.expand_full(gen)
            while len(gen.leaf_seq) > seqlen:
                gen = SimpleTree(self.start)
                ll = self.expand_full(gen)
            res.append((gen, ll))
        return res

    def generate_batch_q(self, n, max_steps=99999):
        res = []
        for _ in range(n):
            gen = SimpleTree(self.start)
            ll = self.expand_full_q(gen, max_steps=max_steps)
            res.append((gen, ll))
        return res

    @torch.no_grad()
    def print(self, tostr=str, topk=5):
        nt_prods = []
        for nt in range(self.num_nt):
            nt_prod = f"{tostr(nt+self.num_t)} -> "
            type_p = self.get_ruletype_logit(nt).sigmoid()
            all_logits = torch.cat(
                [
                    (1 - type_p) * self.get_unary_rule_logits(nt).softmax(0),
                    type_p * self.get_binary_rule_logits(nt).view(-1).softmax(0),
                ],
                0,
            )
            all_logits /= all_logits.sum()
            topv, topi = all_logits.topk(topk, 0)
            rules = []
            for i, v in zip(topi, topv):
                if i < self.num_t + self.num_nt:
                    rules.append(f"{tostr(i.item())} [{np.round(v.item(), 2):2.2f}]")
                else:
                    i_bin = i.item() - (self.num_t + self.num_nt)
                    a, b = i_bin // self.num_nt, i_bin % self.num_nt
                    rules.append(
                        f"{tostr(a+self.num_t)} {tostr(b+self.num_t)} [{np.round(v.item(), 2):2.2f}]"
                    )
            nt_prod += " | ".join(rules)
            nt_prods.append(nt_prod)
        return "\n".join(nt_prods)

    def compute_marginal(self, sentence):
        if self.preterminal_mask is None:
            raise NotImplementedError(
                "Marginal computation only supported with preterminal mask."
            )

        sentence_len = len(sentence)
        beta = np.ones((sentence_len, sentence_len, self.num_nt)) + LOGINF
        for i in range(sentence_len):
            for r in range(self.num_nt):
                idx = sentence[i]
                idx = torch.LongTensor([idx]).to(self.ruletype_logits.device)
                beta[i, 0, r] = (
                    (
                        torch.nn.functional.logsigmoid(self.get_ruletype_logit(r) * -1)
                        + self.get_unary_rule_logits(r)
                        .log_softmax(0)
                        .gather(0, idx)
                        .squeeze(0)
                    )
                    .cpu()
                    .numpy()
                )
        for j in range(1, sentence_len):
            for i in range(sentence_len - j):
                for l in range(j):
                    for nt in range(self.num_nt):
                        for k1 in range(self.num_nt):
                            for k2 in range(self.num_nt):
                                idx = k1 * self.num_nt + k2
                                idx = torch.LongTensor([idx]).to(
                                    self.ruletype_logits.device
                                )
                                rule_ll = (
                                    (
                                        torch.nn.functional.logsigmoid(
                                            self.get_ruletype_logit(nt)
                                        )
                                        + self.get_binary_rule_logits(nt)
                                        .view(-1)
                                        .log_softmax(0)
                                        .gather(0, idx)
                                        .squeeze(0)
                                    )
                                    .cpu()
                                    .numpy()
                                )
                                beta[i, j, nt] = np.logaddexp(
                                    beta[i, j, nt],
                                    rule_ll
                                    + beta[i, l, k1]
                                    + beta[i + l + 1, j - l - 1, k2],
                                )
                        # for k in range(self.num_nt):
                        #     idx = k + self.num_t
                        #     idx = torch.LongTensor([idx]).to(self.ruletype_logits.device)
                        #     rule_ll = (torch.nn.functional.logsigmoid(self.get_ruletype_logit(nt) * -1) + \
                        #         self.get_unary_rule_logits(nt).log_softmax(0).gather(0, idx).squeeze(0)).cpu().numpy()
                        #     beta[i, j, nt] = np.logaddexp(beta[i, j, nt], rule_ll + beta[i, l, k])

        if self.only_unary_from_s:
            logits_from_s = (
                self.get_unary_rule_logits(self.start - self.num_t)
                .log_softmax(0)
                .cpu()
                .numpy()[self.num_t :]
            )
            return logsumexp(logits_from_s + beta[0, sentence_len - 1])
        else:
            return beta[0, sentence_len - 1, self.start - self.num_t]

    def compute_marginal_torch(self, sentence):
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
                ) + self.get_unary_rule_logits(r).log_softmax(0).gather(0, idx).squeeze(
                    0
                )
        for j in range(1, sentence_len):
            for i in range(sentence_len - j):
                for nt in range(self.num_nt):
                    acc = torch.tensor(0.0)
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

    def compute_marginal_batch(self, b_sentence):
        # b_sentence: batch of sentences = [b, n, vocab]
        pass
