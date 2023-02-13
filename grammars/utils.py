# Norm!/usr/bin/env python3
import numpy as np
import itertools
import random
import torch
import nltk
from pcfg_base import SimpleTree


def get_tree_from_binary_matrix(matrix, length):
    sent = list(map(str, range(length)))
    n = len(sent)
    tree = {}
    for i in range(n):
        tree[i] = sent[i]
    for k in np.arange(1, n):
        for s in np.arange(n):
            t = s + k
            if t > n - 1:
                break
            if matrix[s][t].item() == 1:
                span = "(" + tree[s] + " " + tree[t] + ")"
                tree[s] = span
                tree[t] = span
    return tree[0]


def get_nonbinary_spans(actions, SHIFT=0, REDUCE=1):
    spans = []
    stack = []
    pointer = 0
    binary_actions = []
    nonbinary_actions = []
    num_shift = 0
    num_reduce = 0
    for action in actions:
        # print(action, stack)
        if action == "SHIFT":
            nonbinary_actions.append(SHIFT)
            stack.append((pointer, pointer))
            pointer += 1
            binary_actions.append(SHIFT)
            num_shift += 1
        elif action[:3] == "NT(":
            stack.append("(")
        elif action == "REDUCE":
            nonbinary_actions.append(REDUCE)
            right = stack.pop()
            left = right
            n = 1
            while stack[-1] is not "(":
                left = stack.pop()
                n += 1
            span = (left[0], right[1])
            if left[0] != right[1]:
                spans.append(span)
            stack.pop()
            stack.append(span)
            while n > 1:
                n -= 1
                binary_actions.append(REDUCE)
                num_reduce += 1
        else:
            assert False
    assert len(stack) == 1
    assert num_shift == num_reduce + 1
    return spans, binary_actions, nonbinary_actions


def get_nonbinary_tree(sent, tags, actions):
    pointer = 0
    tree = []
    for action in actions:
        if action[:2] == "NT":
            node_label = action[:-1].split("NT")[1]
            node_label = node_label.split("-")[0]
            tree.append(node_label)
        elif action == "REDUCE":
            tree.append(")")
        elif action == "SHIFT":
            leaf = "(" + tags[pointer] + " " + sent[pointer] + ")"
            pointer += 1
            tree.append(leaf)
        else:
            assert False
    assert pointer == len(sent)
    return " ".join(tree).replace(" )", ")")


def build_tree(depth, sen):
    assert len(depth) == len(sen)

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = np.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1 :]) > 0:
            tree2 = build_tree(depth[idx_max + 1 :], sen[idx_max + 1 :])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1


def get_nonbinary_spans_label(actions, SHIFT=0, REDUCE=1):
    spans = []
    stack = []
    pointer = 0
    binary_actions = []
    num_shift = 0
    num_reduce = 0
    for action in actions:
        # print(action, stack)
        if action == "SHIFT":
            stack.append((pointer, pointer))
            pointer += 1
            binary_actions.append(SHIFT)
            num_shift += 1
        elif action[:3] == "NT(":
            label = "(" + action.split("(")[1][:-1]
            stack.append(label)
        elif action == "REDUCE":
            right = stack.pop()
            left = right
            n = 1
            while stack[-1][0] is not "(":
                left = stack.pop()
                n += 1
            span = (left[0], right[1], stack[-1][1:])
            if left[0] != right[1]:
                spans.append(span)
            stack.pop()
            stack.append(span)
            while n > 1:
                n -= 1
                binary_actions.append(REDUCE)
                num_reduce += 1
        else:
            assert False
    assert len(stack) == 1
    assert num_shift == num_reduce + 1
    return spans, binary_actions


def extract_parse(span, length, inc=1):
    tree = [(i, str(i)) for i in range(length)]
    tree = dict(tree)
    spans = []
    N = span.shape[0]
    cover = span.nonzero()
    # assert cover.shape[0] == N * 2 - 1, \
    #    f"Invalid parses: {length} spans at level 0:\n{span[0]} {cover.shape} != {N * 2 - 1}"
    try:
        fake_me = False
        for i in range(cover.shape[0]):
            if i >= N * 2 - 1:
                break
            w, r, A = cover[i].tolist()
            w = w + inc
            r = r + w
            l = r - w
            spans.append((l, r - l + 1, A))
            if l != r:
                span = "({} {})".format(tree[l], tree[r])
                tree[r] = tree[l] = span
    except Exception as e:
        fake_me = True
        warnings.warn(f"unparsable because `{e}`.")
    if fake_me or cover.shape[0] > N * 2 - 1:
        spans = [(l, length - 1, 0) for l in range(0, length - 1)]
        tree = dict([(i, str(i)) for i in range(length)])
        spans.reverse()
        for l, r, _ in spans:
            tree[r] = tree[l] = "({} {})".format(tree[l], tree[r])
    return spans, tree[0]


def extract_parses(matrix, lengths, seqs, n_vocab, inc=1):
    batch = matrix.shape[0]
    spans = []
    trees = []
    # import pdb; pdb.set_trace();
    for b in range(batch):
        span, tree = extract_parse(matrix[b], lengths[b], inc=inc)
        # print(get_tree(get_actions(tree), [str(i) for i in seqs[b].cpu().numpy().tolist()]))
        # trees.append(build_tree(span, seqs[b], n_vocab))
        spans.append(span)
    return spans, trees


def get_tree(actions, sent=None, SHIFT=0, REDUCE=1):
    # input action and sent (lists), e.g. S S R S S R R, A B C D
    # output tree ((A B) (C D))
    stack = []
    pointer = 0
    if sent is None:
        sent = list(map(str, range((len(actions) + 1) // 2)))
    #  assert(len(actions) == 2*len(sent) - 1)
    for action in actions:
        if action == SHIFT:
            word = sent[pointer]
            stack.append(word)
            pointer += 1
        elif action == REDUCE:
            right = stack.pop()
            left = stack.pop()
            stack.append("(" + left + " " + right + ")")
    assert len(stack) == 1
    return stack[-1]


def get_actions(tree, SHIFT=0, REDUCE=1, OPEN="(", CLOSE=")"):
    # input tree in bracket form: ((A B) (C D))
    # output action sequence: S S R S S R R
    actions = []
    tree = tree.strip()
    i = 0
    num_shift = 0
    num_reduce = 0
    left = 0
    right = 0
    while i < len(tree):
        if tree[i] != " " and tree[i] != OPEN and tree[i] != CLOSE:  # terminal
            if tree[i - 1] == OPEN or tree[i - 1] == " ":
                actions.append(SHIFT)
                num_shift += 1
        elif tree[i] == CLOSE:
            actions.append(REDUCE)
            num_reduce += 1
            right += 1
        elif tree[i] == OPEN:
            left += 1
        i += 1
    assert num_shift == num_reduce + 1
    return actions


def build_tree(spans, seq, n_vocab):
    spans = sorted(list(spans), key=lambda x: x[1])
    print(spans)

    # max length span is the root
    root = spans[-1]
    node = SimpleTree(root[2] + n_vocab)

    if len(spans) == 1:  # preterminal; hang the terminal and return
        node.hang(SimpleTree(seq[root[0]]))
        return node
    import pdb

    pdb.set_trace()
    # find the children
    # one of the children is the longest span that has the same start
    left = list(filter(lambda span: span[0] == root[0], spans[:-1]))[-1]
    # the other child is the longest span that starts after the left child ends
    right_start = root[0] + left[1]
    right = list(filter(lambda span: span[0] == right_start, spans[:-1]))[-1]
    left_node = build_tree(
        filter(
            lambda span: span[0] >= root[0] and span[0] + span[1] <= right_start, spans
        )
    )
    right_node = build_tree(
        filter(
            lambda span: span[0] >= right_start
            and span[0] + span[1] <= root[0] + root[1],
            spans,
        )
    )

    node.hang(left_node, right_node)

    return node


def build_parse(spans, caption):
    tree = [[i, word, 0, 0] for i, word in enumerate(caption)]
    for l, r in spans:
        if l != r:
            tree[l][2] += 1
            tree[r][3] += 1
    new_tree = [
        "".join(["("] * nl) + word + "".join([")"] * nr) for i, word, nl, nr in tree
    ]
    return " ".join(new_tree)


def get_tree(actions, sent=None, SHIFT=0, REDUCE=1):
    # input action and sent (lists), e.g. S S R S S R R, A B C D
    # output tree ((A B) (C D))
    stack = []
    pointer = 0
    if sent is None:
        sent = list(map(str, range((len(actions) + 1) // 2)))
    # assert(len(actions) == 2 * len(sent) - 1)
    for action in actions:
        if action == SHIFT:
            word = sent[pointer]
            stack.append(word)
            pointer += 1
        elif action == REDUCE:
            right = stack.pop()
            left = stack.pop()
            stack.append("(" + left + " " + right + ")")
    assert len(stack) == 1
    return stack[-1]


def get_actions(tree, SHIFT=0, REDUCE=1, OPEN="(", CLOSE=")"):
    # input tree in bracket form: ((A B) (C D))
    # output action sequence: S S R S S R R
    actions = []
    tree = tree.strip()
    i = 0
    num_shift = 0
    num_reduce = 0
    left = 0
    right = 0
    while i < len(tree):
        if tree[i] != " " and tree[i] != OPEN and tree[i] != CLOSE:  # terminal
            if tree[i - 1] == OPEN or tree[i - 1] == " ":
                actions.append(SHIFT)
                num_shift += 1
        elif tree[i] == CLOSE:
            actions.append(REDUCE)
            num_reduce += 1
            right += 1
        elif tree[i] == OPEN:
            left += 1
        i += 1
    assert num_shift == num_reduce + 1
    return actions
