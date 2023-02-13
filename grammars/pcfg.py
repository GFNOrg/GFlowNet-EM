import random
import numpy as np
from nltk import PCFG
from nltk.grammar import is_terminal

# from gflownet_parser import Node
from math import log

LOGINF = -20


class Node(object):
    def __init__(self, ele, args):
        self.ele = ele
        self.left = None
        self.right = None
        self.parent = None
        self.vocab_dict = args["vocab_dict"]
        self.nt_dict = args["nt_dict"]

    def hang(self, left_child, right_child=None):
        self.left = left_child
        self.right = right_child
        left_child.parent = self
        if right_child is not None:
            right_child.parent = self

    def nodes(self):
        yield self
        if self.left is not None:
            for node in self.left.nodes():
                yield node
        if self.right is not None:
            for node in self.right.nodes():
                yield node

    @property
    def leaf_seq(self):
        if self.is_terminal:
            return [self.ele]
        elif self.is_full:
            return self.left.leaf_seq + self.right.leaf_seq
        else:
            return self.left.leaf_seq

    def _convert(self, ele):
        if ele > len(self.vocab_dict) - 1:
            return self.nt_dict[ele - len(self.vocab_dict)]
        return self.vocab_dict[ele]

    def __repr__(self) -> str:
        if self.is_full:
            return f"{{{self._convert(self.ele)} {self.left} {self.right}}}"
        elif self.is_terminal:
            return f"{self._convert(self.ele)}"
        else:
            return f"{{{self._convert(self.ele)} {self.left}}}"

    @property
    def is_root(self):
        return self.parent is None

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
        return (
            not self.is_terminal
            and (True if self.left is None else self.left.is_terminal)
            and (True if self.right is None else self.right.is_terminal)
        )

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


class PCFGExtended(PCFG):
    def __init__(self, start, productions):
        PCFG.__init__(self, start, productions, calculate_leftcorners=True)
        self._calculate_str_indices()
        self.vocab_dict = None

    def _calculate_str_indices(self):
        self._lhs_index_str = {}
        self._rhs0_index_str = {}
        self._rhs1_index_str = {}
        self._empty_index_str = {}
        self._lexical_index_str = {}
        _symbols = []
        _nt_symbols = []
        for prod in self._productions:
            # Left hand side.
            lhs = str(prod._lhs)
            _symbols.append(lhs)
            _nt_symbols.append(lhs)
            if lhs not in self._lhs_index_str:
                self._lhs_index_str[lhs] = []
            self._lhs_index_str[lhs].append(prod)
            if prod._rhs:
                # First item in right hand side.
                rhs0 = str(prod._rhs[0])
                _symbols.append(rhs0)
                if rhs0 not in self._rhs0_index_str.keys():
                    self._rhs0_index_str[rhs0] = []
                self._rhs0_index_str[rhs0].append(prod)
                if len(prod._rhs) > 1:
                    rhs1 = str(prod._rhs[1])
                    _symbols.append(rhs1)
                    if rhs1 not in self._rhs1_index_str.keys():
                        self._rhs1_index_str[rhs1] = []
                    self._rhs1_index_str[rhs1].append(prod)
            else:
                # The right hand side is empty.
                self._empty_index_str[str(prod.lhs())] = prod
            # Lexical tokens in the right hand side.
            for token in prod._rhs:
                if is_terminal(token):
                    self._lexical_index_str.setdefault(token, set()).add(prod)
        self.symbols = set(_symbols)
        self.nt_symbols = set(_nt_symbols)
        self.nt_symbols_list = sorted(list(self.nt_symbols))
        self.nt_index = {}
        for i, sym in enumerate(self.nt_symbols_list):
            self.nt_index[sym] = i

    def _get_exact_prod(self, lhs, rhs):
        if len(rhs) == 1:
            match_prod = [
                prod
                for prod in self._lhs_index_str.get(lhs, [])
                if prod in self._rhs0_index_str.get(rhs[0], [])
                and len(prod.rhs()) == len(rhs)
            ]
        else:
            match_prod = [
                prod
                for prod in self._lhs_index_str.get(lhs, [])
                if prod in self._rhs0_index_str.get(rhs[0], [])
                and prod in self._rhs1_index_str.get(rhs[1], [])
            ]
        return match_prod[0] if len(match_prod) == 1 else None

    def generate_tree(self, n, seqlen=32):
        res = []
        for _ in range(n):
            gen = self.gen_tree_recursive(self.start())
            while len(gen.leaf_seq) > seqlen:
                gen = self.gen_tree_recursive(self.start())
            res.append(gen)
        return res

    def _get_id(self, tok):
        if tok in self.vocab_dict:
            return self.vocab_dict.index(tok)
        else:
            return len(self.vocab_dict) + self.nt_index[tok]

    def gen_tree_recursive(self, sym):
        curr_node = Node(
            self._get_id(str(sym)),
            args={"vocab_dict": self.vocab_dict, "nt_dict": self.nt_symbols_list},
        )
        if isinstance(sym, str):
            # print(curr_node.ele, curr_node.left, curr_node.right)
            return curr_node
        symbols, lls = self._reduce_once(sym)
        if len(symbols) == 1:
            curr_node.left = self.gen_tree_recursive(symbols[0])
            # print(curr_node.ele, curr_node.left.ele)
        else:
            curr_node.left = self.gen_tree_recursive(symbols[0])
            curr_node.right = self.gen_tree_recursive(symbols[1])
            # print(curr_node.ele, curr_node.left.ele, curr_node.right.ele)
        return curr_node

    def generate(self, n):
        res = []
        for _ in range(n):
            gen, ll = self._generate_derivation(self.start())
            res.append((gen, ll))
        return res

    def _generate_derivation(self, nonterminal):
        sentence = []
        derivation = ""
        total_ll = 0
        symbols, lls = self._reduce_once(nonterminal)
        total_ll += lls
        for symbol in symbols:
            if isinstance(symbol, str):
                derivation = symbol
            else:
                derivation, lls = self._generate_derivation(symbol)
                total_ll += lls
            if derivation != "":
                sentence.append(derivation)
        return " ".join(sentence), total_ll

    def _reduce_once(self, nonterminal):
        prod = self._choose_production_reducing(nonterminal)
        # print(prod)
        return prod.rhs(), np.log(prod.prob())

    def _choose_production_reducing(self, nonterminal):
        productions = self._lhs_index[nonterminal]
        probabilities = [production.prob() for production in productions]
        return random.choices(productions, weights=probabilities)[0]

    def _get_ll(self, node, vd):
        def _convert(ele):
            if ele > len(vd) - 1:
                # print(self.nt_symbols_list[ele-len(vd)])
                return self.nt_symbols_list[ele - len(vd)]
            # print(vd[ele])
            return vd[ele]

        if node.left is None and node.right is None:
            return 0
        elif node.right is None:
            ll_child = self._get_ll(node.left, vd)
            current_production = self._get_exact_prod(
                lhs=_convert(node.ele), rhs=(_convert(node.left.ele),)
            )
            if current_production is not None:
                current_ll = np.log(current_production.prob())
            else:
                current_ll = LOGINF
            return ll_child + current_ll
        elif node.left is None:
            ll_child = self._get_ll(node.right, vd)
            current_production = self._get_exact_prod(
                lhs=_convert(node.ele), rhs=(_convert(node.right.ele),)
            )
            if current_production is not None:
                current_ll = np.log(current_production.prob())
            else:
                current_ll = LOGINF
            return ll_child + current_ll
        else:
            ll_child = self._get_ll(node.left, vd) + self._get_ll(node.right, vd)
            current_production = self._get_exact_prod(
                lhs=_convert(node.ele),
                rhs=(_convert(node.left.ele), _convert(node.right.ele)),
            )
            if current_production is not None:
                current_ll = np.log(current_production.prob())
            else:
                current_ll = LOGINF

            return ll_child + current_ll

    def get_log_likelihood(self, root, vocab_dict):
        return self._get_ll(root, vocab_dict)

    def compute_marginal(self, sentence, vocab_dict):
        # sentence = list of terminals
        sentence_len = len(sentence)
        num_nt = len(self.nt_symbols_list)
        beta = np.zeros((sentence_len, sentence_len, num_nt)) - 999
        nt_index = {}
        nt_syms = [str(self._start)] + [
            sym for sym in self.nt_symbols_list if sym != str(self._start)
        ]
        sentence = [vocab_dict[token] for token in sentence]
        # Prob CYK Algorithm
        for i, nt in enumerate(nt_syms):
            nt_index[nt] = i
        for i in range(sentence_len):
            for prod in self._productions:
                if len(prod.rhs()) == 1 and str(prod.rhs()[0]) == sentence[i]:
                    beta[i, 0, nt_index[str(prod.lhs())]] = log(prod.prob())
                    # print(prod)
        for j in range(1, sentence_len):
            for i in range(sentence_len - j):
                for l in range(j):
                    for prod in self._productions:
                        if len(prod.rhs()) == 2:
                            k = nt_index[str(prod.lhs())]
                            k1 = nt_index[str(prod.rhs()[0])]
                            k2 = nt_index[str(prod.rhs()[1])]
                            beta[i, j, k] += (
                                log(prod.prob())
                                + beta[i, l, k1]
                                + beta[i + l + 1, j - l - 1, k2]
                            )
        return beta[0, sentence_len - 1, 0]


def get_test_pcfg(choice=1, vocab_dict=None):
    if choice == 1:
        pcfg = PCFGExtended.fromstring(
            """
            S -> NP VP [1.0]
            NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]
            Det -> 'the' [0.8] | 'my' [0.2]
            N -> 'man' [0.5] | 'telescope' [0.5]
            VP -> VP PP [0.2] | V NP [0.8]
            V -> 'ate' [0.35] | 'saw' [0.65]
            PP -> P NP [1.0]
            P -> 'with' [0.61] | 'under' [0.39]
            """
        )
        pcfg.vocab_dict = vocab_dict
        return pcfg
    elif choice == 2:
        pcfg = PCFGExtended.fromstring(
            """
            A -> B C [1.0]
            B -> 'b' [1.0]
            C -> 'c' [1.0]
            """
        )
        pcfg.vocab_dict = vocab_dict
        return pcfg
    elif choice == 3:
        pcfg = PCFGExtended.fromstring(
            """
            A -> B C [0.5] | D E [0.5]
            B -> 'b' [1.0]
            C -> 'c' [1.0]
            D -> 'd' [1.0]
            E -> 'e' [1.0]
            """
        )
        pcfg.vocab_dict = vocab_dict
        return pcfg
    elif choice == 4:
        pcfg = PCFGExtended.fromstring(
            """
            A -> B C [1.0]
            B -> X B [0.5] | X X [0.5]
            C -> Y C [0.5] | Y Y [0.5]
            X -> 'x' [1.0]
            Y -> 'y' [1.0]
            """
        )
        pcfg.vocab_dict = vocab_dict
        return pcfg
    elif choice == 5:
        pcfg = PCFGExtended.fromstring(
            """
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
        )
        pcfg.vocab_dict = vocab_dict
        return pcfg


def get_test_tree():
    root = Node(1)
    left1 = Node(2)
    right1 = Node(2)
    root.left = left1
    root.right = right1
    left21 = Node(8)
    left1.left = left21
    left22 = Node(9)
    right1.left = left22
    return root


def test_pcfg():
    toy_pcfg1 = PCFGExtended.fromstring(
        """
            A -> B C [0.5] | D E [0.5]
            B -> 'b' [1.0]
            C -> 'c' [1.0]
            D -> 'd' [1.0]
            E -> 'e' [1.0]
            """
    )

    node = toy_pcfg1.generate_tree(1)[0]
    # import pdb;pdb.set_trace();
    queue = [node]
    while len(queue) > 0:
        curr = queue[0]
        print(curr.ele)
        if curr.left is not None:
            queue.append(curr.left)
        if curr.right is not None:
            queue.append(curr.right)
        queue = queue[1:]
    # print(toy_pcfg1.get_log_likelihood(get_test_tree(), {1:"A", 2:"B", 3: "C", 4:"D", 5:"E", 6:"b", 7:"c", 8:"d", 9:"e"}))
    print()


def create_data(choice=6, num=10000, base_path="data/ptb_pcfg/"):
    if choice == 1:
        pcfg = PCFGExtended.fromstring(
            """
            S -> NP VP [1.0]
            NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]
            Det -> 'the' [0.8] | 'my' [0.2]
            N -> 'man' [0.5] | 'telescope' [0.5]
            VP -> VP PP [0.2] | V NP [0.8]
            V -> 'ate' [0.35] | 'saw' [0.65]
            PP -> P NP [1.0]
            P -> 'with' [0.61] | 'under' [0.39]
            """
        )
    elif choice == 2:
        pcfg = PCFGExtended.fromstring(
            """
            A -> B C [1.0]
            B -> 'b' [1.0]
            C -> 'c' [1.0]
            """
        )
    elif choice == 3:
        pcfg = PCFGExtended.fromstring(
            """
            A -> B C [0.5] | D E [0.5]
            B -> 'b' [1.0]
            C -> 'c' [1.0]
            D -> 'd' [1.0]
            E -> 'e' [1.0]
            """
        )
    elif choice == 4:
        pcfg = PCFGExtended.fromstring(
            """
            A -> B C [1.0]
            B -> X B [0.5] | X X [0.5]
            C -> Y C [0.5] | Y Y [0.5]
            X -> 'x' [1.0]
            Y -> 'y' [1.0]
            """
        )
    elif choice == 5:
        pcfg = PCFGExtended.fromstring(
            """
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
        )
    elif choice == 6:
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

        # print(productions)
        pcfg = PCFGExtended(S, prods)
    train_num = int(0.7 * num)
    val_num = int(0.8 * num)
    test_num = num
    # import pdb; pdb.set_trace();
    examples = pcfg.generate(num)
    # examples = list(set(examples))
    # print(examples)
    with open(base_path + "train_ll.src", "w") as f_ll:
        with open(base_path + "train.src", "w") as f:
            for example in examples[:train_num]:
                f.write(example[0] + "\n")
                f_ll.write(str(example[1]) + "\n")

    with open(base_path + "val_ll.src", "w") as f_ll:
        with open(base_path + "valid.src", "w") as f:
            for example in examples[train_num:val_num]:
                f.write(example[0] + "\n")
                f_ll.write(str(example[1]) + "\n")

    with open(base_path + "test_ll.src", "w") as f_ll:
        with open(base_path + "test.src", "w") as f:
            for example in examples[val_num:test_num]:
                f.write(example[0] + "\n")
                f_ll.write(str(example[1]) + "\n")


if __name__ == "__main__":
    # test_pcfg()
    create_data()
    # pcfg = PCFGExtended.fromstring(
    #         """
    #         S -> NP VP [1.0]
    #         NP -> 'time' [0.4]
    #         NP -> N N [0.2]
    #         NP -> D N [0.4]
    #         V -> 'like' [0.3]
    #         V -> 'flies' [0.7]
    #         P -> 'like' [1.0]
    #         VP -> V NP [0.5]
    #         VP -> V PP [0.5]
    #         PP -> P NP [1.0]
    #         N -> 'time' [0.5]
    #         N -> 'arrow' [0.3]
    #         N -> 'flies' [0.2]
    #         D -> 'an' [1.0]
    #         """
    # )
    # print(pcfg.compute_marginal("time flies like an arrow".split(" ")))
