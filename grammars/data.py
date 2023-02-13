import os
from io import open
import torch
import random
import pickle
import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


# [corpus.dictionary.idx2word[idx] for idx in data[:40].tolist()]
from collections import defaultdict


class SeqCorpus(object):
    def __init__(
        self,
        path,
        seqlen=-1,
        train_batch_size=1,
        valid_batch_size=1,
        batch_group_size=100,
        add_master_token=False,
        device="cuda",
        load_spans=False,
    ):
        self.dict = Dictionary()
        self.train = None
        self.valid = None
        self.test = None
        self.device = device
        self.seqlen = seqlen
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.batch_group_size = batch_group_size
        self.add_master_token = add_master_token
        self.load_spans = load_spans
        # add special tokens
        self.dict.add_word("<pad>")
        self.dict.add_word("<MT>")
        if not self.load_cache(path):
            if load_spans:
                self.train, self.train_spans = self.tokenize(path, split="train")
                self.valid, self.valid_spans = self.tokenize(path, split="valid")
                self.test, self.test_spans = self.tokenize(path, split="test")
            else:
                self.train = self.tokenize(path, split="train")
                self.valid = self.tokenize(path, split="valid")
                self.test = self.tokenize(path, split="test")
            self.save_cache(path)
        for loader in ["train", "valid", "test"]:
            if loader == "train":
                bsz = self.train_batch_size
            else:
                bsz = self.valid_batch_size
            tmp_copy = getattr(self, loader)
            tmp_copy = tmp_copy[: tmp_copy.size(0) // bsz * bsz]
            setattr(
                self,
                loader,
                tmp_copy.reshape(tmp_copy.size(0) // bsz, bsz, tmp_copy.size(-1)),
            )
            if self.load_spans:
                tmp_copy = getattr(self, loader + "_spans")
                tmp_copy = tmp_copy[: len(tmp_copy) // bsz * bsz]
                setattr(
                    self,
                    loader + "_spans",
                    np.split(np.array(tmp_copy), len(tmp_copy) // bsz),
                )
        # import pdb; pdb.set_trace();

    def sort_n_shuffle(self, dataloader, split):
        if split == "train":
            bsz = self.train_batch_size
        else:
            bsz = self.valid_batch_size
        if self.load_spans:
            dataloader = sorted(dataloader, key=lambda x: len(x[0]))
        else:
            dataloader = sorted(dataloader, key=lambda x: len(x))
        groups = []
        for i, sample in enumerate(dataloader, 0):
            if i % (bsz * self.batch_group_size) == 0:
                groups.append([])
            groups[-1].append(sample)
        for group in groups:
            random.shuffle(group)
        if self.load_spans:
            seq_dataloader = [ele[0] for group in groups for ele in group]
            span_dataloader = [ele[1] for group in groups for ele in group]
            return seq_dataloader, span_dataloader
        else:
            dataloader = [ele for group in groups for ele in group]
            return dataloader

    def load_cache(self, path):
        suffix = f".{self.seqlen}" if self.seqlen > 0 else ""
        suffix += ".no_MT" if not self.add_master_token else ""
        for cache in ["train.pt", "valid.pt", "test.pt", "dict.pt"]:
            cache_path = os.path.join(path, cache + suffix)
            if not os.path.exists(cache_path):
                return False
        self.dict = torch.load(os.path.join(path, f"dict.pt{suffix}"))
        self.train = torch.load(os.path.join(path, f"train.pt{suffix}"))
        self.valid = torch.load(os.path.join(path, f"valid.pt{suffix}"))
        self.test = torch.load(os.path.join(path, f"test.pt{suffix}"))
        if self.load_spans:
            self.train_spans = torch.load(os.path.join(path, f"train-spans.pt{suffix}"))
            self.valid_spans = torch.load(os.path.join(path, f"valid-spans.pt{suffix}"))
            self.test_spans = torch.load(os.path.join(path, f"test-spans.pt{suffix}"))
        return True

    def save_cache(self, path):
        suffix = f".{self.seqlen}" if self.seqlen > 0 else ""
        suffix += ".no_MT" if not self.add_master_token else ""
        torch.save(self.dict, os.path.join(path, f"dict.pt{suffix}"))
        torch.save(self.train, os.path.join(path, f"train.pt{suffix}"))
        torch.save(self.valid, os.path.join(path, f"valid.pt{suffix}"))
        torch.save(self.test, os.path.join(path, f"test.pt{suffix}"))
        if self.load_spans:
            torch.save(self.train_spans, os.path.join(path, f"train-spans.pt{suffix}"))
            torch.save(self.valid_spans, os.path.join(path, f"valid-spans.pt{suffix}"))
            torch.save(self.test_spans, os.path.join(path, f"test-spans.pt{suffix}"))

    def tokenize(self, path, split):
        """Tokenizes a text file."""
        src_path = os.path.join(path, split) + ".src"
        trees_path = os.path.join(path, split) + "-trees.pkl"
        assert os.path.exists(src_path)
        # Add words to the dictionary
        with open(src_path, "r", encoding="utf8") as src_f:
            src_idss = []
            spans = []
            if self.load_spans:
                src_spans = pickle.load(open(trees_path, "rb"))
            iterable = zip(src_f, src_spans) if self.load_spans else src_f
            for item in iterable:
                if self.load_spans:
                    src_line, span = item
                else:
                    src_line = item

                src_ids = []
                if self.add_master_token:
                    src_words = ["<MT>"] + src_line.split()
                else:
                    src_words = src_line.split()
                if self.seqlen > 0 and (len(src_words) > self.seqlen):
                    continue
                for word in src_words:
                    self.dict.add_word(word)
                    src_ids.append(self.dict.word2idx[word])
                src_idss.append(
                    torch.tensor(src_ids, device=self.device).type(torch.int64)
                )
                if self.load_spans:
                    spans.append(span)
        if self.load_spans:
            src_idss, spans = self.sort_n_shuffle(zip(src_idss, spans), split)
        else:
            src_idss = self.sort_n_shuffle(src_idss, split)
        # import pdb; pdb.set_trace();
        src_idss = torch.nn.utils.rnn.pad_sequence(src_idss, batch_first=True)
        if self.load_spans:
            return src_idss, spans
        else:
            return src_idss


class Seq2SeqCorpus(object):
    def __init__(
        self,
        path,
        seqlen=-1,
        batch_size=1,
        batch_group_size=100,
        add_master_token=True,
        device="cuda",
    ):
        self.dict_src, self.dict_tgt = Dictionary(), Dictionary()
        self.train_src, self.train_tgt = None, None
        self.valid_src, self.valid_tgt = None, None
        self.test_src, self.test_tgt = None, None
        self.device = device
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.batch_group_size = batch_group_size
        self.add_master_token = add_master_token
        # add special tokens
        self.dict_src.add_word("<pad>")
        self.dict_tgt.add_word("<pad>")
        self.dict_tgt.add_word("<MT>")
        if not self.load_cache(path):
            self.train_src, self.train_tgt = self.tokenize(os.path.join(path, "train"))
            self.valid_src, self.valid_tgt = self.tokenize(os.path.join(path, "valid"))
            self.test_src, self.test_tgt = self.tokenize(os.path.join(path, "test"))
            self.save_cache(path)
        for loader in [
            "train_src",
            "valid_src",
            "test_src",
            "train_tgt",
            "valid_tgt",
            "test_tgt",
        ]:
            tmp_copy = getattr(self, loader)
            tmp_copy = tmp_copy[: tmp_copy.size(0) // self.batch_size * self.batch_size]
            setattr(
                self,
                loader,
                tmp_copy.reshape(
                    tmp_copy.size(0) // self.batch_size,
                    self.batch_size,
                    tmp_copy.size(-1),
                ),
            )

    def sort_n_shuffle(self, dataloader):
        dataloader = sorted(dataloader, key=lambda x: len(x[0]))
        groups = []
        for i, sample in enumerate(dataloader, 0):
            if i % (self.batch_size * self.batch_group_size) == 0:
                groups.append([])
            groups[-1].append(sample)
        for group in groups:
            random.shuffle(group)
        dataloader = [ele for group in groups for ele in group]
        return dataloader

    def load_cache(self, path):
        suffix = f".{self.seqlen}" if self.seqlen > 0 else ""
        suffix += ".no_MT" if not self.add_master_token else ""
        for cache in [
            "train_src.pt",
            "valid_src.pt",
            "test_src.pt",
            "dict_src.pt",
            "train_tgt.pt",
            "valid_tgt.pt",
            "test_tgt.pt",
            "dict_tgt.pt",
        ]:
            cache_path = os.path.join(path, cache + suffix)
            if not os.path.exists(cache_path):
                return False
        self.dict_src = torch.load(os.path.join(path, f"dict_src.pt{suffix}"))
        self.dict_tgt = torch.load(os.path.join(path, f"dict_tgt.pt{suffix}"))
        self.train_src = torch.load(os.path.join(path, f"train_src.pt{suffix}"))
        self.valid_src = torch.load(os.path.join(path, f"valid_src.pt{suffix}"))
        self.test_src = torch.load(os.path.join(path, f"test_src.pt{suffix}"))
        self.train_tgt = torch.load(os.path.join(path, f"train_tgt.pt{suffix}"))
        self.valid_tgt = torch.load(os.path.join(path, f"valid_tgt.pt{suffix}"))
        self.test_tgt = torch.load(os.path.join(path, f"test_tgt.pt{suffix}"))
        return True

    def save_cache(self, path):
        suffix = f".{self.seqlen}" if self.seqlen > 0 else ""
        suffix += ".no_MT" if not self.add_master_token else ""
        torch.save(self.dict_src, os.path.join(path, f"dict_src.pt{suffix}"))
        torch.save(self.dict_tgt, os.path.join(path, f"dict_tgt.pt{suffix}"))
        torch.save(self.train_src, os.path.join(path, f"train_src.pt{suffix}"))
        torch.save(self.valid_src, os.path.join(path, f"valid_src.pt{suffix}"))
        torch.save(self.test_src, os.path.join(path, f"test_src.pt{suffix}"))
        torch.save(self.train_tgt, os.path.join(path, f"train_tgt.pt{suffix}"))
        torch.save(self.valid_tgt, os.path.join(path, f"valid_tgt.pt{suffix}"))
        torch.save(self.test_tgt, os.path.join(path, f"test_tgt.pt{suffix}"))

    def tokenize(self, path):
        """Tokenizes a text file."""
        src_path = path + ".src"
        tgt_path = path + ".tgt"
        assert os.path.exists(src_path)
        assert os.path.exists(tgt_path)
        # Add words to the dictionary
        with open(src_path, "r", encoding="utf8") as src_f:
            with open(tgt_path, "r", encoding="utf8") as tgt_f:
                src_idss, tgt_idss = [], []
                for src_line, tgt_line in zip(src_f, tgt_f):
                    src_ids, tgt_ids = [], []
                    src_words = src_line.split()
                    if self.add_master_token:
                        tgt_words = ["<MT>"] + tgt_line.split()
                    else:
                        tgt_words = tgt_line.split()
                    if self.seqlen > 0 and (
                        len(src_words) > self.seqlen
                        or len(tgt_words) > (self.seqlen + 1)
                    ):
                        continue
                    for word in src_words:
                        self.dict_src.add_word(word)
                        src_ids.append(self.dict_src.word2idx[word])
                    for word in tgt_words:
                        self.dict_tgt.add_word(word)
                        tgt_ids.append(self.dict_tgt.word2idx[word])
                    src_idss.append(
                        torch.tensor(src_ids, device=self.device).type(torch.int64)
                    )
                    tgt_idss.append(
                        torch.tensor(tgt_ids, device=self.device).type(torch.int64)
                    )

        (src_idss, tgt_idss) = tuple(zip(*self.sort_n_shuffle(zip(src_idss, tgt_idss))))
        src_idss = torch.nn.utils.rnn.pad_sequence(src_idss, batch_first=True)
        tgt_idss = torch.nn.utils.rnn.pad_sequence(tgt_idss, batch_first=True)

        return src_idss, tgt_idss


class Seq2TreeCorpus(object):
    def __init__(self, path, seqlen=-1, device="cuda"):
        self.dict_src, self.dict_tgt = Dictionary(), Dictionary()
        self.train_src, self.train_tgt = None, None
        self.valid_src, self.valid_tgt = None, None
        self.test_src, self.test_tgt = None, None
        self.device = device
        self.seqlen = seqlen
        # add special tokens
        self.dict_src.add_word("<pad>")
        self.dict_tgt.add_word("<pad>")
        self.dict_tgt.add_word("TOP")
        if not self.load_cache(path):
            self.train_src, self.train_tgt = self.tokenize(os.path.join(path, "train"))
            self.valid_src, self.valid_tgt = self.tokenize(os.path.join(path, "valid"))
            self.test_src, self.test_tgt = self.tokenize(os.path.join(path, "test"))
            self.save_cache(path)

    def load_cache(self, path):
        for cache in [
            "train_src.pt",
            "valid_src.pt",
            "test_src.pt",
            "dict_src.pt",
            "train_tgt.pt",
            "valid_tgt.pt",
            "test_tgt.pt",
            "dict_tgt.pt",
        ]:
            cache_path = os.path.join(
                path, f"{cache}.{self.seqlen}" if self.seqlen > 0 else cache
            )
            if not os.path.exists(cache_path):
                return False
        suffix = f".{self.seqlen}" if self.seqlen > 0 else ""
        self.dict_src = torch.load(os.path.join(path, f"dict_src.pt{suffix}"))
        self.dict_tgt = torch.load(os.path.join(path, f"dict_tgt.pt{suffix}"))
        self.train_src = torch.load(os.path.join(path, f"train_src.pt{suffix}"))
        self.valid_src = torch.load(os.path.join(path, f"valid_src.pt{suffix}"))
        self.test_src = torch.load(os.path.join(path, f"test_src.pt{suffix}"))
        self.train_tgt = torch.load(os.path.join(path, f"train_tgt.pt{suffix}"))
        self.valid_tgt = torch.load(os.path.join(path, f"valid_tgt.pt{suffix}"))
        self.test_tgt = torch.load(os.path.join(path, f"test_tgt.pt{suffix}"))
        return True

    def save_cache(self, path):
        suffix = f".{self.seqlen}" if self.seqlen > 0 else ""
        torch.save(self.dict_src, os.path.join(path, f"dict_src.pt{suffix}"))
        torch.save(self.dict_tgt, os.path.join(path, f"dict_tgt.pt{suffix}"))
        torch.save(self.train_src, os.path.join(path, f"train_src.pt{suffix}"))
        torch.save(self.valid_src, os.path.join(path, f"valid_src.pt{suffix}"))
        torch.save(self.test_src, os.path.join(path, f"test_src.pt{suffix}"))
        torch.save(self.train_tgt, os.path.join(path, f"train_tgt.pt{suffix}"))
        torch.save(self.valid_tgt, os.path.join(path, f"valid_tgt.pt{suffix}"))
        torch.save(self.test_tgt, os.path.join(path, f"test_tgt.pt{suffix}"))

    def tokenize(self, path):
        """Tokenizes a text file."""
        src_path = path + ".src"
        tgt_path = path + ".tgt"
        assert os.path.exists(src_path)
        assert os.path.exists(tgt_path)
        # Add words to the dictionary
        with open(src_path, "r", encoding="utf8") as src_f:
            with open(tgt_path, "r", encoding="utf8") as tgt_f:
                src_idss, tgt_idss, tgt_nchs = [], [], []
                for src_line, tgt_line in zip(src_f, tgt_f):
                    tgt_tokens, tgt_n_child = tgt_line.split("\t")
                    src_ids, tgt_ids = [], []
                    src_words = src_line.split()
                    tgt_words = tgt_tokens.split()
                    tgt_n_childs = [int(x) for x in tgt_n_child.split()]
                    if self.seqlen > 0 and (
                        len(src_words) > self.seqlen
                        or len(tgt_words) > (self.seqlen + 1)
                    ):
                        continue
                    for word in src_words:
                        self.dict_src.add_word(word)
                        src_ids.append(self.dict_src.word2idx[word])
                    for word in tgt_words:
                        self.dict_tgt.add_word(word)
                        tgt_ids.append(self.dict_tgt.word2idx[word])
                    src_idss.append(
                        torch.tensor(src_ids, device=self.device).type(torch.int64)
                    )
                    tgt_idss.append(
                        torch.tensor(tgt_ids, device=self.device).type(torch.int64)
                    )
                    tgt_nchs.append(
                        torch.tensor(tgt_n_childs, device=self.device).type(torch.int64)
                    )
        src_idss = torch.nn.utils.rnn.pad_sequence(src_idss, batch_first=True)
        tgt_idss = torch.nn.utils.rnn.pad_sequence(tgt_idss, batch_first=True)
        tgt_nchs = torch.nn.utils.rnn.pad_sequence(tgt_nchs, batch_first=True)

        return src_idss, zip(tgt_idss, tgt_nchs)
