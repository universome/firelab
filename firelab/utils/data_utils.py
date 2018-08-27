import torch

from .training_utils import cudable


def onehot_encode(x, vocab_size):
    "One-hot encodes batch of sequences of numbers"
    assert x.dim() == 2 # batch_size * seq_len

    out = cudable(torch.zeros(x.size(0), x.size(1), vocab_size).long())
    out = out.scatter_(2, x.unsqueeze(2), 1) # Filling with ones

    return out


def filter_sents_by_len(sents, min_len, max_len):
    return [s for s in sents if min_len <= len(s.split()) <= max_len]
