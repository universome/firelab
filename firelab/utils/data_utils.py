import torch
from typing import List

from .training_utils import cudable


# TODO: looks like these functions are too specific. Remove them?
def onehot_encode(x, vocab_size):
    "One-hot encodes batch of sequences of numbers"
    assert x.dim() == 2 # batch_size * seq_len

    out = cudable(torch.zeros(x.size(0), x.size(1), vocab_size).long())
    out = out.scatter_(2, x.unsqueeze(2), 1) # Filling with ones

    return out


def filter_sents_by_len(sents:List[str], min_len:int, max_len:int) -> List[str]:
    return [s for s in sents if min_len <= len(s.split()) <= max_len]


def text_to_markdown(text: str) -> str:
    """
    Converts an arbitrarily text into a text that would be well-displayed in TensorBoard.
    TensorBoard uses markdown to render the text that's why it strips spaces and line breaks.
    This function fixes that.
    """
    text = text.replace(' ', '&nbsp;&nbsp;') # Because markdown does not support text indentation normally...
    text = text.replace('\n', '  \n') # Because tensorboard uses markdown

    return text
