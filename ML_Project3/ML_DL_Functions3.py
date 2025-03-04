import numpy as np
from torch import nn as nn
import torch
def convert_words_to_indices(sents,vocab_stoi): # 10% grade
    """
    This function takes a list of sentences 
    input: list of list of words [[word,word,..,word],..,[word,..,word]] where each word is a string with no spaces
    and returns a new list with the same structure, but where each word is replaced by its index in `vocab_stoi`.
    output: list of lists of integers [[int,int,..,int],..,[int,..,int]] where each int is the idx of the word according to vocab_stoi

    Example:
    >>> convert_words_to_indices([['one', 'in', 'five', 'are', 'over', 'here'], ['other', 'one', 'since', 'yesterday'], ['you']])
    [[148, 98, 70, 23, 154, 89], [151, 148, 181, 246], [248]]
    """
    indices = []
    for sentence in sents:
        index_sentence = []  
        for word in sentence:
            index_sentence.append(vocab_stoi.get(word)) 
        indices.append(index_sentence)
    return indices
    # Write your code here

def generate_4grams(seqs): # 10% grade
    """
    This function takes a list of sentences (list of lists) and returns
    a new list containing the 4-grams (four consequentively occuring words)
    that appear in the sentences. Note that a unique 4-gram can appear multiple
    times, one per each time that the 4-gram appears in the data parameter `seqs`.

    Example:

    >>> generate_4grams([[148, 98, 70, 23, 154, 89], [151, 148, 181, 246], [248]])
    [[148, 98, 70, 23], [98, 70, 23, 154], [70, 23, 154, 89], [151, 148, 181, 246]]
    >>> generate_4grams([[1, 1, 1, 1, 1]])
    [[1, 1, 1, 1], [1, 1, 1, 1]]
    """
    fourgrams = []
    for sequence in seqs:
      for i in range(len(sequence) - 3):
        fourgrams.append(sequence[i:i+4])
    return fourgrams
    

    
def make_onehot(data): # 10% grade
    """
    Converts data in index notation into one-hot notation.

    Args:
        data: A 1D or 2D NumPy array of integers representing class indices.
        num_classes: Total number of classes (default: 250).

    Returns:
        A NumPy array where each index in `data` is converted to its one-hot representation.
        - If `data` is 1D with shape (D,), the output will be (D, num_classes).
        - If `data` is 2D with shape (N, D), the output will be (N, D, num_classes).
    """
    data = np.array(data)  # Ensure input is a NumPy array
    num_classes=250
    onehot = np.eye(num_classes)[data]
    return onehot

    # Write your code here

class PyTorchMLP(nn.Module): # 35% grade for each model
    def __init__(self):
        super(PyTorchMLP, self).__init__()
        self.num_hidden = 500 # TODO: choose number of hidden neurons
        self.layer1 = nn.Linear(750, self.num_hidden) 
        self.layer2 = nn.Linear(self.num_hidden, 250)
        
    def forward(self, inp):
        inp = inp.reshape([-1, 750])
        hidden = torch.relu(self.layer1(inp))
        output = self.layer2(hidden)
        return output
        
        # TODO: complete this function
        # Note that we will be using the nn.CrossEntropyLoss(), which computes the softmax operation internally, as loss criterion



        