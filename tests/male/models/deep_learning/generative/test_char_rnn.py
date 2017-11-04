from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from male.datasets import demo
from male.models.deep_learning.generative import CharRNN
import pytest

def test_charRNN():
    print("========== Test charRNN for generating characters==========")

    x_data, char_to_ix, ix_to_char = demo.load_tiny_shakespeare()

    model = CharRNN(seq_length=50, num_epochs=50, char_to_ix=char_to_ix, ix_to_char=ix_to_char)
    model.fit(x_data)  # x_data is the training data

    generative_txt = model.sample(n_length=200)
    print("\n---sample---\n")
    print(generative_txt)

if __name__ == '__main__':
    pytest.main([__file__])
    # test_charRNN()
