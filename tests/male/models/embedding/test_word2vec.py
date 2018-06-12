from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import pytest
from male.datasets import demo
from male.callbacks import Display
from male.models.embedding import Word2Vec


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_word2vec_text8(show_figure=False, block_figure_on_end=False):
    print("========== Test Word2Vec on Text8 - 1% data ==========")

    txt = demo.load_text8_1pct()

    loss_display = Display(layout=(1, 1),
                           dpi='auto',
                           freq=1,
                           show=show_figure,
                           block_on_end=block_figure_on_end,
                           monitor=[{'metrics': ['loss'],
                                     'type': 'line',
                                     'labels': ["Loss"],
                                     'title': "Losses",
                                     'xlabel': "epoch",
                                     'ylabel': "loss",
                                     },
                                    ])

    emb_display = Display(layout=(1, 1),
                          dpi='auto',
                          figsize=(10, 10),
                          freq=1,
                          show=show_figure,
                          block_on_end=block_figure_on_end,
                          monitor=[{'metrics': ['embedding'],
                                    'title': "Word Embeddings",
                                    'type': 'img',
                                    'num_words': 10,  # set to 500 for full run
                                    },
                                   ])

    model = Word2Vec(num_epochs=4,  # set to 100 for full run
                     metrics=['loss'],
                     callbacks=[loss_display, emb_display],
                     verbose=1)
    # model.fit(txt[:250])  # the first 42 words
    model.fit(txt[:300])  # the first 51 words
    # model.fit(txt)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_word2vec_text8(show_figure=True, block_figure_on_end=True)
