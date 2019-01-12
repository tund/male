try:
    from .word2vec import Word2Vec
except ImportError as e:
    print('[WARNING]', e)

try:
    from .node2vec import Node2Vec
except ImportError as e:
    print('[WARNING]', e)
