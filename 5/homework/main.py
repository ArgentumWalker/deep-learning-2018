import pandas as pd
from sklearn.manifold import TSNE

from model.torch_model import *
from utils.utils import *
from utils.tensorboard_util import *
from utils.pyplot_utils import *

VOCAB_SIZE = 9424
EMBED_SIZE = 100  # dimension of the word embedding vectors
NEGATIVE_SAMPLES = 5
LEARNING_RATE = 0.8
ITERATIONS = 100000

BATCH_SIZE = 1024
SKIP_WINDOW = 12  # the context window

seq_df = pd.read_table('data/family_classification_sequences.tab')
seq_df.head()
dictionary = make_dictionary(seq_df)
all_codones = read_or_create(read_path='data/all_codones.pickle',
                             producer=lambda: create_all_codones(seq_df, dictionary))

final_embed_matrix = train_skipgram(all_codones, VOCAB_SIZE, EMBED_SIZE, SKIP_WINDOW, NEGATIVE_SAMPLES, BATCH_SIZE, ITERATIONS)


filename = 'data/acid_properties.csv'
props = pd.read_csv(filename)

save_embedding("tensorboard", dictionary, final_embed_matrix, props)
plot_embedding("pyplot", dictionary, final_embed_matrix, props)
