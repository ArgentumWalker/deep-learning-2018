from utils.utils import *


def plot_embedding(directory, dictionary, final_embed_matrix, props):
    tsne = TSNE(n_components=2, random_state=42)
    XX = tsne.fit_transform(final_embed_matrix)
    tsne_df = pd.DataFrame(XX, columns=['x0', 'x1'])
    unique_codones = sorted(dictionary, key=dictionary.get)
    tsne_df['codone'] = list(unique_codones)
    tsne_df.head()

    save_path = 'data/all_acid_dicts.pickle'
    producer = lambda: [acid_dict(some_c, props) for some_c in tsne_df.codone]
    all_acid_dicts = read_or_create(save_path, producer)

    all_acid_df = pd.DataFrame(all_acid_dicts)
    all_acid_df.head()

    final_df = all_acid_df.join(tsne_df.set_index('codone'), on='acid')
    final_df.head()

    plot_embedding_properties(final_df, directory, "colored")
