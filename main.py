import logging
from os.path import exists
import re
from glob import glob
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot as plt
import matplotlib
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO)

# transform data to get only text inputs
def join_text(line):
    ementa, ementaDetalhada, keywords = line['ementa'], line['ementaDetalhada'], line['keywords']
    text = ''
    if not pd.isna(ementa):
        text += ementa
    if not pd.isna(ementaDetalhada):
        text += ' ' + ementaDetalhada
    if not pd.isna(keywords):
        text += ' ' + keywords
    return text

def tokenizer_builder(stopwords, stemmer, stem_to_word):
    def tokenize_with_stemming(text):
        words = [word for word in word_tokenize(text) if word.lower() not in stopwords]
        words = [word for word in words if re.match(r'^[a-z]+$', word)]
        result = []
        for word in words:
            stemmed = stemmer.stem(word)
            result.append(stemmed)
            stem_to_word[stemmed] = word
        return result
    return tokenize_with_stemming

class Topic(str):
    def set_topic(self, topic):
        self.topic = topic
        return self

    def __hash__(self):
        topic_hash = (str(self), self.topic)
        return topic_hash.__hash__()

def topic_color_builder(colormap):
    def topic_color(word, **kwargs):
        r, g, b, a = [int(i*255) for i in colormap(word.topic)]
        return f'rgba({r}, {g}, {b}, {a})'
    return topic_color


def plot_wordcloud(width, height, max_words, frequencies, output_file, colormap = None):
    wordcloud = WordCloud(width=width,
            height=height,
            max_words=max_words,
            color_func=topic_color_builder(colormap) if colormap else None,
            background_color='white').generate_from_frequencies(frequencies)
    wordcloud.to_file(output_file)

def plot_topic_bars(topics, Sigma, output_file):
    y_pos = np.arange(len(topics))

    fig, ax = plt.subplots()
    fig.set_figheight(40)
    fig.set_figwidth(20)

    ax.grid()
    ax.barh(y_pos, Sigma, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics, fontsize=20)
    ax.set_ylim([-1, len(topics)])
    ax.set_xlabel('Relevance', fontsize=20)
    ax.set_title('LSA topics relevance', fontsize=40)
    plt.savefig(output_file)

def filter_dataset(dataset, ano_start, ano_end, partido):
    dataset = dataset.loc[dataset['ano'] >= ano_start]
    dataset = dataset.loc[dataset['ano'] <= ano_end]
    if partido:
      dataset = dataset.loc[dataset['siglaPartidoAutor'] == partido]

    dataset['documents'] = dataset[['ementa', 'ementaDetalhada', 'keywords']].agg(join_text, axis=1)
    # drop all columns except id and documents
    dataset = dataset.drop(columns=[c for c in dataset.columns if c not in ['id', 'documents']])
    # drop nan
    dataset = dataset.replace('', np.nan)
    dataset = dataset.dropna(subset=['documents'])
    return dataset

def process_dataset(dataset, num_topics, num_words_per_topic):
    stemmer = RSLPStemmer()
    pt_stopwords = stopwords.words('portuguese')
    pt_stopwords += ['lei', 'federal', 'nacional']
    stem_to_word = {}

    vectorizer = TfidfVectorizer(strip_accents='ascii', tokenizer=tokenizer_builder(pt_stopwords, stemmer, stem_to_word))
    X = vectorizer.fit_transform(dataset['documents'])

    svd = TruncatedSVD(n_components=num_topics, n_iter=7, random_state=42)
    lsa_data = svd.fit_transform(X)
    Sigma = svd.singular_values_

    vocab = vectorizer.get_feature_names()
    topics = []
    topics_relevance = []

    for i, comp in enumerate(svd.components_):
        vocab_comp = zip(vocab, comp)
        sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:num_words_per_topic]
        topics_relevance.append(sorted_words)
        topics.append(', '.join([stem_to_word[w] for w, v in sorted_words]))
    return stem_to_word, Sigma, topics, topics_relevance

def main():
    logging.info('starting program...')
    # install nltk dependencies
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('rslp')

    # setup parameters
    ano_start = 1934
    ano_end = 1954
    partido = ''
    num_topics = 100
    num_words_per_topic = 10
    output_dir = 'output'

    # read dataset
    preprocessed_file = 'data/bmt_projeto_final_preprocessed.gzip'
    logging.info(f'reading dataset in {preprocessed_file}...')
    base = pd.read_json(preprocessed_file, compression='gzip')

    for ano in range(1955, 2023):
        ano_start = ano
        ano_end = ano
        # filter dataset
        logging.info(f'filtering dataset with ano_start={ano_start}, ano_end={ano_end}, partido={partido}...')
        dataset = filter_dataset(base.copy(), ano_start, ano_end, partido)

        logging.info(f'processing dataset with num_topics={num_topics}, num_words_per_topic={num_words_per_topic}...')
        stem_to_word, Sigma, topics, topics_relevance = process_dataset(dataset, num_topics, num_words_per_topic)

        # plot relevance
        logging.info(f'plotting results in {output_dir}...')
        party = partido if partido else 'all'
        plot_topic_bars(topics, Sigma, f'{output_dir}/topic_relevance_{party}_{ano_start}_{ano_end}.png')

        topics_obj = {Topic(stem_to_word[k]).set_topic(i):v for i, items in enumerate(topics_relevance) for k, v in items}
        colormap = matplotlib.cm.get_cmap(name='magma', lut=num_topics)

        # topic relevance
        plot_wordcloud(800, 400,
                num_words_per_topic*num_topics,
                {v:Sigma[i] for i, v in enumerate(topics)},
                f'{output_dir}/wordcloud_topics_{party}_{ano_start}_{ano_end}.png')

        # word relevance for each topic
        plot_wordcloud(800, 400,
                num_words_per_topic*num_topics,
                topics_obj,
                f'{output_dir}/wordcloud_word_relevance_by_topic_{party}_{ano_start}_{ano_end}.png',
                colormap)

        # word relevance for each topic combined with topic relevance
        plot_wordcloud(800, 400,
                num_words_per_topic*num_topics,
                {k:v*Sigma[k.topic] for k, v in topics_obj.items()},
                f'{output_dir}/wordcloud_word_relevance_{party}_{ano_start}_{ano_end}.png',
                colormap)


if __name__ == '__main__':
    main()
