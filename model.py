# Base
import warnings
import numpy as np
import pandas as pd
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)

# Visualization
from IPython.display import Markdown
import ipywidgets as widgets
from plotly.graph_objs import FigureWidget
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm

#Sentence Transformes framework (HuggingFace)
from sentence_transformers import SentenceTransformer

# Cosine similarity
from sklearn.metrics.pairwise  import cosine_similarity

# NLP
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stop_words
sp = spacy.load('en_core_web_sm')

roberta_stsb = SentenceTransformer('stsb-roberta-large')
encoded_indicators = pd.read_csv('Indicators/encoded_indicators.csv').set_index('indicator')
encoded_indicators.columns =[int(c) for c in encoded_indicators.columns]

with open('Files/fitted_pca.pickle', "rb") as file:
    pca = pickle.load(file)
    
with open('Files/commitment_texts.pickle', "rb") as file:
    commitment_texts = pickle.load(file)

def remove_stopwords(text: str):
    
    """
    Removes stopwords from a given text
    """ 
    
    return ' '.join([token.text for token in sp(text) if not token.is_stop]).strip()


def base_encoding(sentences: list):
    
    """
    Encodes a given list of sentences to match 
    the dimensions of the encoded indicators
    """
    encoded_sentences = pca.transform(roberta_stsb.encode(pd.Series(sentences).apply(remove_stopwords)))
    return pd.DataFrame(encoded_sentences, index=sentences)


def match_sentences_indicators(sentences: list):
    
    """
    Matches given sentences to encoded
    indicators in a "correlation" matrix
    """
    
    final_matrix = pd.concat([encoded_indicators, base_encoding(sentences)])
    return pd.DataFrame(cosine_similarity(final_matrix), index=final_matrix.index, columns=final_matrix.index)


def sentence_top_matches(matched_sentences: list,
                         sentences_matrix: pd.DataFrame = None,
                         valid_sentences: list = list(encoded_indicators.index),
                         n_top=5):
    
    """
    Applies the previously defined functions to return 
    the top matches of the given list of sentences
    """
    
    if not sentences_matrix:
        sentences_matrix = match_sentences_indicators(matched_sentences)
    
    matched_sentences = sentences_matrix.loc[matched_sentences][valid_sentences].drop_duplicates()
    matched_sentences = matched_sentences.unstack().sort_values(ascending=False)
    final_matched_sentences = matched_sentences[matched_sentences < 0.999].groupby(level=1).head(n_top).reset_index()
    final_matches = final_matched_sentences.groupby("level_1").agg(list)
    
    indexes = pd.DataFrame(final_matches["level_0"].to_list())
    indexes.columns = ['top_match_' + str(c) for c in indexes.columns]
    indexes.index = final_matches.index
    
    similarities = pd.DataFrame(final_matches[0].to_list())
    similarities.columns = ['similarity_' + str(c) for c in similarities.columns]  
    similarities.index = final_matches.index
    
    return pd.concat([final_matches.drop(columns=["level_0", 0]), pd.concat([indexes, similarities], axis=1)], axis=1)


def remove_subtext(text_list: list, minimum_length: int):
    
    """
    Filters strings with less words than the given number 
    (minimum_length) from a given list of strings (text_list)
    """
    
    return [text.strip() for text in text_list if len(text.strip().split(" ")) >= minimum_length]


def process_sentences(text_list: list, minimum_length: int=3):
    
    """
    Processes the strings within the given list with 
    the remove_subtext function splitting the
    original list when finding a dot or a coma
    """
    
    return pd.Series(text_list).str.replace(",", ".").str.split(".").apply(lambda x: remove_subtext(x, minimum_length))


def subsentences_top_matches(text_list: list,
                             minimum_length: int=3,
                             n_top_calc:int=10):
    
    """
    
    """
    
    text_dataframe = pd.DataFrame(pd.Series(text_list, name='text'))
    exploded_df = pd.DataFrame(process_sentences(text_list, minimum_length).explode()).reset_index().set_index(0)
    exploded_df.index.names = ['sentence']
    
    final_df = text_dataframe.merge(exploded_df, left_index=True, right_on='index').drop(columns='index')
    final_df = final_df[final_df.index.notnull()]
    
    return final_df.join(sentence_top_matches(list(final_df.index), n_top=n_top_calc))


def soft_voting_classifier(series: pd.Series,
                           n_top:int=3):
    
    """
    Ranks the highest similarity matches for a  
    given series containing all of their values
    """
    
    values = pd.DataFrame(series).apply(pd.Series.explode).reset_index()
    
    final_top_matches = {}
    
    matches = values[values["index"].str.contains('top_match')][series.name]
    similarities = values[values["index"].str.contains('similarity')][series.name]
    
    for match, similarity in zip(matches, similarities):
        if match not in final_top_matches:
            final_top_matches[match] = similarity
        else:
            final_top_matches[match] += similarity
            
    processed_dict = pd.Series(final_top_matches).sort_values(ascending=False)[:n_top].reset_index()
    processed_dict = processed_dict.rename(columns={"index": 'top_match', 0: 'similarity'})
    
    return {'_'.join([str(i) for i in k]):v for k, v in processed_dict.T.stack().to_dict().items()}


def robust_sentence_top_matches(text_list: list,
                                minimum_length: int=3,
                                n_top:int=5,
                                n_top_calc:int=10):
    
    """
    Applies the previously defined functions to return 
    the top submatches ranked by the voting classifier and
    returns them in the same format as sentence_top_matches
    """
    
    processed_sentences = subsentences_top_matches(text_list, minimum_length, n_top_calc).groupby('text').agg(list).T
    return processed_sentences.apply(lambda x: soft_voting_classifier(x, n_top)).apply(pd.Series)
