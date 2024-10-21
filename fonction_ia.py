import tensorflow as tf
device_name = tf.test.gpu_device_name()# get_ipython().system('pip install sentence_transformers')
import scipy.spatial
import numpy as np
import os, json
import glob
import re
import torch
from sentence_transformers import SentenceTransformer, util
from tokenizers import Tokenizer
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
from tqdm import tqdm
import pandas as pd
import torch
import random
import re

def replace_emails(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.sub(email_pattern,'', text)

def replace_urls(text):
    url_pattern = r'(http|https|ftp)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(\/\S*)?'
    return re.sub(url_pattern, '', text)

def replace_citations(text):
    citation_pattern = r'\(\d+\)'
    return re.sub(citation_pattern,'', text)

def replace_citations2(text):
    citation_pattern = r'\(*\)'
    return re.sub(citation_pattern,'', text)

def replace_paragraphe(text):
    paragraphe_pattern=r'\<\/p>\<p id=""p\d*"" num=""\d*"">'
    return re.sub(paragraphe_pattern,'',text)

def delete_commentaire(text):
    commentaire_pattern=r'<!-- EPO <DP n="\d*"> -->'
    return re.sub(commentaire_pattern,'',text)

def delete_commentaire_double(text):
    commentaire_pattern=r'<!-- EPO <DP n=""\d*""> -->'
    return re.sub(commentaire_pattern,'',text)

def delete_balise(text):
    balise_pattern = r'<[^>A-Z]*>'
    return re.sub(balise_pattern,'',text)

def prediction(df,k=7):
    top_n = k
    for item in range(df.shape[0]):
        k_similar_patents = df.nlargest(top_n, ['cosine_similarity'])
        result_k_similar_patents = pd.DataFrame(0, index=np.arange(1),columns= k_similar_patents.columns[5:])
        for i in range(top_n):
            result_k_similar_patents  = result_k_similar_patents + k_similar_patents.iloc[i, 5:].values
        result_k_similar_patents_df = pd.DataFrame(result_k_similar_patents, columns= k_similar_patents.columns[5:])

    data = torch.tensor((result_k_similar_patents_df.to_numpy()).astype(float), dtype=torch.float32)
    print(data)
    m = nn.Sigmoid()
    output = m(data)
    probabilities = output.detach().numpy()
    output = (output>0.9).float()
    output_df = pd.DataFrame(output, columns=k_similar_patents.columns[5:]).astype(float)
    y_pred = output_df

    return y_pred,probabilities

def get_top_n_similar_patents_df(new_claim, claim_embeddings,stored_patent_train_embeddings_id):
    search_hits_list = []
    search_hits = util.semantic_search(new_claim, claim_embeddings, 10000, 5000000, 20)
    top_claim_order = []
    top_claim_ids = []
    top_similarity_scores = []
    for item in range(len(search_hits[0])):
        top_claim_order = search_hits[0][item].get('corpus_id')
        top_claim_ids.append(stored_patent_train_embeddings_id[top_claim_order])
        top_similarity_scores.append(search_hits[0][item].get('score'))

    top_100_similar_patents_df = pd.DataFrame({
        'top_claim_ids': top_claim_ids,
        'cosine_similarity': top_similarity_scores,
    })

    return top_100_similar_patents_df


def input_new_text(text):
  listofpredictdfs = []
  text = replace_emails(text)
  text = replace_urls(text)
  text = replace_citations(text)
  text = replace_citations2(text)
  text = replace_paragraphe(text)
  text = delete_commentaire(text)
  text = delete_commentaire_double(text)
  text = delete_balise(text)
  text_list = [text]

  model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')

  #Sentences are encoded by calling model.encode()
  test_claim_embeddings = model.encode(text_list, convert_to_tensor=True, show_progress_bar=True)
  train_claim_embeddings = torch.load('claim_vectorized_train.pt',map_location=torch.device('cpu'))
  train_df = torch.load('df_train.pt',map_location=torch.device('cpu'))
  train_df = pd.DataFrame(train_df)
  stored_patent_train_embeddings_id = train_df["Numéro d\'application"].reset_index(drop=True)

  claims = list(train_df['claim'])
  patent_id = list(train_df["Numéro d\'application"])


  for i in tqdm(range(len(text_list))):
      get_top_n_similar_patents_df_predict = get_top_n_similar_patents_df(np.array(test_claim_embeddings[i]).reshape(1,-1), train_claim_embeddings,stored_patent_train_embeddings_id)
      result = pd.merge(get_top_n_similar_patents_df_predict, train_df, left_on='top_claim_ids',right_on="Numéro d\'application",how='left',suffixes=('_left','_right'))
      predict = result.copy()
      listofpredictdfs.append(predict)

  df = pd.DataFrame(listofpredictdfs[0])
  y_pred, probabilities = prediction(df)

  return y_pred,probabilities,listofpredictdfs[0]

#text='A method for sending a keycode of a non-keyboard apparatus, comprising the steps of: (a) connecting the non-keyboard apparatus to a computer so as to perform device enumeration and generate enumeration information, wherein the enumeration information is recorded by the non-keyboard apparatus and includes an enumeration value; (b) identifying, according to the enumeration value, the kind of an operating system used by the computer, and recording the kind of the operating system by the non-keyboard apparatus; and (c) reading the kind of the operating system so as to determine a preset second keycode that matches the kind of the operating system, wherein the second keycode is an ASCII (American Standard Code for Information Interchange) code.'

#pred,prob,neighbors = input_new_text(text)

#print(prob)

