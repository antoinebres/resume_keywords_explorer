import pickle
import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from scripts.extract_top_skills import *

corpus_type = st.sidebar.selectbox(
    "Select corpus",
    ("individuals", "concatenated")
)

root_dir = os.path.dirname(os.path.abspath(__file__))
with open(root_dir +'/data/directories.pkl', "rb") as f:
	directories = list(map(int, pickle.load(f)))

offer_to_dir = load_offer_category_and_cv_dir(root_dir +'/data/offer_to_cvdir.csv.zip')


@st.cache
def load_tfidf(corpus_type):
	if corpus_type == 'individuals':
		return pd.read_csv(root_dir +'/data/tfidf_corpus.csv.zip', index_col='dir_id')
	elif corpus_type == 'concatenated':
		return pd.read_csv(root_dir +'/data/tfidf_n_corpus.csv.zip', index_col='offer_title')

	return pd.read_csv(root_dir +'/data/tfidf_corpus.csv.zip', index_col='dir_id')


tfidf = load_tfidf(corpus_type)

st.title('Job keyword app\nExplore the most [relevant words](https://fr.wikipedia.org/wiki/TF-IDF) in the resumes associated with a given job offer\n\n(resumes are those of candidates applying to a french software consulting company over 2018 and 2019)')
st.header('Select a job offer and see the keywords associated with it')
offer_title = st.selectbox('', sorted(offer_to_dir['offer_title'].unique()))
sub_list_of_dirs = offer_to_dir[offer_to_dir['offer_title'] == offer_title]['dir_id'].astype(int).to_list()
sub_list_of_dirs = list(set(sub_list_of_dirs) & set(directories))
st.write(f'There are <{(round(len(sub_list_of_dirs)/10) + 1) * 10} CVs associated with the offer {offer_title}')
st.subheader('Keywords associated with those CVs:')
col1, col2 = st.columns(2)
slot1a = col1.empty()
slot1b = col1.empty()
slot2a = col2.empty()
slot2b = col2.empty()
if st.checkbox('Filter out'):
	skip_words = st.text_area('Write each term (unigram/bigram) you want to filter out of the wordclouds on a new line')
	try:
		skip_words = list(map(lambda x: x.strip(), skip_words.split("\n")))
	except SyntaxError:
		pass
	except Exception as e:
		pass
else:
	skip_words = []

if corpus_type == 'individuals':
	subset_candidates = tfidf[tfidf.index.isin(sub_list_of_dirs)].dropna()
	top = subset_candidates.sum().sort_values(ascending=False)
elif corpus_type == 'concatenated':
	subset_candidates = tfidf.loc[offer_title]
	top = subset_candidates.sort_values(ascending=False)

top_unigrams, top_bigrams = separe_unigrams_and_bigrams(filter_out(top, skip_words + ['cid', 'cid cid','gramme gramme', 'gramme']))

if top_unigrams and top_bigrams:
	slot1a.header("Single words (unigrams)")
	fig = plt.figure()
	wordcloud = WordCloud(background_color="white").generate_from_frequencies(top_unigrams)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	slot1b.pyplot(fig)
	slot2a.header("Couple of words (bigrams)")
	fig = plt.figure()
	wordcloud = WordCloud(background_color="white").generate_from_frequencies(top_bigrams)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	slot2b.pyplot(fig)
else:
	slot1a.header("Error: No keyword")
	slot2a.header("Error: No keyword")
