import os
import pandas as pd
import pickle

def load_offer_category_and_cv_dir(file_path='/data/offer_to_cvdir.csv'):
    offer_to_dir = pd.read_csv(file_path)
    offer_to_dir['dir_id'] = offer_to_dir['dir_id'].astype('int')
    return offer_to_dir[['dir_id', 'offer_title']]


def separe_unigrams_and_bigrams(top, top_n=50):
	top_bigrams = [e for e in top.index if len(e.split()) > 1]
	top_onegrams = top[~top.index.isin(top_bigrams)][:top_n]
	top_onegrams = top_onegrams[top_onegrams > 0].to_dict()
	top_bigrams = top[top.index.isin(top_bigrams)][:top_n]
	top_bigrams = top_bigrams[top_bigrams > 0].to_dict()
	return top_onegrams, top_bigrams


def filter_out(top, filter_out_lst=[]):
	if filter_out_lst:
		top = top[~top.index.isin(filter_out_lst)]
	return top

if __name__ == '__main__':
	root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	tfidf = pd.read_csv(root_dir + '/data/tfidf_corpus.csv', index_col='dir_id')
	# word_count = pd.read_csv(root_dir + '/data/word_count_corpus.csv', index_col='dir_id')
	offer_to_dir = load_offer_category_and_cv_dir(root_dir +'/data/offer_to_cvdir.csv')


	epm_candidates = offer_to_dir[offer_to_dir['offer_title'].str.contains('EPM')]
	epm_candidates_id = epm_candidates['dir_id'].unique().tolist()
	datageek_candidates = offer_to_dir[offer_to_dir['offer_title'].str.contains('Data Geek')]
	datageek_candidates_id = datageek_candidates['dir_id'].unique().tolist()

	# print('datageek:' , get_specific_elements_to_an_offer(datageek_candidates_id, tfidf))
	# print('epm:' , get_specific_elements_to_an_offer(epm_candidates_id, tfidf))
