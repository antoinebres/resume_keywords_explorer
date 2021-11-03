import pickle
import pandas as pd
import os


def count_kw(series, must_have, nice_to_have):
	count_kw = {}
	for w in must_have+nice_to_have:
	    count_kw[w] = series.str.lower().str.count(w)
	df_count_kw = pd.concat(list(count_kw.values()), axis=1, keys=count_kw.keys())
	return df_count_kw


def score(series):
	return round((series > 0).sum()  / len(series), 2)


def new_score(series):
	if (series[must_have] > 0).sum()  / len(must_have) >= 0.5:
		return round(0.5 + (score(series) / 2), 2)
	return score(series)

if __name__ == '__main__':
	root_dir = os.getcwd()

	with open(root_dir + "/data/directories.pkl", 'rb') as f: 
	    directories = pickle.load(f)

	with open(root_dir + "/data/corpus.pkl", 'rb') as f: 
	    corpus = pickle.load(f)

	epm_candidates = pd.read_csv(root_dir +'/data/epm_twig.csv', header=None)[0]
	corpus_series = pd.Series(corpus, name='raw_text', index=map(int, directories))
	corpus_epm = corpus_series[corpus_series.index.isin(epm_candidates)]
	corpus_epm = corpus_epm[~corpus_epm.index.duplicated(keep='first')]

	must_have = ["anaplan", "tagetik", "bpc", "epm"]
	nice_to_have = ["sap ip", "integrative planning"]

	df_count_kw = count_kw(corpus_epm, must_have, nice_to_have)

	out_df = pd.read_csv(root_dir +'/data/epm_twig.csv', header=None)
	out_df = out_df[[0,1,2,30,32]]
	out_df = out_df.set_index(0)
	out_df['old_score'] = df_count_kw.apply(score, axis=1)
	out_df['new_score'] = df_count_kw.apply(new_score, axis=1)
	out_df['total_kw_occurences'] = df_count_kw.sum(1).astype(int)
	out_df = out_df.dropna()
	out_df = pd.concat([out_df, df_count_kw], axis=1)
	out_df = out_df.rename(columns={1:'prenom', 2:'nom', 30:'cv', 32:'candidature'})
	# out_df.to_csv('/Users/antoine.bres/Desktop/candidatures_epm.csv')