import os
import sys
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from itertools import chain
from timeit import default_timer as timer
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from stop_words import get_stop_words
import pandas as pd
from datetime import timedelta
import textract 

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from utils.text_processor import TextProcessor


def apply_CountVectorizer(directories, corpus, min_df=None):
    if not min_df:
        min_df=.05 # les termes gardés sont ceux qui apparaissent dans au moins 5% du corpus

    vectorizer = CountVectorizer(ngram_range=(1,2), strip_accents='unicode', min_df=min_df)
    vectors = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    df = pd.concat([df, pd.Series(directories, name='dir_id').astype(int)], axis=1).set_index('dir_id')
    return df


def apply_TfidfVectorizer(directories, corpus, min_df=None):
    if not min_df:
        min_df=.05 # les termes gardés sont ceux qui apparaissent dans au moins 5% du corpus

    vectorizer = TfidfVectorizer(ngram_range=(1,2), strip_accents='unicode', min_df=min_df)
    vectors = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    df = pd.concat([df, pd.Series(directories, name='dir_id').astype(int)], axis=1).set_index('dir_id')
    return df


def parse_file(file_path):
    try:
        return textract.process(file_path), 'fr'

    except Exception as e:
        print("Error", e, file_path)
        return None, None


def parse_files(pdf_path, list_of_directories):
    pdf_skipped = 0
    pdf_parsed = 0

    directories = []
    documents = []

    language_count = {'en': 0, 'fr': 0}
    pdf_dir = os.listdir(pdf_path)
    
    for directory in tqdm(list_of_directories):
        if str(directory) != '.DS_Store' and str(directory) in pdf_dir:
            file_name = os.listdir(f'{pdf_path}/{directory}')[0]
            if (len(file_name) > 1): #  and file_name.endswith('pdf')
                file_path = f'{pdf_path}/{directory}/{file_name}'
                text, language = parse_file(file_path)
                if text and language:
                    language_count[language] += 1
                    directories.append(directory)
                    documents.append(text)
                    pdf_parsed = pdf_parsed + 1
                else:
                    pdf_skipped = pdf_skipped + 1
            else:
                pdf_skipped = pdf_skipped + 1
        else:
            pdf_skipped = pdf_skipped + 1

    print(language_count)
    print(f'pdf parsed: {pdf_parsed}/{len(list_of_directories)} ({pdf_skipped} skipped)')
    documents = list(map(lambda x: x.decode("utf-8"), documents))
    return directories, documents


def tokenize(text):
    processor = TextProcessor()
    language = detect(text)
    if language in ['fr','en']:
        return processor.full_pipeline(text, language)
    else:
        return processor.full_pipeline(text, 'fr')


def estimate_remaining_time(time_so_far, i):
    partitions_remaining = 100 - (i + 1)
    avg_time_per_partition = time_so_far / (i + 1)
    estimation = avg_time_per_partition * partitions_remaining
    print(f'{str(timedelta(seconds=estimation))}s remaining')


def cmd_parse_n_tokenize_file(file_path):
    with open(root_dir + "/data/directories.pkl", 'rb') as f: 
        directories = pickle.load(f)

    with open(root_dir + "/data/corpus.pkl", 'rb') as f: 
        corpus = pickle.load(f)

    with open(root_dir + '/data/tokens.pkl', 'rb') as f: 
        tokens = pickle.load(f)

    text, _ = parse_file(file_path)
    directories.append(os.path.basename(os.path.dirname(file_path)))
    corpus.append(text)
    tokens.append(tokenize(text))

    print(f'Saving directories at {root_dir + "/data/directories.pkl"}')
    with open(root_dir + "/data/directories.pkl", 'wb') as f: 
        directories = pickle.dump(directories, f)

    print(f'Saving corpus at {root_dir + "/data/corpus.pkl"}')
    with open(root_dir + "/data/corpus.pkl", 'wb') as f: 
        corpus = pickle.dump(corpus, f)

    print(f'Saving tokens at {root_dir + "/data/tokens.pkl"}')
    with open(root_dir + '/data/tokens.pkl', 'wb') as f: 
        tokens = pickle.dump(tokens, f)


def cmd_create_corpus(pdf_path):
    # parse pdf multi process
    start = timer()
    list_of_directories = os.listdir(pdf_path)
    print(f'Parsing every pdf file ({len(list_of_directories)}) using multiprocessing')
    n = int(len(list_of_directories) / os.cpu_count())
    partitions = [list_of_directories[i:i + n] for i in range(0, len(list_of_directories), n)]
    proc_pool = Pool(processes = os.cpu_count())
    with proc_pool as p: 
        results = p.map(partial(parse_files, pdf_path), partitions) 

    directories = []
    corpus = []
    for e0, e1 in results:
        directories.extend(e0)
        corpus.extend(e1)
    end = timer()
    total_time = round(end - start)
    print(f'{str(timedelta(seconds=total_time))}s elapsed')

    print(f'Saving directories at {root_dir + "/data/directories.pkl"}')
    with open(root_dir + "/data/directories.pkl", 'wb') as f: 
        directories = pickle.dump(directories, f)

    print(f'Saving corpus at {root_dir + "/data/directories.pkl"}')
    with open(root_dir + "/data/corpus.pkl", 'wb') as f: 
        corpus = pickle.dump(corpus, f)


def cmd_tokenize_corpus(corpus):
    # Tokenize
    start = timer()
    tokens = []
    n = int(len(corpus) / 100)
    partitions = [corpus[i:i + n] for i in range(0, len(corpus), n)]
    for i, part in enumerate(partitions):
        proc_pool = Pool(processes = os.cpu_count())
        with proc_pool as p: 
            tokens_part = p.map(tokenize, part)
        tokens.extend(tokens_part)
        print(f'{i}/100')
        int_timer = timer()
        time_so_far = round(int_timer - start)
        print(f'{str(timedelta(seconds=time_so_far))}s elapsed')
        estimate_remaining_time(time_so_far, i)

    end = timer()
    total_time = round(end - start)
    print(f'{str(timedelta(seconds=total_time))}s elapsed')

    print(f'Saving tokens at {root_dir + "/data/tokens.pkl"}')
    with open(root_dir + '/data/tokens.pkl', 'wb') as f: 
        tokens = pickle.dump(tokens, f)


def cmd_apply_tfidf():
    with open(root_dir + '/data/tokens.pkl', 'rb') as f: 
        tokens = pickle.load(f)
    
    with open(root_dir + "/data/directories.pkl", 'rb') as f: 
        directories = pickle.load(f)

    tfidf = apply_TfidfVectorizer(directories, list(map(lambda x: ' '.join(x), tokens)))
    tfidf.to_csv(root_dir + '/data/tfidf_corpus.csv')
