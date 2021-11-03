import spacy
import nltk
import re
import string
import os

from stop_words import get_stop_words

script_path = os.path.dirname(__file__)


class TextProcessor:
    nlp_en = spacy.load("en_core_web_sm")
    nlp_fr = spacy.load("fr_core_news_md")

    fr_stop_words = get_stop_words('fr') + ['expérience', 'professionnel', 'professionnelle', 'stage',
                                            'université', 'master', 'licence', 'lycée', 'classe', 'préparatoire',
                                            'centre', 'ecole', 'école', 'baccalauréat',
                                            'maternelle',
                                            'www', 'https', 'http',
                                            'rue']
    en_stop_words = get_stop_words('en') + ['university', 'www', 'https', 'http']
    fr_months = ['mois', 'janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre',
                 'octobre', 'novembre', 'décembre']

    def tokenize_and_clean(self, document, language) -> list:
        """Return a list of tokens from an article. Punctuation, email addresses and digits are removed."""
        # Remove email addresses
        document = re.sub(r"\b[^\s]+@[^\s]+[.][^\s]+\b", " ", document)
        # Remove ponctuation
        document = document.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        # Tokenisation
        tokens = document.split()
        # Cleaning from stop words, capital letters and numericals
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token.isalpha()]
        tokens = self.remove_stop_words(tokens, language)
        return tokens

    def remove_stop_words(self, tokens, language):
        if language == 'en':
            stop_words = self.en_stop_words
        else:
            stop_words = self.fr_stop_words
        tokens = [token for token in tokens if token not in stop_words]
        return tokens

    def remove_named_entities(self, tokens, language):
        tokens_without_entities = []

        if language == 'en':
            for token in tokens:
                named_entity_en = self.nlp_en(token)
                if named_entity_en[0].ent_type_ not in ['DATE', 'GPE', 'LOC']:
                    tokens_without_entities.append(token)
        if language == 'fr':
            for token in tokens:
                named_entity_fr = self.nlp_fr(token)
                if (named_entity_fr[0].ent_type_ not in ['LOC']) and (named_entity_fr[0].text not in self.fr_months):
                    tokens_without_entities.append(token)
        return tokens_without_entities

    def POS_tagging(self, tokens: list, language: str) -> list:
        """Return a list of tagged tokens for each cv."""
        tagged = []
        if language == 'en':
            tagged = nltk.pos_tag(tokens)
        if language == 'fr':
            tagged = [(token.text, token.pos_) for token in self.nlp_fr(' '.join(tokens))]  # Spacy POS Tagger
        return tagged

    def tag_selection(self, text_tagging: list, language: str) -> list:
        """Return a list of final tokens for each article. Only nouns are kept."""
        # On rajoute les adjectifs parce que c'est important dans le cas par exemple d'intelligence artificielle
        if language == "en":
            kept_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'VBG']
        elif language == "fr":
            kept_tags = ['PROPN', 'X', 'NOUN', 'VERB', 'ADJ']
        else:
            raise Exception('Unrecognized language')

        tokens = []
        for j in range(len(text_tagging)):
            if text_tagging[j][1] in kept_tags:
                tokens.append(text_tagging[j][0])
        return tokens

    def lemmatize(self, tokens, language) -> list:
        if language == 'en':
            return [self.nlp_en(token)[0].lemma_ for token in tokens]
        elif language == 'fr':
            return [self.nlp_fr(token)[0].lemma_ for token in tokens]

    def full_pipeline(self, document, language):
        tokens = self.tokenize_and_clean(document, language)
        filtered_tokens = self.remove_named_entities(tokens, language)
        pos_tagged_tokens = self.POS_tagging(filtered_tokens, language)
        selected_tagged_tokens = self.tag_selection(pos_tagged_tokens, language)
        lemmas = self.lemmatize(selected_tagged_tokens, language)
        return lemmas
