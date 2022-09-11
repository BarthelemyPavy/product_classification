#! /bin/bash
# add custom post-install commands below
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet
python -m nltk.downloader averaged_perceptron_tagger
python -m nltk.downloader omw-1.4
python -m spacy download en
