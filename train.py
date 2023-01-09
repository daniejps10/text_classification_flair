
from flair.datasets import CSVClassificationCorpus, ClassificationCorpus
from flair.data import Corpus

# Working with the Word Embeddings
from flair.embeddings import FlairEmbeddings,WordEmbeddings,DocumentLSTMEmbeddings

# Load NLP Pkgs: Build and train classifier
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

import constants

# https://blog.jcharistech.com/2020/10/04/text-classification-with-flair-pytorch-nlp-framework/
# https://www.section.io/engineering-education/how-to-create-nlp-application-with-flair/

print(constants.LINE)
print(constants.SUB_LINE)
print("Mapping columns...")
column_name_map = {2:"label_topic",1:"text"}

#Create Corpus For CSV
print(constants.SUB_LINE)
print("Read and create CORPUS...")
corpus_csv: Corpus = CSVClassificationCorpus(constants.INPUT_FOLDER,\
   column_name_map=column_name_map, skip_header=True, delimiter=',', label_type="label_topic")
print(corpus_csv)
# corpus_fst: Corpus = ClassificationCorpus(constants.INPUT_FOLDER)

#Creating the labeling dictionary
print(constants.SUB_LINE)
print("Create the labeling dictionarys...")
label_dict_csv = corpus_csv.make_label_dictionary(label_type="label_topic")

#Create our WEmbeddings
print(constants.SUB_LINE)
print("Create our WEmbeddings...")
word_embeddings = [FlairEmbeddings('news-forward-fast'),FlairEmbeddings('news-backward-fast')]

# Document Embeddings
print(constants.SUB_LINE)
print("Create our DocumentsEmbeddings...")
document_embeddings = DocumentLSTMEmbeddings(word_embeddings, \
   hidden_size=512, reproject_words=True, reproject_words_dimension=256)

# Text classifier
print(constants.SUB_LINE)
print("Text classifier...")
clf = TextClassifier(document_embeddings,label_dictionary=label_dict_csv, label_type="label_topic")

# Training model
print(constants.SUB_LINE)
print("Training model...")
trainer = ModelTrainer(clf,corpus_csv)
trainer.train(constants.INPUT_FOLDER, max_epochs=8)

