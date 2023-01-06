#Base on: Text Classification with State of the Art NLP Library — Flair
#Link: https://towardsdatascience.com/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f

#Using a Pre-trained Classification Model
#Sentiment analysis model trained on the IMDB dataset
from flair.models import TextClassifier
from flair.data import Sentence

import constants
import pandas as pd

#Read data
print(constants.LINE)
print(constants.SUB_LINE)
print('Read SPAM file...')
data = pd.read_csv("spam.csv", encoding='latin-1').sample(frac=1).drop_duplicates()
print(data.head())

#Rename columns
print(constants.SUB_LINE)
print('Rename columns...')
data = data[['v1', 'v2']].rename(columns={"v1":"label", "v2":"text"})
data['label'] = '__label__' + data['label'].astype(str)
print(data.head())

#Split into train, test, dev data
print(constants.SUB_LINE)
print('Split into train, test, dev data...')
data.iloc[0:int(len(data)*0.8)].to_csv('train.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('test.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.9):].to_csv('dev.csv', sep='\t', index = False, header = False)

