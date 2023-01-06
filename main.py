#Base on: Text Classification with State of the Art NLP Library — Flair
#Link: https://towardsdatascience.com/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f

#Using a Pre-trained Classification Model
#Sentiment analysis model trained on the IMDB dataset
from flair.models import TextClassifier
from flair.data import Sentence

classifier = TextClassifier.load('en-sentiment')

sentence = Sentence('Flair is pretty neat!')
classifier.predict(sentence)

#Print sentence with predicted labels
print('Sentence above is: ', sentence.labels)


