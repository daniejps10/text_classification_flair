
from flair.datasets import CSVClassificationCorpus, ClassificationCorpus
from flair.data import Corpus

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
# corpus_fst: Corpus = ClassificationCorpus(constants.INPUT_FOLDER)

#Creating the labeling dictionary
print(constants.SUB_LINE)
print("Create the labeling dictionarys...")
label_dict_csv = corpus_csv.make_label_dictionary(label_type="label_topic")