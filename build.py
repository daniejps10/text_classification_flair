# Based on: How to Create an NLP Application using Flair
# Link: https://www.section.io/engineering-education/how-to-create-nlp-application-with-flair/

#Using a Pre-trained Classification Model
import constants
import numpy as np
import pandas as pd

#Read data
print(constants.LINE)
print(constants.SUB_LINE)
print('Read SPAM file...')
data_df = pd.read_csv(constants.INPUT_FILE).sample(frac=1).drop_duplicates()
print(data_df.head())

# Checking value counts
print(constants.SUB_LINE)
print('Checking value counts...')
print(data_df['class'].value_counts())
print('Show columns...')
print(data_df.columns)

# Strip: object column data
print(constants.SUB_LINE)
print('Strip: object columns dataframe...')
data_df = data_df.select_dtypes(['object'])
data_df[data_df.columns] = data_df.apply(lambda x: x.str.strip())

# Remove null data
print(constants.SUB_LINE)
print('Remove null data...')
data_df['clean_tweet'].replace('', np.nan, inplace=True)
data_df.dropna(subset=['clean_tweet'], inplace=True)

# Select two columns: clean_tweet, labels
print(constants.SUB_LINE)
print('Selecting two columns...')
data_df = data_df[['clean_tweet','labels']]

#Rename columns
print(constants.SUB_LINE)
print('Rename columns...')
data_df = data_df.rename(columns={"clean_tweet":"text", "labels":"label"})
#data_df['label'] = '__label__' + data_df['label'].astype(str)
print(data_df.head())

#Split into train, test, dev data
print(constants.SUB_LINE)
print('Split into train, test, dev data...')
train_df, test_df, dev_df = np.split(data_df,[int(.6*len(data_df)),int(.8*len(data_df))])
print(data_df.shape)
print(train_df.shape)
print(test_df.shape)
print(dev_df.shape)
train_df.to_csv(constants.INPUT_FOLDER_FILE.format('train.csv'))
test_df.to_csv(constants.INPUT_FOLDER_FILE.format('test.csv'))
dev_df.to_csv(constants.INPUT_FOLDER_FILE.format('dev.csv'))
