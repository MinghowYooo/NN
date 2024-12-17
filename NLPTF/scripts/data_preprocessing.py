import numpy as np
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch
from wordcloud import WordCloud, STOPWORDS
import warnings
import kagglehub
kritanjalijain_amazon_reviews_path = kagglehub.dataset_download('kritanjalijain/amazon-reviews')
print('Data source import complete.')
# Read the train.csv file into a pandas dataframe
df_train = pd.read_csv(kritanjalijain_amazon_reviews_path + '/train.csv')

# Read the test.csv file into a pandas dataframe
df_test = pd.read_csv(kritanjalijain_amazon_reviews_path + '/test.csv')
# read in the data and add headers.
df_train = pd.read_csv(kritanjalijain_amazon_reviews_path + '/train.csv', header=None, names=['polarity', 'title', 'text'])
df_test = pd.read_csv(kritanjalijain_amazon_reviews_path + '/train.csv', header=None, names=['polarity', 'title', 'text'])
# append both files
combined_df = pd.concat([df_train, df_test], ignore_index=True)

# drop duplicates just in case
combined_df.drop_duplicates(inplace=True)

# show the structure of the dataframe
combined_df.info()
# Get the count of the different values in the polarity column
polarity_counts = combined_df['polarity'].value_counts()

# Create the bar chart
plt.figure(figsize=(8,6))
polarity_counts.plot(kind='bar', color=['red', 'green'], edgecolor='black')

# Add labels and title
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.title('Breakdown of Polarity in Amazon Reviews')

# Add count on top of each bar
for index, value in enumerate(polarity_counts):
    plt.text(index, value, str(value), ha='center', va='bottom')

# Show the plot
plt.show()