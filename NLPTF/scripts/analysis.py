# Combine all review text into one string
all_text = ' '.join(subset_df['text'].values)
word_counts = Counter(all_text.split())

# Generate word cloud, removing stopwords
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(all_text)

# Display word cloud
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Review Text')
plt.show()

# Add a new column for the length of each review text
subset_df['Text Length'] = subset_df['text'].apply(len)

# Boxplot to show length of reviews by sentiment
plt.figure(figsize=(6, 4))
subset_df.boxplot(column='Text Length', by='Sentiment', grid=False, color='purple')
plt.title('Review Length by Sentiment')
plt.suptitle('')  # Removes the automatic "Boxplot grouped by" title
plt.ylabel('Text Length')
plt.xlabel('Sentiment')
plt.show()

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Convert the sentiment to numerical values for the plot
subset_df['Sentiment Numeric'] = subset_df['Sentiment'].apply(lambda x: 0 if x == 'NEGATIVE' else 1)

# Pairplot to visualize the relationship between polarity, confidence, and text length
sns.pairplot(subset_df[['polarity', 'Confidence', 'Text Length', 'Sentiment Numeric']],
             hue='Sentiment Numeric', palette={0: 'red', 1: 'green'}, diag_kind='kde')

plt.show()

# Compare polarity and sentiment columns
polarity_sentiment_match = (subset_df['polarity'] == subset_df['Sentiment'].apply(lambda x: 1 if x == 'NEGATIVE' else 2)).mean()

# Display match percentage
print(f"Polarity matches sentiment: {polarity_sentiment_match * 100:.2f}%")

# Create a mask for mismatches (rows where polarity doesn't match sentiment)
mismatch_mask = subset_df['polarity'] != subset_df['Sentiment'].apply(lambda x: 1 if x == 'NEGATIVE' else 2)

# Subset the mismatches
mismatches_df = subset_df[mismatch_mask]

mismatches_df.head(10)

# Filter the dataframe to only include rows where Confidence is above 0.90
high_confidence_df = mismatches_df[mismatches_df['Confidence'] > 0.90]

# Sort the filtered dataframe by Confidence in descending order
high_confidence_df_sorted = high_confidence_df.sort_values(by='Confidence', ascending=False)

# Print the number of rows that will be printed
print(f"Number of rows with Confidence above 0.90: {high_confidence_df_sorted.shape[0]}")

# Iterate over the sorted dataframe and print the relevant rows
for index, row in high_confidence_df_sorted.head(20).iterrows():
    print(f"Title: {row['title']}")
    print(f"Text: {row['text']}")
    print(f"Polarity: {row['polarity']}")
    print(f"Predicted Sentiment: {row['Sentiment']}")
    print(f"Confidence: {row['Confidence']}")
    print("-" * 50)
