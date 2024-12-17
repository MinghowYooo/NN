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

# Confidence Score Distribution
plt.figure(figsize=(6, 4))
subset_df['Confidence'].plot(kind='hist', bins=5, color='skyblue')
plt.title('Confidence Scores Distribution')
plt.xlabel('Confidence')
plt.show()

