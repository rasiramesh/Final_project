import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt   

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize Streamlit app
st.title("NLP Text Processing Application")

# Text input
user_input = st.text_area("Enter your text here")

if user_input:
    # Tokenization
    tokens = word_tokenize(user_input)
    st.subheader("Tokenization")
    st.write(tokens)

    # Removing special characters
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
    st.subheader("After Removing Special Characters")
    st.write(tokens)

    # Removing Stop Words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    st.subheader("After Removing Stop Words")
    st.write(filtered_tokens)

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    st.subheader("Stemming")
    st.write(stemmed_tokens)

    # Keyword Extraction (based on frequency)
    st.subheader("Keyword Extraction")
    keyword_freq = Counter(stemmed_tokens)
    keywords = keyword_freq.most_common(5)
    st.write(keywords)

    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(user_input)
    st.subheader("Sentiment Analysis")
    st.write(sentiment)
    # Word Cloud
    wordcloud = WordCloud(width = 800, height = 800, 
                          background_color ='white', 
                          stopwords = stop_words, 
                          min_font_size = 10).generate(user_input)
    st.subheader("Word Cloud")

# Create a figure for the word cloud
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')

# Use Streamlit's function to display the figure
st.pyplot(fig)

# Display the words and their frequencies
st.write(wordcloud.words_)
    
  

   
    

# Run this script using: streamlit run your_script_name.py
