import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from PIL import Image


# Initialize the PorterStemmer
ps = PorterStemmer()

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load and display the image
image = Image.open('spam-filter.png')  # Ensure this image is in the same directory or provide the full path
st.image(image, caption='EMAIL Spam Filter')

# Text transformation function
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize text
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    
    # Perform stemming
    text = [ps.stem(i) for i in text]
    
    return " ".join(text)

# Load the vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error("Required files not found. Ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")

# Streamlit UI
st.title('Email Spam Classifier')

# User input fields
input_sms = st.text_input('Enter the Message ')
option = st.selectbox("You Got Message From:", ["Via Email", "Via SMS", "Other"])

# Button for prediction
if st.button('Click to Predict'):
    if input_sms:
        try:
            # Transform and vectorize input
            transform_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transform_sms])
            result = model.predict(vector_input)[0]

            # Display result
            if result == 1:
                st.header("Spam")
            else:
                st.header('Not Spam')
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a message to classify.")
