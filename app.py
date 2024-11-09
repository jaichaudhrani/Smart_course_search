import streamlit as st
import pandas as pd
from embedding_model import generate_embeddings
from search import perform_search

@st.cache_data
def load_data():
    return pd.read_csv('./free_courses.csv')

st.title('Smart Search: Free Courses on Analytics Vidhya')

course_data = load_data()

query = st.text_input('Search for courses:', '')

if query:
    query_embedding = generate_embeddings(query)

    results = perform_search(query_embedding, course_data)
    
    st.write('Search Results:')
    for index, row in results.iterrows():
        st.write(f"**{row['Title']}**")
        st.write(f"Price: {row['Price']}")
        st.write(f"Lessons: {row['Lessons']}")
        st.write(f"[Course Link]({row['Course URL']})")
        st.image(row['Image URL'])