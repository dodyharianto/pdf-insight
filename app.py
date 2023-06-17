import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import datetime
import os
import re
from PyPDF2 import PdfReader, PdfFileReader
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
plt.style.use('fivethirtyeight')
nltk.download('stopwords')

def set_page_configuration():
    st.set_page_config(page_title = 'PDF Insight | Home',
                     page_icon = '📚',
                     layout = 'wide',
                     initial_sidebar_state = 'expanded')

def display_introduction():
    st.title('PDF Insight 📚')
    st.write('Get the summary of your long PDF files in no time.')

def remove_stopwords(text):
    factory = StopWordRemoverFactory()

    indonesian_stopwords = factory.get_stop_words()
    english_stopwords = stopwords.words('english')

    # Add extra stopwords
    indonesian_custom_stopwords = ['tak', 'enggak', 'lalu', 'sekarang', 'nanti', 'tersebut', 'kata', 'juga', 'jadi']
    english_custom_stopwords = ['the', 'and', 'we', 'at']

    # Combine all stopwords
    all_stopwords = indonesian_stopwords + english_stopwords + indonesian_custom_stopwords + english_custom_stopwords

    # Filter the tokenized words from the stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in all_stopwords and len(token) > 2]
    print('TOKENS: {}'.format(tokens))
    return ' '.join(tokens)

def scrape_pdf(filepath):
    reader = PdfReader(filepath)
    number_of_pages = len(reader.pages)
    text_list = []
    for i in range(number_of_pages):
        page = reader.pages[i]
        text = page.extract_text().lower()
        cleaned_text = ' '.join(re.findall('[a-zA-Z]+', text))
        filtered_text = remove_stopwords(cleaned_text)
        text_list.append(filtered_text)
    return text_list

def generate_word_cloud(df):
    plt.figure(figsize = (10, 8))
    wordcloud = WordCloud(width = 3000,
                          height = 2000,
                          random_state = 1,
                          background_color = 'black',
                          colormap = 'Wistia',
                          collocations = False,
                          stopwords = STOPWORDS).generate(' '.join(df['text']))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

def plot_n_grams(df, n):
    # Combine all text into a single string
    text = ' '.join(df['text'])

    # Generate n-grams using the specified value of n
    ngrams = nltk.ngrams(text.split(), n)
    ngram_counts = Counter(ngrams)

    # Create a dataframe with n-gram counts
    top_ngrams = pd.DataFrame(ngram_counts.most_common(25), columns=['Words', 'Count'])
    st.dataframe(top_ngrams.style.background_gradient(cmap='Oranges'))

    with st.expander('View the bar chart'):
        words = [' '.join(words) for words, _ in top_ngrams['Words']]
        counts = top_ngrams['Count']

        fig = go.Figure(data=go.Bar(
            x=counts,
            y=words,
            marker=dict(
                color=counts,
                colorbar=dict(title='Count'),
                colorscale='YlOrRd',
            ),
            orientation='h'
        ))

        fig.update_layout(
            title=f'Top {n}-Grams',
            xaxis=dict(title='Count'),
            yaxis=dict(title='Words'),
            height=800
        )

        fig.update_yaxes(autorange='reversed')
        st.plotly_chart(fig, use_container_width=True)


# def plot_n_grams(df):
#     data = df['text'].apply(lambda x: str(x).split())
#     top = Counter([item for sublist in data for item in sublist])
#     top_ngrams = pd.DataFrame(top.most_common(25))
#     top_ngrams.columns = ['Words', 'Count']
#     st.dataframe(top_ngrams.style.background_gradient(cmap='Oranges'))

#     with st.expander('View the bar chart'):
#         words = top_ngrams['Words']
#         counts = top_ngrams['Count']

#         fig = go.Figure(data=go.Bar(
#             x=counts,
#             y=words,
#             marker=dict(
#                 color=counts,
#                 colorbar=dict(title='Count'),
#                 colorscale='YlOrRd',
#             ),
#             orientation='h'
#         ))

#         fig.update_layout(
#             title='Top N-Grams',
#             xaxis=dict(title='Count'),
#             yaxis=dict(title='Words'),
#             height = 800
#         )

#         fig.update_yaxes(autorange = 'reversed')
#         st.plotly_chart(fig, use_container_width=True)

def display_product():
    uploaded_file = st.file_uploader('Choose a PDF file', accept_multiple_files = False)
    st.caption('This will be counted as 1 quota for today.')
    if uploaded_file is not None:
        # quota_used += 1
        bytes_data = uploaded_file.read()
        filename = uploaded_file.name

        folderpath = 'C:/Users/acer/Documents/SEMESTER 6 COURSES/SOFTWARE ENGINEERING/Project - PDF Scraper - Copy/uploaded_files'
        filepath = os.path.join(folderpath, filename)
        with open(filepath, "wb") as file:
            file.write(bytes_data)

        text_list = scrape_pdf(filepath)
        df = pd.DataFrame({'text': text_list})
        csv_filename = f'{os.path.splitext(filename)[0]}.csv'
        csv_data = df.to_csv(index = False).encode('utf-8')
        st.dataframe(df, use_container_width=True)

        st.download_button(
            label = 'Download as CSV File',
            data = csv_data,
            file_name = csv_filename,
            mime = 'text/csv')

        st.markdown('<hr>', unsafe_allow_html=True)
        st.success('Analyzing your PDF file: {}'.format(uploaded_file.name))
        st.write('This may take a few minutes...')
        st.markdown('<hr>', unsafe_allow_html=True)

        generate_word_cloud(df)
        st.pyplot(plt.gcf())
        st.markdown('<hr>', unsafe_allow_html=True)

        n = st.slider('Number of Words', min_value = 1, max_value = 4, value = 1)
        plot_n_grams(df, n)


def display_pricing():
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.expander('Free'):
            st.write('Free Version')
    with col2:
        with st.expander('Premium'):
            st.write('Premium Version')
    with col3:
        with st.expander('Enterprise'):
            st.write('Enterprise Version')

def display_contact_us():
    selected_contact = option_menu(
        menu_title = None, # required
        options = ['Phone', 'Email', 'Address'],
        icons = ['phone', 'mail', 'address'],
        menu_icon = 'cast',
        default_index = 0,
        orientation = 'vertical',
        styles = {'nav-link': {'--hover-color': '#fc6b03',
                               '--active-background-color': '#eee'},
        'nav-link-selected': {'background-color': '#eb3502'},
        })

def render_html(filename):
    with open(filename) as f:
        st.markdown(f'{f.read()}', unsafe_allow_html=True)

def render_css(filename):
    with open(filename) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    set_page_configuration()
    display_introduction()

    render_html('html/index.html')
    render_css('css/style.css')

    selected_navbar_menu = option_menu(
        menu_title = None, # required
        options = ['Home', 'Product', 'Pricing', 'Contact Us'],
        icons = ['house', 'bag', 'currency-dollar', 'telephone'],
        menu_icon = 'cast',
        default_index = 0,
        orientation = 'horizontal',
        styles = {'nav-link': {'--hover-color': '#fc6b03',
                               '--active-background-color': '#eee'},
        'nav-link-selected': {'background-color': '#eb3502'},
        })

    if selected_navbar_menu == 'Home':
        st.write('home')
    if selected_navbar_menu == 'Product':
        display_product()
    if selected_navbar_menu == 'Pricing':
        display_pricing()
    if selected_navbar_menu == 'Contact Us':
        display_contact_us()

if __name__ == '__main__':
    main()
