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
import folium
from streamlit_folium import folium_static
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

def get_selected_navbar_menu():
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

    return selected_navbar_menu

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

def plot_n_grams(df):
    data = df['text'].apply(lambda x: str(x).split())
    top = Counter([item for sublist in data for item in sublist])
    top_ngrams = pd.DataFrame(top.most_common(25))
    top_ngrams.columns = ['Words', 'Count']
    st.dataframe(top_ngrams.style.background_gradient(cmap='Oranges'))

    with st.expander('View most common words in more detail here'):
        words = top_ngrams['Words']
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
            title='Top N-Grams',
            xaxis=dict(title='Count'),
            yaxis=dict(title='Words'),
            height=750
        )

        fig.update_yaxes(autorange='reversed')

        # Render the plotly figure using Streamlit
        st.plotly_chart(fig, use_container_width=True)

def display_home():
    st.header('Unleash the Power of Data with Our PDF Scraping Solutions')
    st.markdown('<br>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write('Take your data analysis to the next level with our cutting-edge PDF scraping technology, and start your free trial today to experience the power of actionable insights at your fingertips.')
        st.write('With our PDF scraping solutions, you can unlock valuable data hidden within PDF files and leverage it for in-depth analysis. Whether you need to extract textual information, perform sentiment analysis, or uncover trends and patterns, our cutting-edge technology has got you covered. By automating the extraction process, we save you valuable time and effort, allowing you to focus on deriving actionable insights from your data.')
    with col2:
        st.image('home-image.png', use_column_width=True)

    st.markdown('<br><br><br><br>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align:center">Why Choose Us?</h3>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<h4>99.9%</h4>', unsafe_allow_html=True)
        st.write('Uptime over the last 3 months.')
    with col2:
        st.markdown('<h4>2 mins.</h4>', unsafe_allow_html=True)
        st.write('Help response averages.')
    with col3:
        st.markdown('<h4>+100K</h4>', unsafe_allow_html=True)
        st.write('PDF scraped monthly.')
    with col4:
        st.markdown('<h4>>90%</h4>', unsafe_allow_html=True)
        st.write('Customer are happy with the results.')

def display_product():
    uploaded_file = st.file_uploader('Choose a PDF file', accept_multiple_files = False)
    st.caption('This will be counted as 1 quota for today.')
    if uploaded_file is not None:
        # quota_used += 1
        bytes_data = uploaded_file.read()
        filename = uploaded_file.name

        folderpath = 'uploaded_files'
        filepath = os.path.join(folderpath, filename)
        with open(filepath, "wb") as file:
            file.write(bytes_data)

        text_list = scrape_pdf(filepath)
        df = pd.DataFrame({'text': text_list})
        csv_filename = f'{os.path.splitext(filename)[0]}.csv'
        csv_data = df.to_csv(index = False).encode('utf-8')

        with st.expander('View CSV file'):
            st.dataframe(df, use_container_width=True)
            st.download_button(label = 'Download as CSV File',
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

        # n = st.slider('Number of Words', min_value = 1, max_value = 4, value = 1)
        # st.write('n: {}'.format(n))
        # st.write(type(n))
        plot_n_grams(df)


def display_pricing():
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.expander('Free Version'):
            st.header('Rp 0')
            st.write('✔️  3 Files / Day')
            st.write('✔️  Word Cloud')
            st.write('✔️  N-Grams')
            st.markdown('<div style="height: 180px;">&nbsp;</div>', unsafe_allow_html=True)
    with col2:
        with st.expander('Premium'):
            st.header('Rp 5k')
            st.write('✔️  Unlimited Files to Scrape')
            st.write('✔️  Word Cloud')
            st.write('✔️  N-Grams')
            st.write('✔️  Semantic Analysis')
            premium_button = st.button('Subscribe Premium Plan')
            st.markdown('<div style="height: 87px;">&nbsp;</div>', unsafe_allow_html=True)
    with col3:
        with st.expander('Enterprise'):
            st.header('Rp 15k')
            st.write('✔️  Unlimited Files to Scrape')
            st.write('✔️  Word Cloud')
            st.write('✔️  N-Grams')
            st.write('✔️  Topic Modeling')
            st.write('✔️  Cluster Analysis')
            enterprise_button = st.button('Subscribe Enterprise Plan')
            st.markdown('<div style="height: 45px;">&nbsp;</div>', unsafe_allow_html=True)

def display_contact_us():
    st.write('📲 ┆ +62 895 3391 31039')
    st.write('✉️ ┆ pdfinsight@gmail.com')
    st.write('🏠 ┆ PDFInsight Headquarter, Jl. Jenderal Sudirman No.Kav. 1, RT.1/RW.8, Karet Tengsin, Kecamatan Tanah Abang, Kota Jakarta Pusat, Daerah Khusus Ibukota Jakarta 10220')

def render_html(filename):
    with open(filename, 'r') as f:
        html_content = f.read()
        st.components.v1.html(html_content, height=1000)

def render_css(filename):
    with open(filename, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    set_page_configuration()
    display_introduction()

    render_css('css/style.css')

    selected_navbar_menu = get_selected_navbar_menu()
    if selected_navbar_menu == 'Home':
        display_home()
    if selected_navbar_menu == 'Product':
        display_product()
    if selected_navbar_menu == 'Pricing':
        display_pricing()
    if selected_navbar_menu == 'Contact Us':
        display_contact_us()

if __name__ == '__main__':
    main()
