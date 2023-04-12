import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import os
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import io

# Load the pre-trained model from a .pkl file
file_path = os.path.abspath('sentiment_model.pkl')
model = joblib.load(file_path)


# Define a function to perform sentiment analysis on a given review text
def predict_sentiment(review):
    # Preprocess the review text (e.g., remove stop words, punctuation, etc.)
    preprocessed_review = preprocess_review(review)
    # Use the pre-trained model to predict the sentiment of the review
    sentiment = model.predict([preprocessed_review])[0]
    # Return the predicted sentiment as a string ('positive' or 'negative')
    return 'positive' if sentiment == 1 else 'negative'


# Define a function to preprocess a given review text
def preprocess_review(review):
    # Add your preprocessing steps here (e.g., remove stop words, punctuation, etc.)
    return review


# Load the Amazon product review dataset
file_path = os.path.abspath('sentiment_with_results.csv')
data = pd.read_csv(file_path)

# Define the layout of the Streamlit app
st.title('Sentiment Analysis on Amazon Product Review')
review = st.text_input('Enter a Review')
if st.button('Analyze'):
    sentiment = predict_sentiment(review)
    st.write('Sentiment:', sentiment)
    sentiment_counts = data['result'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_title('Sentiment Distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    # sentiment_pie = sentiment_counts.plot.pie(title='Sentiment Distribution')
    # fig = sentiment_pie.get_figure()
    # fig.savefig("sentiment_pie.png")
    # st.image("sentiment_pie.png")

# Sidebar filters
st.sidebar.title("Filters")
companies = st.sidebar.multiselect("Select Companies", data['Company'].unique())
products = st.sidebar.multiselect("Select Products", data['Product'].unique())
results = st.sidebar.multiselect("Select Results", data['result'].unique())

# Apply filters
if companies and not products:
    filtered_data = data[data['Company'].isin(companies)]
elif products and not companies:
    filtered_data = data[data['Product'].isin(products)]
elif companies and products:
    filtered_data = data[(data['Company'].isin(companies)) & (data['Product'].isin(products))]
else:
    filtered_data = data

if results:
    filtered_data = filtered_data[filtered_data['result'].isin(results)]

# Show filtered data
if not filtered_data.empty:
    st.title("Filtered Data")
    columns_to_show = ["id", "Company", "Product", "Review", "result"]
    st.write(filtered_data[columns_to_show])
else:
    st.warning("No data available with selected filters.")

# KPIs
total_reviews = filtered_data.shape[0]
positive_reviews = filtered_data[filtered_data['result'] == 'positive'].shape[0]
negative_reviews = filtered_data[filtered_data['result'] == 'negative'].shape[0]

if total_reviews == 0:
    positive_percentage = 0
    negative_percentage = 0
else:
    positive_percentage = round((positive_reviews / total_reviews) * 100, 2)
    negative_percentage = round((negative_reviews / total_reviews) * 100, 2)

# Show KPIs
st.title("Key Performance Indicators")
st.write(f"Total Reviews: **{total_reviews}**")
st.write(f"Positive Reviews: **{positive_reviews}** ({positive_percentage}%)")
st.write(f"Negative Reviews: **{negative_reviews}** ({negative_percentage}%)")
# Create a string of all the reviews
all_reviews = ' '.join(filtered_data['Review'])

# Generate the word cloud
st.title("WorldCloud")
wordcloud = WordCloud(width=800, height=400).generate(all_reviews)

# Display the word cloud
fig, ax = plt.subplots(figsize=(16, 8))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')

# Save the figure to a BytesIO object and display it in Streamlit

buf = io.BytesIO()
fig.savefig(buf, format='png')
buf.seek(0)
st.image(buf)

# Visualization
if not filtered_data.empty:
    st.title("Visualization")
    chart_type = st.selectbox("Select Chart Type", ["Pie Chart", "Bar Chart"])

    if chart_type == "Pie Chart":
        fig = px.pie(filtered_data, values='Rating', names='result', title='Sentiment Distribution')
        st.plotly_chart(fig)

    elif chart_type == "Bar Chart":
        if len(products) == 1:
            fig = px.bar(filtered_data, x='Company', y='Rating', color='result', barmode='group',
                         title='Sentiment by Company')
            st.plotly_chart(fig)
        else:
            fig = px.bar(filtered_data, x='Product', y='Rating', color='result', barmode='group',
                         title='Sentiment by Product')
            st.plotly_chart(fig)
