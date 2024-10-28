import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from streamlit_option_menu import option_menu
from scipy.cluster.hierarchy import dendrogram, linkage
from groq import Groq
from sklearn.metrics import silhouette_score, davies_bouldin_score


# Function to create TF-IDF matrix
def create_tfidf_matrix(data):
    # Fill NaN values with an empty string
    data['product_name'] = data['product_name'].fillna('')
    data['description'] = data['description'].fillna('')
    vectorizer = TfidfVectorizer()
    documents = data['product_name'] + " " + data['description']
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer

# Function for K-Means Clustering
def kmeans_clustering(tfidf_matrix, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(tfidf_matrix)
    return kmeans.labels_, kmeans.cluster_centers_

# Function for Agglomerative Clustering
def agglomerative_clustering(tfidf_matrix, num_clusters):
    agglo = AgglomerativeClustering(n_clusters=num_clusters)
    labels = agglo.fit_predict(tfidf_matrix.toarray())  # Requires dense input
    return labels


# Function for DBSCAN Clustering
def dbscan_clustering(tfidf_matrix, eps_value, min_samples_value):
    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
    labels = dbscan.fit_predict(tfidf_matrix.toarray())  # Requires dense input
    return labels

# Function to get top N products based on cosine similarity
def get_top_products(tfidf_matrix, vectorizer, query, data, top_n=10):
    input_tfidf = vectorizer.transform([query])
    similarities = cosine_similarity(input_tfidf, tfidf_matrix).flatten()
    
    # Get the indices of the top N products
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Display top products
    st.write(f"### Top {top_n} Products based on Similarity Score:")
    for idx in top_indices:
        product = data.iloc[idx]
        st.write(f"**Product Name:** {product['product_name']}")
        st.write(f"**Description:** {product['description']}")
        st.write(f"**Product Link:** {product['product_url']}")
        st.write(f"**OG PRICE:** {product['retail_price']}")
        st.write(f"**Discount Price:** {product['discounted_price']}")
        st.write(f"**Similarity Score:** {similarities[idx]:.4f}")
        st.write("------")

# Function for plotting clustering results with t-SNE
def plot_clusters(tfidf_matrix, labels):
    plt.figure(figsize=(10, 7))
    tsne_data = tfidf_matrix.toarray()  # Convert sparse matrix to dense for plotting
    tsne = TSNE(n_components=2, random_state=0)
    reduced_data = tsne.fit_transform(tsne_data)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, s=50, cmap='viridis')
    plt.title('Clustering Results with t-SNE')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    st.pyplot(plt)

# Function for Boolean retrieval with AND/OR logic
def boolean_retrieval(data, query):
    if ' AND ' in query:
        terms = query.split(' AND ')
        boolean_results = [
            entry for entry in data.to_dict('records') 
            if all(term.lower() in entry['product_name'].lower() or term.lower() in entry['description'].lower() 
                for term in terms if pd.notna(entry['product_name']) and pd.notna(entry['description']))
        ]
    elif ' OR ' in query:
        terms = query.split(' OR ')
        boolean_results = [
            entry for entry in data.to_dict('records') 
            if any(term.lower() in entry['product_name'].lower() or term.lower() in entry['description'].lower() 
                for term in terms if pd.notna(entry['product_name']) and pd.notna(entry['description']))
        ]
    else:
        boolean_results = [
            entry for entry in data.to_dict('records')
            if query.lower() in entry['product_name'].lower() or query.lower() in entry['description'].lower()
        ]
    return boolean_results

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Function for AI Chatbot
def ai_chatbot(user_input):
    character_description = """You are a sporty and practical expert in women's athletic and casual clothing, especially focused on comfort and functionality, like cycling shorts. You’re approachable and have a keen eye for fashion that prioritizes comfort and versatility. You break down the features of products with ease, offering clear and detailed advice on fabric types, patterns, and care instructions. You enjoy helping others make smart choices by highlighting essential details, such as fabric blends like "Cotton Lycra," practical washing tips, and the benefits of versatile styles like solid-colored shorts that come in convenient packs of three."""
    
    full_input = f"{character_description}\nUser: {user_input}\nAssistant:"

    client = Groq(api_key="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")
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": full_input}],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""

    return response

# Streamlit App
st.title("Product Search on Flipkart and Chatbot App")

with st.sidebar:
    page = option_menu("Flipkart Product Search & AI Chatbot",
                       ["Home", "Boolean Retrieval", "Clustering Analysis", "Product Similarity", "Ai Chatbot"],
                       icons=['house', 'bar-chart', 'graph-up-arrow', 'info-circle', 'file-text', 'link'],
                       menu_icon="cast", default_index=0)

# Initialize a variable for uploaded data
data = None

# Home Page
if page == "Home":
    st.write("## About the Project")
    st.write("""
    This application is designed to assist users in finding products based on various criteria,
    including descriptions, similarity, and clustering analysis. The project leverages 
    natural language processing techniques to provide meaningful insights into product data.
    """)

    st.write("### Key Features:")
    st.write("""
    - **Boolean Retrieval**: Search for products using Boolean queries to find related items.
    - **Clustering Analysis**: Explore clustering techniques (K-Means, Agglomerative, DBSCAN) to understand product groupings.
    - **Product Similarity**: Find similar products based on descriptions using cosine similarity.
    - **AI Chatbot**: Interact with an AI chatbot to get product recommendations and answers to your queries.
    """)

    st.write("### Technologies Used:")
    st.write("""
    - Streamlit for creating the web application.
    - Pandas for data manipulation and analysis.
    - Scikit-learn for machine learning and clustering.
    - Groq API for AI-driven responses.
    """)

    st.write("### How to Use:")
    st.write("""
    1. Upload your product CSV file in the respective sections.
    2. Use the features provided to search for products, analyze clusters, or interact with the AI chatbot.
    """)

# Boolean Retrieval Section
elif page == "Boolean Retrieval":
    st.write("### Step 1: Find Products using Boolean Operations")
    uploaded_file_boolean = st.file_uploader("Upload your CSV file for Boolean Retrieval", type="csv")

    if uploaded_file_boolean is not None:
        data_boolean = pd.read_csv(uploaded_file_boolean)
        
        # Check for required columns
        if 'product_name' not in data_boolean.columns or 'description' not in data_boolean.columns:
            st.error("CSV file must contain 'product_name' and 'description' columns.")
        else:
            st.write("### Data Preview:")
            st.dataframe(data_boolean.head())

            # Ensure 'title' and 'description' columns are strings
            data_boolean['product_name'] = data_boolean['product_name'].astype(str)
            data_boolean['description'] = data_boolean['description'].astype(str)

            query = st.text_area("Enter product description to find related products:")

            if st.button("Search"):
                boolean_results = boolean_retrieval(data_boolean, query)

                if boolean_results:
                    st.write("### Search Results:")
                    for result in boolean_results:
                        st.image(result['image_links'], width=100)
                        st.write(f"**Title:** {result['product_name']}")
                        st.write(f"**Description:** {result['description']}")
                        st.write(f"**Rating:** {result.get('product_rating', 'N/A')} | **Price:** {result.get('selling_price', 'N/A')} | **MRP:** {result.get('mrp', 'N/A')}")
                        st.write("------")
                else:
                    st.write("No products found.")

# Clustering Analysis Section
elif page == "Clustering Analysis":
    st.write("### Step 2: Clustering Analysis")
    uploaded_file_clustering = st.file_uploader("Upload your CSV file for Clustering Analysis", type="csv")

    if uploaded_file_clustering is not None:
        data_clustering = pd.read_csv(uploaded_file_clustering)
        
        # Check for required columns
        if 'product_name' not in data_clustering.columns or 'description' not in data_clustering.columns:
            st.error("CSV file must contain 'product_name' and 'description' columns.")
        else:
            st.write("### Data Preview:")
            st.dataframe(data_clustering.head())

            # Ensure 'product_name' and 'description' columns are strings
            data_clustering['product_name'] = data_clustering['product_name'].astype(str)
            data_clustering['description'] = data_clustering['description'].astype(str)

            # Take only the first 1000 rows for clustering
            data_clustering = data_clustering.iloc[:1000]

            # Create TF-IDF matrix
            tfidf_matrix, vectorizer = create_tfidf_matrix(data_clustering)

            # Clustering options
            st.write("### Choose Clustering Algorithm:")
            clustering_algorithm = st.selectbox("Select an algorithm:", ["K-Means", "Agglomerative", "DBSCAN"])

            if clustering_algorithm == "K-Means":
                num_clusters = st.number_input("Number of clusters:", min_value=2, max_value=10, value=3)
                if st.button("Run Clustering"):
                    labels, _ = kmeans_clustering(tfidf_matrix, num_clusters)
                    plot_clusters(tfidf_matrix, labels)
    
                # Calculate and display evaluation metrics
            silhouette_avg = silhouette_score(tfidf_matrix, labels)
            davies_bouldin = davies_bouldin_score(tfidf_matrix, labels)
            
            st.write("### Clustering Evaluation Metrics")
            st.metric("Silhouette Score", f"{silhouette_avg:.4f}")
            st.metric("Davies-Bouldin Index", f"{davies_bouldin:.4f}")
            elif clustering_algorithm == "Agglomerative":
                num_clusters = st.number_input("Select number of clusters for Agglomerative Clustering:", min_value=1, max_value=10, value=2)
                if st.button("Run Agglomerative Clustering"):
                    agglo_labels = agglomerative_clustering(tfidf_matrix, num_clusters)
                    st.write("Agglomerative Clustering Labels:")
                    # st.write(agglo_labels)
                    
                    # Create and plot the dendrogram
                    plt.figure(figsize=(10, 7))
                    linked = linkage(tfidf_matrix.toarray(), method='ward')
                    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
                    plt.title('Dendrogram for Agglomerative Clustering')
                    plt.xlabel('Samples')
                    plt.ylabel('Distance')
                    st.pyplot(plt)

            elif clustering_algorithm == "DBSCAN":
                eps_value = st.number_input("Epsilon:", min_value=0.1, value=0.5)
                min_samples_value = st.number_input("Min Samples:", min_value=1, value=5)
                if st.button("Run Clustering"):
                    labels = dbscan_clustering(tfidf_matrix, eps_value, min_samples_value)
                    plot_clusters(tfidf_matrix, labels)

# Product Similarity Section
elif page == "Product Similarity":
    st.write("### Step 3: Product Based on Description Using HITS Algorithm")

    # Upload CSV file for Product Similarity
    uploaded_file = st.file_uploader("Upload your CSV file for Product Similarity", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Check for required columns
        if 'product_name' not in data.columns or 'description' not in data.columns or 'product_url' not in data.columns:
            st.error("CSV file must contain 'product_name', 'description', and 'product_url' columns.")
        else:
            st.write("### Data Preview:")
            st.dataframe(data.head())

            # Ensure 'product_name', 'description' are strings
            data['product_name'] = data['product_name'].astype(str)
            data['description'] = data['description'].astype(str)

            # Create TF-IDF matrix
            tfidf_matrix, vectorizer = create_tfidf_matrix(data)

            query = st.text_area("Enter product description to find similar products:")

            if st.button("Find Similar Products"):
                get_top_products(tfidf_matrix, vectorizer, query, data)

# AI Chatbot Section
elif page == "Ai Chatbot":
    st.write("### AI Chatbot")

    for message in st.session_state['conversation']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask anything..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state['conversation'].append({"role": "user", "content": prompt})

        # Generate bot response using the AI chatbot function
        with st.spinner("Generating response..."):
            response = ai_chatbot(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state['conversation'].append({"role": "assistant", "content": response})
