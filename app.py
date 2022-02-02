from bertopic import BERTopic
import streamlit as st
import streamlit.components.v1 as components
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="HF-BERTopic")
st.title("HF-BERTopic A front end for BERTopic")
st.caption("By Allen Roush")
st.caption("github: https://github.com/Hellisotherpeople")
st.caption("Linkedin: https://www.linkedin.com/in/allen-roush-27721011b/")
st.image("https://raw.githubusercontent.com/MaartenGr/BERTopic/master/images/logo.png", width = 380)
st.caption("By Maarten Grootendorst")
st.caption("github: https://github.com/MaartenGr/BERTopic")
st.caption("Linkedin: https://www.linkedin.com/in/mgrootendorst/")
st.image("https://maartengr.github.io/BERTopic/img/algorithm.png")


form = st.sidebar.form("Main Settings")

form.header("Main Settings")
#form.image("https://maartengr.github.io/BERTopic/img/algorithm.png", width = 270)


dataset_name = form.text_area("Enter the name of the huggingface dataset to do analysis of:", value = "Hellisotherpeople/DebateSum")
dataset_name_2 = form.text_area("Enter the name of the config for the dataset if it has one", value = "")
split_name = form.text_area("Enter the name of the split of the dataset that you want to use", value = "train")
number_of_records = form.number_input("Enter the number of documents that you want to analyze from the dataset", value = 200)
column_name = form.text_area("Enter the name of the column that we are doing analysis on (the X value)", value = "Full-Document")
labels = form.checkbox("Does this dataset have labels that you want to use?", value = True)
if labels == True:
    labels_column_name = form.text_area("Enter the name of the column that we are using for labels doing analysis on (the Y value)", value = "OriginalDebateFileName")

model_name = form.text_area("Enter the name of the pre-trained model from sentence transformers that we are using for featurization", value = "all-MiniLM-L6-v2")
form.caption("This will download a new model, so it may take awhile or even break if the model is too large")
form.caption("See the list of pre-trained models that are available here! https://www.sbert.net/docs/pretrained_models.html")

form.header("BERTopic Settings")
use_topic_reduction = form.selectbox("How do you want to handle topic reduction", ["HDBScan", "Auto", "Manual"])
form.caption("Leave this if you want HDBScan to choose the number of topics (clusters) for you. Set to Auto to have BERTopic prune these topics further, set to Manual to specify the number yourself")
number_of_topics = form.number_input("Enter the number of topics to use if doing Manual topic reduction", value = 3)
use_random_seed = form.checkbox("Do you want to make the results reproducible? This significantly slows down BERTopic", value = False)


form.header("CounterVectorizer Settings")
cv_lowercase = form.checkbox("Shall we automatically lowercase the text?", value = True)
cv_ngram_min = form.number_input("What's the lower boundary of the range of n-values for different word n-grams or char n-grams to be extracted", value = 1)
cv_ngram_max = form.number_input("What's the upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted", value = 1)
form.caption("Set the lower and upper boundry to 1 if you want the topics to be at the word level only (unigrams)")
cv_analyzer = form.selectbox("Enter the analyzer mode:", ["word", "char", "char_wb"])
form.caption("This selects between looking at n-grams, character n-grams, and character n-grams only from text within word boundries.")
cv_max_df = form.number_input("Ignore terms that have a document frequency strictly higher than the given threshold", value = 1.0)
form.caption("This parameter represents a proportion of documents if a float is given")
cv_min_df = form.number_input("Ignore terms that have a document frequency strictly lower than the given threshold", value = 0.5)
form.caption("This parameter represents a proportion of documents if a float is given")
cv_max_features = form.number_input("Enter the maximum number of n-grams to be featurized", value = 100000)



form.header("HDBScan Settings")
hdbscan_min_cluster_size = form.number_input("Enter the number of points necessary to form a new cluster", value = 5)
form.caption("Set it to the smallest size grouping that you wish to consider a cluster. This is the most impactful setting for HDBscan")
hdbscan_min_samples = form.number_input("Enter the minimum number of points to be declared a cluster instead of noise", value = 3)
form.caption("The larger the value of min_samples you provide, the more conservative the clustering – more points will be declared as noise, and clusters will be restricted to progressively more dense areas.")
hdbscan_metric = form.text_area("Enter the name of the metric used for computing distances for HDBscan. Common metrics for NLP are euclidean and cosine. Cosine is not supported by HDBscan", value = "euclidean")



form.header("UMAP Settings")
umap_n_neighbors = form.number_input("Enter the number of neighbors used by UMAP for generating its manifold", value = 15)
form.caption("This parameter controls how UMAP balances local versus global structure in the data. It does this by constraining the size of the local neighborhood UMAP will look at when attempting to learn the manifold structure of the data. This means that low values of n_neighbors will force UMAP to concentrate on very local structure (potentially to the detriment of the big picture), while large values will push UMAP to look at larger neighborhoods of each point when estimating the manifold structure of the data, losing fine detail structure for the sake of getting the broader of the data.")
umap_min_dist = form.number_input("Enter the minimum distance that points are allowed to be apart in the low dimensional manifold", value = 0.1)
form.caption("The min_dist parameter controls how tightly UMAP is allowed to pack points together. It, quite literally, provides the minimum distance apart that points are allowed to be in the low dimensional representation. This means that low values of min_dist will result in clumpier embeddings. This can be useful if you are interested in clustering, or in finer topological structure. Larger values of min_dist will prevent UMAP from packing points together and will focus on the preservation of the broad topological structure instead.")
umap_n_components = form.number_input("Enter the number of dimensions/components for UMAP to create", value = 2)
form.caption("UMAP provides a n_components parameter option that allows the user to determine the dimensionality of the reduced dimension space we will be embedding the data into. Unlike some other visualisation algorithms such as t-SNE, UMAP scales well in the embedding dimension, so you can use it for more than just visualisation in 2- or 3-dimensions.")
form.caption("UMAP is used in BERTopic primarily to allow the highly effective clustering algorithm, HDBScan, to effectively cluster effectively because HDBScan becomes extremely slow on high dimensional data. Setting this value above 10 may slow the analysis to a crawl")
umap_metric = form.text_area("Enter the name of the metric used for computing distances. Common metrics for NLP are euclidean and cosine", value = "cosine")
form.caption("A complete list of all available metrics supported by UMAP can be found here: https://umap-learn.readthedocs.io/en/latest/parameters.html#metric")




form.form_submit_button("Submit")








@st.cache
def load_and_process_data(path, name, streaming, split_name, number_of_records):
    dataset = load_dataset(path = path, name = name, streaming=streaming)
    #return list(dataset)
    dataset_head = dataset[split_name].take(number_of_records)
    df = pd.DataFrame.from_dict(dataset_head)
    return df



hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples = hdbscan_min_samples, metric=hdbscan_metric, prediction_data=True)
if use_random_seed:
    umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, min_dist=umap_min_dist, metric=umap_metric, random_state = 42)
else:
    umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, min_dist=umap_min_dist, metric=umap_metric)
vectorizer_model = CountVectorizer(lowercase = cv_lowercase, ngram_range=(cv_ngram_min, cv_ngram_max), analyzer=cv_analyzer, max_df=cv_max_df, min_df=cv_min_df, stop_words="english")



@st.cache(allow_output_mutation=True)
def load_model(model_name, hdbscan_model=hdbscan_model, umap_model=umap_model, vectorizer_model=vectorizer_model, use_topic_reduction = use_topic_reduction, number_of_topics = number_of_topics):
    sentence_model = SentenceTransformer(model_name)
    if use_topic_reduction == "Auto":
        kw_model = BERTopic(embedding_model=sentence_model, umap_model = umap_model, hdbscan_model = hdbscan_model, vectorizer_model = vectorizer_model, nr_topics = "auto", calculate_probabilities = True)
    elif use_topic_reduction == "Manual":
        kw_model = BERTopic(embedding_model=sentence_model, umap_model = umap_model, hdbscan_model = hdbscan_model, vectorizer_model = vectorizer_model, nr_topics = number_of_topics, calculate_probabilities = True)
    else:
        kw_model = BERTopic(embedding_model=sentence_model, umap_model = umap_model, hdbscan_model = hdbscan_model, vectorizer_model = vectorizer_model, calculate_probabilities = True)
    return kw_model

@st.cache()
def fit_transform(model, docs):
    topics, probs = model.fit_transform(docs)
    return topics, probs


model = load_model(model_name=model_name)

df = load_and_process_data(dataset_name, dataset_name_2, True, split_name, number_of_records)

X = df[column_name]

st.header("Original Dataset")
st.write(df)


topics, probs = fit_transform(model, X)

st.header("Topic assignment for each example")
st.write(topics)
st.header("Topic assignment probability for each example")
st.write(probs)
topic_info = model.get_topic_info()
st.header("Topic Info")
st.write(topic_info)
st.header("Detailed Topic Info")
topic_info_list = []
with st.expander("Open to see Detailed Topic Info:"):
    for xhat in range(-1, len(topic_info)):
        topic_info_list.append(model.get_topic(xhat))
    st.write(topic_info_list)
st.header("Representitive docs for each topic")
with st.expander("Open to see Representitive docs for each topic"):
    rep_doc_list = []
    for yhat in range(0, len(topic_info)-1):
        rep_doc_list.append(model.get_representative_docs(topic = yhat))
    st.write(rep_doc_list)
if labels:
    y = df[labels_column_name]
    st.header("Topics per class")
    topics_per_class = model.topics_per_class(X, topics, classes=y)
    st.plotly_chart(model.visualize_topics_per_class(topics_per_class))
#TODO:Each of these need there own options!

st.header("Visualizations")


st.plotly_chart(model.visualize_topics())
st.plotly_chart(model.visualize_barchart(top_n_topics = 9990, n_words = 9999))
st.plotly_chart(model.visualize_heatmap())
st.plotly_chart(model.visualize_hierarchy())
st.plotly_chart(model.visualize_term_rank())

st.header("Prediction on unseen data")

unseen_doc_txt = """
Police in southern China paraded four suspects through the streets for allegedly smuggling people across sealed borders in breach of pandemic control measures -- a controversial act of public shaming that triggered backlash on Chinese social media.
On Tuesday, four people wearing hazmat suits, face masks and goggles were paraded in Jingxi city, Guangxi province -- each carrying placards showing their names and photos on their chest and back, according to videos shared on social media and republished by state media outlets.
Each suspect was held by two officers -- also wearing hazmat suits and face shields. They were surrounded by yet another circle of police, some holding machine guns and in riot gear, while a large crowd looked on.

The four people were suspected of helping others to illegally cross China's borders, which have been largely sealed during the pandemic as part of the country's "zero-Covid policy," according to the state-run Guangxi Daily, 
"""
unseen_doc = st.text_area("Enter an unseen document to perform topic modeling on using the trained model", value = unseen_doc_txt)

new_topic, new_probs = model.transform(unseen_doc)
st.write(new_topic)
st.plotly_chart(model.visualize_distribution(new_probs[0], min_probability = 0.0)) #### do on an unseen document
