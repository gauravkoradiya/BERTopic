import os
from bertopic import BERTopic
import streamlit as st
import streamlit.components.v1 as components
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import time

st.set_page_config(page_title="Demo", page_icon=":shark:", layout="wide")
st.title("Topic Modeling with BERTopic")
st.caption("ISE 244: Final Project")
st.caption("By Gaurav Koradiya (016587129)")
st.caption("Github Repository: https://github.com/gauravkoradiya/HF-BERTopic")

with st.sidebar.form("Configuration"):
    st.header("Configurations")
    with st.expander ("Dataset Configuration",):
        # st.subheader("Dataset Configuration")
        dataset_name = st.text_input(label="Enter the name of the huggingface dataset to do analysis of:", value = "ag_news", help="Enter the name of the huggingface dataset path. Find list of datasets here: https://huggingface.co/datasets?task_ids=task_ids:topic-classification&sort=trending",)
        dataset_name_2 = st.text_input("Enter the name of the config for the dataset if it has one", help="Enter the name of the config for the dataset if it has one (usually it does not have one)")
        split_name = st.selectbox(label = "Enter the name of the split of the dataset that you want to use", options=["train", "test"], key="split_name", help="Select a split of the dataset to use and make sure that the split exists in the dataset. chek the dataset documentation for more information.")
        #split_name = form.text_area("Enter the name of the split of the dataset that you want to use", value = "train")
        number_of_records = st.number_input("Enter the number of documents that you want to analyze from the dataset", value = 200, help="It is recommended to keep this number low if you are using a large dataset to avoid long runtimes.")
        column_name = st.text_area("Enter the name of the column that we are doing analysis on (the X value)", value='text', help= "User should enter the name of the column that contains the text data that considers for topic modeling")
        labels = st.checkbox("Does this dataset have labels that you want to use?", value = True, help="Check this if you want to use labels for analysis")
        if labels == True:
            labels_column_name = st.text_area("Enter the name of the column that we are using for labels doing analysis on (the Y value)", value = "label", help="User should enter the name of the column that contains the label/class or any annotation that are used for topic modeling")
 
    with st.expander("Model Configuration"):
        # form.header("BERTopic Settings")
        use_topic_reduction = st.selectbox(label="How do you want to handle topic reduction", options=["HDBScan", "Auto", "Manual"], help="Leave this if you want HDBScan to choose the number of topics (clusters) for you. Set to Auto to have BERTopic prune these topics further, set to Manual to specify the number yourself")
        number_of_topics = st.number_input(label="Enter the number of topics to use if doing Manual topic reduction", value = 4, help="This is the number of topics that you want to reduce the topics to. This is only used if you select Manual topic reduction")
        use_random_seed = st.checkbox("Do you want to make the results reproducible?", value = False, help="Check this if you want to make the results reproducible. This will make the results reproducible but will slow down the analysis significantly.")

        st.header("1. Embedding Model Configuration")
        model_name = st.text_area("Enter the name of the pre-trained model from sentence transformers that we are using for featurization/embedding", value = "all-MiniLM-L6-v2", help="This will download a new model, so it may take awhile or even break if the model is too large. See the list of pre-trained models that are available here! https://www.sbert.net/docs/pretrained_models.html")

        st.header("2. CounterVectorizer Configuration")
        cv_lowercase = st.checkbox("Shall we automatically lowercase the text?", value = True)
        cv_ngram_min = st.number_input(label="minimum n-value for n-grams", value = 1, help="What's the lower boundary of the range of n-values for different word n-grams or char n-grams to be extracted")
        cv_ngram_max = st.number_input(label='maximum n-value for n-grams',help="What's the upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted", value = 1)
        cv_analyzer = st.selectbox(label="Enter the analyzer mode:", options=["word", "char", "char_wb"], help="This selects between looking at n-grams, character n-grams, and character n-grams only from text within word boundries.")
        cv_max_df = st.number_input("Ignore terms that have a document frequency strictly higher than the given threshold", value = 0.95, help="This parameter represents a proportion of documents if a float is given")
        cv_min_df = st.number_input("Ignore terms that have a document frequency strictly lower than the given threshold", value = 0.1, help="This parameter represents a proportion of documents if a float is given")
        cv_max_features = st.number_input("Enter the maximum number of n-grams to be featurized", value = 100000, help="This parameter represents the maximum number of n-grams to be featurized. This is used to limit the number of n-grams to be featurized.")

        st.header("3. HDBScan Configuration")
        hdbscan_min_cluster_size = st.number_input("Enter the number of points necessary to form a new cluster", value = 3, help="Set it to the smallest size grouping that you wish to consider a cluster. This is the most impactful setting for HDBscan")
        hdbscan_min_samples = st.number_input("Enter the minimum number of points to be declared a cluster instead of noise", value = 2, help="The larger the value of min_samples you provide, the more conservative the clustering – more points will be declared as noise, and clusters will be restricted to progressively more dense areas.")
        hdbscan_metric = st.text_area(label = 'Enter the distance matric', help = "Enter the name of the metric used for computing distances for HDBscan. Common metrics for NLP are euclidean and cosine. Cosine is not supported by HDBscan", value = "euclidean")


        st.header("4. UMAP Configuration")
        umap_n_neighbors = st.number_input("Enter the number of neighbors used by UMAP for generating its manifold", value = 15, help="This parameter controls how UMAP balances local versus global structure in the data. It does this by constraining the size of the local neighborhood UMAP will look at when attempting to learn the manifold structure of the data. This means that low values of n_neighbors will force UMAP to concentrate on very local structure (potentially to the detriment of the big picture), while large values will push UMAP to look at larger neighborhoods of each point when estimating the manifold structure of the data, losing fine detail structure for the sake of getting the broader of the data.")
        umap_min_dist = st.number_input("Enter the minimum distance that points are allowed to be apart in the low dimensional manifold", value = 0.1, help="The min_dist parameter controls how tightly UMAP is allowed to pack points together. It, quite literally, provides the minimum distance apart that points are allowed to be in the low dimensional representation. This means that low values of min_dist will result in clumpier embeddings. This can be useful if you are interested in clustering, or in finer topological structure. Larger values of min_dist will prevent UMAP from packing points together and will focus on the preservation of the broad topological structure instead.")
        umap_n_components = st.number_input("Enter the number of dimensions/components for UMAP to create", value = 2, help="UMAP provides a n_components parameter option that allows the user to determine the dimensionality of the reduced dimension space we will be embedding the data into. Unlike some other visualisation algorithms such as t-SNE, UMAP scales well in the embedding dimension, so you can use it for more than just visualisation in 2- or 3-dimensions. UMAP is used in BERTopic primarily to allow the highly effective clustering algorithm, HDBScan, to effectively cluster effectively because HDBScan becomes extremely slow on high dimensional data. Setting this value above 10 may slow the analysis to a crawl")
        umap_metric = st.text_area("Enter the name of the metric used for computing distances. Common metrics for NLP are euclidean and cosine for UMAP", value = "cosine", help="A complete list of all available metrics supported by UMAP can be found here: https://umap-learn.readthedocs.io/en/latest/parameters.html#metric")

        st.header("5. c-TF-IDF Configuration")
        bm25_weighting = st.checkbox("Do you want to use BM25 weighting?", value = False, help="Check this if you want to use BM25 weighting instead of the default c-TF-IDF weighting")
        reduce_frequent_words = st.checkbox("Do you want to reduce frequent words?", value = False, help="Check this if you want to reduce frequent words from the c-TF-IDF weighting")


    submitted_configuration_form = st.form_submit_button("Submit")

@st.cache_data
def load_and_process_data(path, name, streaming, split_name, number_of_records):
    dataset = load_dataset(path = path, name = name, streaming=streaming)
    #return list(dataset)
    dataset_head = dataset[split_name].take(number_of_records)
    df = pd.DataFrame.from_dict(dataset_head)
    return df

@st.cache_resource
def load_model(path = None):
    if st.session_state.model is not None:
        return st.session_state.model
    else:
        if path is not None:
            return BERTopic.load(path)
        else:
            #list all the models path in the models folder and take the latest one based on time it was added to the folder
            model_directory = "./models"
            models_file_path = os.listdir(model_directory)
            models_file_path.sort(key=os.path.getmtime)
            latest_model_path = models_file_path[-1]
            return BERTopic.load(os.path.join(model_directory, latest_model_path))

traning_tab, inference_tab = st.tabs(["Traning", "Inference"])
with traning_tab:
    if submitted_configuration_form:

        with st.status("Preparing and Building Artifacts...", expanded=True) as status:

            st.write("Loading and processing data...")
            df = load_and_process_data(dataset_name, dataset_name_2, True, split_name, number_of_records)
            time.sleep(1)

            st.write("Initializing HDBScan...")
            hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples = hdbscan_min_samples, metric=hdbscan_metric, prediction_data=True)
            time.sleep(1)

            st.write("Initializing UMAP...")
            if use_random_seed:
                umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, min_dist=umap_min_dist, metric=umap_metric, random_state = 42)
            else:
                umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, min_dist=umap_min_dist, metric=umap_metric)
            time.sleep(1)

            st.write("Initializing CountVectorizer...")
            vectorizer_model = CountVectorizer(lowercase = cv_lowercase, ngram_range=(cv_ngram_min, cv_ngram_max), analyzer=cv_analyzer, max_df=cv_max_df, min_df=cv_min_df, stop_words="english")
            time.sleep(1)

            st.write("Initializing C-TF-IDF Cectorizer...")
            ctfidf_model = ClassTfidfTransformer(bm25_weighting=bm25_weighting, reduce_frequent_words=bm25_weighting)
            topic_model = BERTopic(ctfidf_model=ctfidf_model)
            time.sleep(1)


            st.write("Initalizing BERTopic Model...")
            #model = load_model(model_name=model_name, _hdbscan_model=hdbscan_model, _umap_model=umap_model, _vectorizer_model=vectorizer_model, use_topic_reduction=use_topic_reduction, number_of_topics=number_of_topics)
            sentence_model = SentenceTransformer(model_name)
            if use_topic_reduction == "Auto":
                model = BERTopic(embedding_model=sentence_model,ctfidf_model=ctfidf_model, umap_model = umap_model, hdbscan_model = hdbscan_model, vectorizer_model = vectorizer_model, nr_topics = "auto", calculate_probabilities = True)
            elif use_topic_reduction == "Manual":
                model = BERTopic(embedding_model=sentence_model,ctfidf_model=ctfidf_model, umap_model = umap_model, hdbscan_model = hdbscan_model, vectorizer_model = vectorizer_model, nr_topics = number_of_topics, calculate_probabilities = True)
            else:
                model = BERTopic(embedding_model=sentence_model,ctfidf_model=ctfidf_model, umap_model = umap_model, hdbscan_model = hdbscan_model, vectorizer_model = vectorizer_model, calculate_probabilities = True)  
            time.sleep(1)
            status.update(label="Data and Model has been loaded !!!!", state="complete", expanded=False)


        X = df[column_name]

        st.header("Original Dataset (First 10 rows)")
        st.write(df.head(10))
        with st.spinner('Training the model...'):
            topics, probs = model.fit_transform(X)
            st.success('Training Completed Successfully!!')
            #remove / from dataset name
            dataset_name_for_path = dataset_name.replace("/", "_")
            os.makedirs(os.path.join('models',dataset_name_for_path), exist_ok=True)
            model_path = os.path.join("models", dataset_name_for_path ,"bertopic_model_{0}_{1}_{2}.pkl".format(dataset_name_for_path, model_name, time.strftime("%Y%m%d-%H%M%S")))
            model.save(path=model_path)
            st.session_state.model = model
            st.success("Model has been saved at `{0}`".format(model_path))
        
        # st.header("Topic assignment for each example")
        # st.write(topics)

        topic_info = model.get_topic_info()
        st.header("Topic Info")
        st.dataframe(topic_info,column_config={
            "Representation":"Topic Representation",
            "Name": None,
            "Count":"Number of Documents",
            "Topic": "Topic ID",
        })

        st.header("Topic probability distribution for each example")
        st.dataframe(
            data = pd.DataFrame({column_name: X, "Topic Distribution": probs.tolist()}),
            column_config={
                column_name: st.column_config.TextColumn("Text"),
                "Topic Distribution": st.column_config.LineChartColumn("Topic Distribution", 
                    help="This is the probability distribution of each topic for each document",                                              
                    y_min = 0.0,
                    y_max = 1.0,
                )
            }
        )
                            

        # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
        embeddings = model._extract_embeddings(X)
        reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
        st.header("Visualizing the Topic Cluster in 2D space")
        st.plotly_chart(model.visualize_documents(df['text'], reduced_embeddings=reduced_embeddings))



        # st.header("Detailed Topic Info")
        # topic_info_list = []
        # with st.expander("Open to see Detailed Topic Info:"):
        #     for xhat in range(-1, len(topic_info)):
        #         topic_info_list.append(model.get_topic(xhat))
        #     st.write(topic_info_list)

        # st.header("Representitive docs for each topic")
        # with st.expander("Open to see Representitive docs for each topic"):
        #     rep_doc_list = []
        #     for yhat in range(0, len(topic_info)-1):
        #         rep_doc_list.append(model.get_representative_docs(topic = yhat))
        #     st.write(rep_doc_list)
        if labels:
            y = df[labels_column_name]
            st.header("Topics per class")
            topics_per_class = model.topics_per_class(X, classes=y)
            st.plotly_chart(model.visualize_topics_per_class(topics_per_class))
        #TODO:Each of these need there own options!
        st.divider()
        st.header("Visualizations")
        st.plotly_chart(model.visualize_topics())
        st.plotly_chart(model.visualize_barchart(top_n_topics = 9990, n_words = 9999))
        st.plotly_chart(model.visualize_heatmap())
        st.plotly_chart(model.visualize_hierarchy(hierarchical_topics=model.hierarchical_topics(X)))
        st.plotly_chart(model.visualize_term_rank())
    else:
        st.info("Please fill out the configuration in sidebar and submit job for traning to see the results")

with inference_tab:

    with st.form("Inference"):

        st.header("Prediction on unseen text document")

        unseen_doc_txt = """
        Police in southern China paraded four suspects through the streets for allegedly smuggling people across sealed borders in breach of pandemic control measures -- a controversial act of public shaming that triggered backlash on Chinese social media.
        On Tuesday, four people wearing hazmat suits, face masks and goggles were paraded in Jingxi city, Guangxi province -- each carrying placards showing their names and photos on their chest and back, according to videos shared on social media and republished by state media outlets.
        Each suspect was held by two officers -- also wearing hazmat suits and face shields. They were surrounded by yet another circle of police, some holding machine guns and in riot gear, while a large crowd looked on.

        The four people were suspected of helping others to illegally cross China's borders, which have been largely sealed during the pandemic as part of the country's "zero-Covid policy," according to the state-run Guangxi Daily, 
        """
        unseen_doc = st.text_area("Enter an unseen document to perform topic modeling on using the trained model", placeholder= 'Enter text here...',)
        model_path = st.text_input("Enter the path of the model that you want to use for inference", placeholder= 'Enter model path here...',help="Enter the path of the model that you want to use for inference. If you leave this blank, the latest model will be used for inference")
        submit_inference_request = st.form_submit_button("Submit")
    if submit_inference_request:
        with st.spinner('Infering a topic...'):
            model = load_model(path = model_path)
            new_topic, new_probs = model.transform(unseen_doc)
        st.plotly_chart(model.visualize_distribution(new_probs[0], min_probability = 0.0)) #### do on an unseen document