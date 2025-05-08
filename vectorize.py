"""
Code to vectorize the pre-processed review text data from 5kmovies_preprocssed.csv
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
import nltk
#nltk.download('wordnet')
#nltk.download('sentiwordnet')
#nltk.download('punkt')
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
import gensim.downloader as api
from gensim import corpora, models
from gensim.models import KeyedVectors
import spacy
from scipy.stats import pearsonr
from gensim.models import CoherenceModel
from textblob import TextBlob

nlp = spacy.load('en_core_web_sm')

def get_sentiment(word):
    """
    get_sentiment - return sentiment score of word
    
    Purpose:
        This method calculates the sentiment score of the word

    Parameters:
        word (string): word we are calculating the sentiment score of

    Returns:
        Returns the sentiment score
    """
    synsets = wn.synsets(word)
    if synsets:
        synset = synsets[0]
        senti_synset = swn.senti_synset(synset.name())
        return senti_synset.pos_score() - senti_synset.neg_score()
    return 0.0

def document_vector_largermodel(doc):
    """
    document_vector_largermodel - vector for the given document
    
    Purpose:
        This method returns the vector for a given document for Google Word2Vec

    Parameters:
        doc (string): review text we are vectorizing

    Returns:
        Returns the average score for each token
    """    
    tokens = doc.split()
    vectors = [word2vec[word] for word in tokens if word in word2vec]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(300)
        
def document_vector(doc):
    """
    document_vector - vector for the given document
    
    Purpose:
        This method returns the vector for a given document for Word2Vec

    Parameters:
        doc (string): review text we are vectorizing

    Returns:
        Returns the vector using nlp function
    """   
    return nlp(doc).vector

if __name__ == '__main__':

    # preprocessing and vectorization (elementary steps )
    df = pd.read_csv("5kmovies_preprocssed.csv")
    df["doc_vector"] = df["cleaned_reviews"].apply(lambda x: nlp(x).vector)
    vec_df = pd.DataFrame(df["doc_vector"].to_list())
    vec_df["ratings"] = df["Rating"]
    vec_df["avg_vector_val"] = vec_df.drop("ratings", axis=1).mean(axis=1)
    corr, _ = pearsonr(vec_df["avg_vector_val"], vec_df["ratings"])
    print(f"Correlation between average document vector and rating: {corr:.4f} ")
    docs = df['cleaned_reviews'].tolist()
    movie_names = df["Movie Name"].tolist()
    ratings = df["Rating"].tolist()

    #BoW

    bow_vector = CountVectorizer()
    bow_matrix = bow_vector.fit_transform(docs)
    bow_words = bow_vector.get_feature_names_out()
    
    #TF-IDF

    tfidf_vector = TfidfVectorizer()
    tfidf_matrix = tfidf_vector.fit_transform(docs)
    tfidf_words = tfidf_vector.get_feature_names_out()

    sentimentscore = np.array([get_sentiment(w) for w in tfidf_words])
    net_sentiment = tfidf_matrix.dot(sentimentscore)
    correlation = np.corrcoef(net_sentiment, ratings)[0,1]
    print(f"Correlation between sentiment-aware TF-IDF and rating: {correlation:.4f}")

    #word2Vec/GloVe

    word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    one_vector = np.array([document_vector_largermodel(docx) for docx in docs])

    doc_vectors = np.array([document_vector(doc) for doc in docs])

    #LDA Topic Modeling

    tokenization = [doc.split() for doc in docs]
    dictionary = corpora.Dictionary(tokenization)
    corpus = [dictionary.doc2bow(text) for text in tokenization]

    lda_model = models.LdaModel(corpus, num_topics=15, id2word=dictionary, passes=40)
    lda_topics = [lda_model.get_document_topics(bow) for bow in corpus]
    all_topic_distributions = [lda_model.get_document_topics(corpus[i], minimum_probability=0.0) for i in range(len(corpus))]
    topic_matrix = np.array([[prob for _, prob in doc] for doc in all_topic_distributions])
    topic_df = pd.DataFrame(topic_matrix, columns=[f"Topic_{i}" for i in range(topic_matrix.shape[1])])
    topic_df['Rating'] = ratings
    coherence_model = CoherenceModel(model=lda_model, texts=tokenization, dictionary=dictionary, coherence='c_v')
    coherent = coherence_model.get_coherence()
    print(f"Coherence Score: {coherent:.4f}")
    correlations = topic_df.corr()['Rating'].drop('Rating')


    #overarc writing and compiling

    sentiment_scores = [TextBlob(doc).sentiment.polarity for doc in docs]
    df['Sentiment'] = sentiment_scores
    print(df[['Rating', 'Sentiment']].corr())

    result_data = []
    for i in range(len(docs)):
        row = {
            'Movie Name': movie_names[i],
            'Rating': ratings[i],
            "BoW Vector (avg)": np.mean(bow_matrix[i].toarray()),
            'TF-IDF Vector (avg)': np.mean(tfidf_matrix[i].toarray()),
            'Word2Vec Vector (avg)': np.mean(doc_vectors[i]),
            'Word2Vec Vector Larger Google Model (avg)': np.mean(one_vector[i]),
            'Top LDA Topic': sorted(lda_topics[i], key=lambda x: -x[1])[0][0],
            'Topic LDA DF': topic_df,
            'Coherence Score Topics': coherent,
            'Sentiment Scores': sentiment_scores[i]
        }
        result_data.append(row)
    result_df = pd.DataFrame(result_data)
    result_df.to_csv("movie_nlp_analysis.csv", index=False)

    correlation = result_df.corr(numeric_only=True)
    print(correlation["Rating"])

