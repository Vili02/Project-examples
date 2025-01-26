# modules/topic_modeling.py
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

class TopicModelingModule:
    def run(self):
        
        documents = [
            "I love programming in Python. It is my favorite language.",
            "I enjoy machine learning and artificial intelligence.",
            "Natural language processing is fascinating.",
            "Python is great for web development, especially with Flask and Django.",
            "Data science is a booming field with lots of opportunities.",
            "Machine learning algorithms are very powerful.",
            "Python provides libraries for machine learning like scikit-learn, tensorflow.",
            "Web development is fun and creative.",
            "The importance of artificial intelligence in future technologies."
        ]

       
        stop_words = set(stopwords.words("english"))
        processed_docs = []

        
        for doc in documents:
            tokens = word_tokenize(doc.lower())
            filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            processed_docs.append(" ".join(filtered_tokens))
        
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(processed_docs)
        
        
        lda = LatentDirichletAllocation(n_components=2, random_state=42)
        lda.fit(X)
        
       
        terms = vectorizer.get_feature_names_out()
        
        print("Top words for each topic:")
        for topic_idx, topic in enumerate(lda.components_):
            print(f"\nTopic {topic_idx + 1}:")
            print(" ".join([terms[i] for i in topic.argsort()[:-10 - 1:-1]]))
