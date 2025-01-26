from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def run():
    text = input("Enter text for tokenization: ").strip()
    
   
    if not text:
        print("No text entered. Please enter some text.")
        return
    

    tokens = word_tokenize(text)
    print(f"Tokens: {tokens}")
    
   
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    print(f"Filtered Tokens (without stopwords): {filtered_tokens}")
    
   
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]
    print(f"Stemmed Tokens: {stemmed_tokens}")


if __name__ == "__main__":
    run()



