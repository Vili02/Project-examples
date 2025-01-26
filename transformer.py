from transformers import pipeline

# Creating a sentiment analysis pipeline
classifier = pipeline('sentiment-analysis')

# List of analysis examples
texts = [
    'We are very happy to introduce pipeline to the transformers repository.',
    'I am so sad that the weather is bad today.',
    'This is a neutral sentence, it does not express much emotion.',
    'I am extremely angry about the situation.',
    'Wow, what a beautiful day it is!'
]

#Sentiment analysis function
def analyze_sentiment(texts):
    for text in texts:
        # Check if text is not empty
        if not text.strip():
            print("Empty text provided, skipping...\n")
            continue
        
        # We perform sentiment analysis
        result = classifier(text)
        
        # We display the result in a more understandable way
        label = result[0]['label']
        score = result[0]['score']
        
        print(f"Text: {text}")
        print(f"Sentiment: {label} (Confidence: {score:.4f})\n")

# We perform the analysis for all texts
analyze_sentiment(texts)
