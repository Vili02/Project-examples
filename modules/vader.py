import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

sentences = [
    "VADER is smart, handsome, and funny.",
    "VADER is smart, handsome, and funny!",
    "VADER is very smart, handsome, and funny.",
    "VADER is VERY SMART, handsome, and FUNNY.",
    "VADER is VERY SMART, handsome, and FUNNY!!!",
    "VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",
    "The book was good.",
    "The book was kind of good.",
    "The plot was good, but the characters are uncompelling and the dialog is not great.",
    "A really bad, horrible book.",
    "At least it isn't a horrible book.",
    ":) and :D",
    "",
    "Today sux",
    "Today sux!",
    "Today SUX!",
    "Today kinda sux! But I'll get by, lol"
]

# Stop words
stop_words = set(stopwords.words('english'))

for sentence in sentences:
    print(f"Original sentence: {sentence}")
    
    #Tokenization of words
    words = word_tokenize(sentence)
    print(f"Tokenized words: {words}")
    
    # Remove stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    print(f"Filtered words (without stop words): {filtered_words}")
    
    # Partial speech tagging (POS tagging)
    pos_tags = pos_tag(filtered_words)
    print(f"POS tagging: {pos_tags}")
    
    # Sentiment analysis via VADER
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print(f"{k}: {ss[k]}, ", end='')
    print("\n" + "-"*50)
