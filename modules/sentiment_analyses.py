# modules/sentiment_analysis.py
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.translate import Translator
from textblob import TextBlob

class SentimentAnalysisModule:
    def run(self):
       
        text = input("Enter a sentence for sentiment analysis: ")
        
        
        translated_text = self.translate_text(text, "fr")
        print(f"Translated text to French: {translated_text}")
        
       
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(text)
        
       
        print("Sentiment analysis result:")
        print(f"Negative: {ss['neg']}, Neutral: {ss['neu']}, Positive: {ss['pos']}, Compound: {ss['compound']}")
        
        
        blob = TextBlob(text)
        print(f"Sentiment polarity: {blob.sentiment.polarity}")
        
        if blob.sentiment.polarity > 0:
            print("Positive Emotion")
        elif blob.sentiment.polarity < 0:
            print("Negative Emotion")
        else:
            print("Neutral Emotion")
    
    def translate_text(self, text, target_language="fr"):
        
        translator = Translator()
        translated = translator.translate(text, dest=target_language)
        return translated.text
