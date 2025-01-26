# modules/pos_tagging.py
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.translate import Translator

class POSTaggingModule:
    def run(self):
       # Enter text for POS marking
        text = input("Enter a sentence for POS tagging: ")
        
       # Translate text into Spanish
        translated_text = self.translate_text(text, "es")
        print(f"Translated text to Spanish: {translated_text}")
        
       # Text tokenization
        tokens = word_tokenize(text)
        
       # Marking parts of speech
        tagged = pos_tag(tokens)
        
       # Named entity recognition
        tree = ne_chunk(tagged)
        
        # Show result
        print("POS tagging result:")
        for word, tag in tagged:
            print(f"{word}: {tag}")
        
        print("\nNamed Entity Recognition result:")
        print(tree)
    
    def translate_text(self, text, target_language="es"):
        # Text translation function
        translator = Translator()
        translated = translator.translate(text, dest=target_language)
        return translated.text
