import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import pos_tag


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")


def load_modules():
    return {
        "Sentiment Analysis": {
            "module": SentimentAnalysisModule(),
            "description": "Analyze the sentiment (positive/negative/neutral) of a given sentence."
        },
        "POS Tagging": {
            "module": POSTaggingModule(),
            "description": "Tag parts of speech (noun, verb, etc.) in a given sentence."
        },
        "Tokenization": {
            "module": TokenizationModule(),
            "description": "Break a sentence into tokens (words or punctuation)."
        },
    }

class SentimentAnalysisModule:
    def run(self):
        text = input("Enter a sentence for sentiment analysis: ")
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(text)
        print(f"Sentiment analysis result: {ss}\n")

class POSTaggingModule:
    def run(self):
        text = input("Enter a sentence for POS tagging: ")
        words = word_tokenize(text)
        tagged = pos_tag(words)
        print(f"POS tagging result: {tagged}\n")

class TokenizationModule:
    def run(self):
        text = input("Enter a sentence for tokenization: ")
        tokens = word_tokenize(text)
        print(f"Tokenization result: {tokens}\n")

def main():
    print("Welcome to the NLTK Interactive System!")
    modules = load_modules()
    print("Available modules:")
    for i, (module_name, module_info) in enumerate(modules.items(), start=1):
        print(f"{i}. {module_name} - {module_info['description']}")
    
    while True:
        choice = input("\nEnter the number of the module you want to use (or 'exit' to quit): ").strip()
        if choice.lower() == "exit":
            print("Goodbye!")
            break
        
        if choice.isdigit() and 1 <= int(choice) <= len(modules):
            module_name = list(modules.keys())[int(choice) - 1]
            module = modules[module_name]['module']
            if hasattr(module, "run"):
                print(f"\nRunning {module_name}...")
                module.run()
            else:
                print(f"{module_name} does not have a 'run' function. Skipping...")
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()

