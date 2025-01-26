import nltk
from nltk.translate import AlignedSent, Alignment, IBMModel1

def run():
    print("\nWelcome to the Machine Translation Module!")

    
    source_sentences = ["домът е голям", "котката е мила"]
    target_sentences = ["the house is big", "the cat is kind"]

    aligned_sentences = [
        AlignedSent(src.split(), tgt.split())
        for src, tgt in zip(source_sentences, target_sentences)
    ]

    
    print("\nTraining IBM Model 1...")
    ibm1 = IBMModel1(aligned_sentences, 5)  
    
    print("Training complete!\n")

    while True:
        source = input("Enter a source sentence in Bulgarian (or 'exit' to quit): ").strip().lower()
        if source == "exit":
            print("Exiting Machine Translation Module...")
            break

        source_words = source.split()
        target_words = []

       
        for word in source_words:
            if word in ibm1.translation_table:
                translations = ibm1.translation_table[word]
                best_translation = max(translations, key=translations.get)
                target_words.append(best_translation)
            else:
                target_words.append("[UNK]")  

        print("Translated sentence:", " ".join(target_words))
