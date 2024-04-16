import gensim

CBOW_model_data =  gensim.models.Word2Vec.load('CBOW_brown.embedding')
SkipGram_model_data =  gensim.models.Word2Vec.load('SkipGram_brown.embedding')

first_word = input("Enter first word to compare:").lower()
second_word = input("Enter second word to compare:").lower()

print("\nCBOW Results:")
print(f"\tSimilarity of {first_word} and {second_word}: {CBOW_model_data.wv.similarity(first_word, second_word)}")

print("\nSkipGram Results:")
print(f"\tSimilarity of {first_word} and {second_word}: {SkipGram_model_data.wv.similarity(first_word, second_word)}")