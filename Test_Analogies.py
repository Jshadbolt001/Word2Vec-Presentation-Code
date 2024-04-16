import gensim

SkipGram_model_data =  gensim.models.Word2Vec.load('SkipGram_brown.embedding')

SkipGram_analogy_result = SkipGram_model_data.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1000)

for word, similarity in SkipGram_analogy_result:
    if word == 'queen':
        print(f"SkipGram: {word}: {similarity}")

CBOW_model_data =  gensim.models.Word2Vec.load('CBOW_brown.embedding')

CBOW_analogy_result = CBOW_model_data.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1000)

for word, similarity in CBOW_analogy_result:
    if word == 'queen':
        print(f"CBOW: {word}: {similarity}")

print('\n')

for word, similarity in CBOW_analogy_result:
    print(f"{word}: {similarity}")

