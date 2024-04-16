import gensim
from nltk.corpus import brown
import time

training_set = brown.sents()

#We make the words all lowercase so different capitalization isn't seen as different words
training_set = [[word.lower() for word in sent] for sent in training_set]

def train_model(training_set, min_count, vector_size, window, sg = 0):

    if sg == 0:
        return gensim.models.Word2Vec(training_set, min_count=min_count, vector_size=vector_size, window=window)
    else:
        return gensim.models.Word2Vec(training_set, min_count=min_count, vector_size=vector_size, window=window, sg=sg)
    

CBOW_training_time = time.time()
CBOW_model = train_model(training_set, min_count=1, vector_size=1000, window=5)
CBOW_training_time = time.time() - CBOW_training_time
CBOW_model.save('CBOW_brown.embedding')

print(f"Trained CBOW in {CBOW_training_time} seconds")

SkipGram_training_time = time.time()
SkipGram_model = train_model(training_set, min_count=1, vector_size=1000, window=5, sg=1)
SkipGram_training_time = time.time() - SkipGram_training_time
SkipGram_model.save('SkipGram_brown.embedding')

print(f"Trained Skip Gram in {SkipGram_training_time} seconds")

#Training the 2D embeddings

CBOW_2D_training_time = time.time()
CBOW_2D_model = train_model(training_set, min_count=1, vector_size=2, window=5)
CBOW_2D_training_time = time.time() - CBOW_2D_training_time
CBOW_2D_model.save('CBOW_brown_2D.embedding')

print(f"\nTrained CBOW_2D in {CBOW_2D_training_time} seconds")

SkipGram_2D_training_time = time.time()
SkipGram_2D_model = train_model(training_set, min_count=1, vector_size=2, window=5, sg=1)
SkipGram_2D_training_time = time.time() - SkipGram_2D_training_time
SkipGram_2D_model.save('SkipGram_brown_2D.embedding')

print(f"Trained Skip Gram_2D in {SkipGram_2D_training_time} seconds")