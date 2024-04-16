import gensim
from nltk.corpus import brown
import time
import matplotlib.pyplot as plt

training_set = brown.sents()

#We make the words all lowercase so different capitalization isn't seen as different words
training_set = [[word.lower() for word in sent] for sent in training_set]

def train_model(training_set, min_count, vector_size, window, sg = 0):

    if sg == 0:
        return gensim.models.Word2Vec(training_set, min_count=min_count, vector_size=vector_size, window=window)
    else:
        return gensim.models.Word2Vec(training_set, min_count=min_count, vector_size=vector_size, window=window, sg=sg)

vector_sizes = []
CBOW_training_times = []
SkipGram_training_times = []

for i in range(50, 1001, 50):
    vector_sizes.append(i)

    CBOW_training_time = time.time()
    CBOW_model = train_model(training_set, min_count=1, vector_size=i, window=5)
    CBOW_training_times.append(time.time() - CBOW_training_time)

    SkipGram_training_time = time.time()
    SkipGram_model = train_model(training_set, min_count=1, vector_size=i, window=5, sg=1)
    SkipGram_training_times.append(time.time() - SkipGram_training_time)

    print(i)


plt.plot(vector_sizes, CBOW_training_times, label='CBOW')
plt.plot(vector_sizes, SkipGram_training_times, label='SkipGram')
plt.xlabel('Vector Size')
plt.ylabel('Training Time')
plt.title('Training Time vs. Vector Size')
plt.legend()
plt.savefig('Training Time Results')
plt.close()