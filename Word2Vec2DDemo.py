import gensim
from nltk.corpus import brown
import numpy as np
import matplotlib.pyplot as plt

CBOW_model_data =  gensim.models.Word2Vec.load('CBOW_brown_2D.embedding')
SkipGram_model_data =  gensim.models.Word2Vec.load('SkipGram_brown_2D.embedding')

words = ['queen', 'king', 'woman', 'man']

x = []
y = []

for word in words:
    x.append(CBOW_model_data.wv[word][0])
    y.append(CBOW_model_data.wv[word][1])

for i in range(len(x)):
    plt.scatter(x[i], y[i], label=words[i])
    plt.text(x[i], y[i], words[i], fontsize=9, ha='left', va='bottom')

plt.title('CBOW 2D Results')
plt.grid(True)  # Add grid if needed
plt.legend()

plt.savefig('CBOW_2D_Results')
plt.close()

x = []
y = []

for word in words:
    x.append(SkipGram_model_data.wv[word][0])
    y.append(SkipGram_model_data.wv[word][1])

for i in range(len(x)):
    plt.scatter(x[i], y[i], label=words[i])
    plt.text(x[i], y[i], words[i], fontsize=9, ha='left', va='bottom')

plt.title('SkipGram 2D Results')
plt.grid(True)  # Add grid if needed
plt.legend()

plt.savefig('SkipGram_2D_Results')
plt.close()
