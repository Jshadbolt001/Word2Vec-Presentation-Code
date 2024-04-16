import gensim
from nltk.corpus import brown
import numpy as np
import matplotlib.pyplot as plt

CBOW_model_data =  gensim.models.Word2Vec.load('CBOW_brown_2D.embedding')
SkipGram_model_data =  gensim.models.Word2Vec.load('SkipGram_brown_2D.embedding')

words = list(CBOW_model_data.wv.key_to_index.keys())

x = [CBOW_model_data.wv[word][0] for word in words]
y = [CBOW_model_data.wv[word][1] for word in words]

# Plot
plt.scatter(x, y)

plt.title('CBOW 2D Results')
plt.grid(True)  # Add grid if needed

plt.savefig('CBOW_2D_Results_All_Words_ReducedWindow')
plt.close()

x = [SkipGram_model_data.wv[word][0] for word in words]
y = [SkipGram_model_data.wv[word][1] for word in words]

# Plot
plt.scatter(x, y)

plt.title('SkipGram 2D Results')
plt.grid(True)  # Add grid if needed

plt.savefig('SkipGram_2D_Results_All_Words_ReducedWindow')
plt.close()
