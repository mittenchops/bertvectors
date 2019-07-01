from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pprint import pprint as pp
from sklearn.manifold import TSNE

#
import pandas as pd
#

from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet

bc = BertClient()

words = ['fish',
         "fishes",
         "werewolf",
         "wolfman",
         "wolf",
         "man",
         "dog",
         "dog dog dog dog dog",
         "cat",
         "monkey",
         "bull",
         "cow",
         "bird",
         "mermaid",
         "woman",
         "sirens",
         "minotaur",
         "Labyrinth",
         "centaur",
         "pegasus",
         "Greek",
         "American",
         "South African",
         "San Francisco",
         "horse",
         "Elon Musk",
         "Brexit",
         "China trade war",
         "conference",
         "astronaut",
         "fraud",
         "ambulance",
         "the cat is on the mat",
         "the mat is on the cat",
         "the cat is under the mat",
         "the mat is under the cat",
         "New York City",
         "New Jersey",
         "Big business",
         "quarterly report",
         "annual filing",
         "income statement",
         "10-K",
         "10-Q",
         "8-K",
         "Theseus",
         "Perseus",
         "Odysseus",
         "David Bowie",
         "David Brooks",
         "Jorge Luis Borges",
         "astronaut",
         "mechanic",
         "lawyer",
         "sailor",
         "longshoreman",
         "pirate"]

def makevecs(words):
  return {word: bc.encode([word])[0] for word in words}
# similarities = cosine_similarity(vecs.values())

vecs = makevecs(words)

def scalar_projection_a_onto_b(a,b):
    return (np.dot(a,b) / np.linalg.norm(b))

def cos(a,b):
    # lol, you discovered the cosine dummy.
    # this is |A|cos (theta) / |A|
    # just use the cosine_similarity function from numpy
    return scalar_projection_a_onto_b(a,b)/np.linalg.norm(a)
    
def all_projections(word, vec_dict):
    vec_of_word = bc.encode([word])[0]
    items = []
    for k,v in vec_dict.items():
        proj = (word, k, scalar_projection_a_onto_b(vec_of_word,v))
        # proj = (word, k, cos(vec_of_word,v))
        items.append(proj)
    items.sort(key = lambda x: -x[2])
    return items

words2 = ["annual filing",
          "income statement",
          "balance sheet",
          "industrials",
          "west texas intermediate",
          "ipod",
          "international trade",
          "housecat"]

vecs2 = makevecs(words2)

tests = ["Rhinocerous", "Athens", "Tokyo", "Elon Musk", "Paris", "tiger", "Tesla has 1,000 shares outstanding and faces risk conditions that can be mitigated in the third quarter by strong revenues: Description of Business, Item 1."]

for test in tests:
    pp(all_projections(test,vecs))

for test in tests:
    pp(all_projections(test,vecs2))
    
# def refresh(words, word):
#   words = copy.deepcopy(words)
#   if word not in words:
#       words.append(word)
#   words = sorted(list(set(words)))
#   vecs = {word: bc.encode([word])[0] for word in words}
#   similarities = cosine_similarity(vecs.values())
#   # vecs = bc.encode(words)
#   # similarities = cosine_similarity(vecs)
#   matches = sorted(list(zip(words,similarities[words.index(word)])), key = lambda x: -x[1])
#   return list(matches)

# matches = lambda word : sorted(list(zip(words,similarities[words.index(word)])), key = lambda x: -x[1])

# matches("fish")



#

tmp = [vecs[word] for word in words]
X_embedded = TSNE(n_components=2, n_iter=2500, verbose=2, method="exact").fit_transform(tmp)

print("Computed t-SNE", X_embedded.shape)

df = pd.DataFrame(columns=['x', 'y', 'word'])
# This is using wrong X and Y, but you understand.
# You should change to use Y as the cos and X as the vectors
# df['x'], df['y'], df['word'] = X_embedded[:,0], X_embedded[:,1], words

# make the Y
y = "Rhinocerous"
Y = [cos(bc.encode([y]),vecs[word])[0] for word in words]
df['x'], df['y'], df['word'] = X_embedded[:,0], Y, words

source = ColumnDataSource(ColumnDataSource.from_df(df))
labels = LabelSet(x="x", y="y", text="word", y_offset=8,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')


plot = figure(plot_width=600, plot_height=600)
plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)
plot.add_layout(labels)
show(plot)
