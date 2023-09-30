import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

import os
import networkx as nx
import numpy as np
import pandas as pd
from tensorflow import keras

from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph.data import BiasedRandomWalk
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph.layer import Node2Vec, link_classification

from stellargraph import datasets
from IPython.display import display, HTML


dataset = datasets.Cora()
G, subjects = dataset.load(largest_connected_component_only=True)
print(G.info())
walk_number = 2
walk_length = 2
walker = BiasedRandomWalk(
    G,
    n=walk_number,
    length=walk_length,
    p=0.5,  # defines probability, 1/p, of returning to source node
    q=2.0,  # defines probability, 1/q, for moving to a node away from the source node
)
unsupervised_samples = UnsupervisedSampler(G, nodes=list(G.nodes()), walker=walker)

batch_size = 50
epochs = 5
generator = Node2VecLinkGenerator(G, batch_size)
emb_size = 1433
node2vec = Node2Vec(emb_size, generator=generator)
x_inp, x_out = node2vec.in_out_tensors()
x_inp_src = x_inp[0]
x_out_src = x_out[0]
prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method="dot")(x_out)
model = keras.Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)

history = model.fit(
    generator.flow(unsupervised_samples),
    epochs=epochs,
    verbose=1,
    use_multiprocessing=False,
    workers=4,
    shuffle=True,
)
x_inp_src = x_inp[0]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
node_gen = Node2VecNodeGenerator(G, batch_size).flow(subjects.index)
prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method="dot")(x_out)
model = keras.Model(inputs=x_inp, outputs=prediction)
node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
print(node_embeddings.shape)
X = node_embeddings + G.node_features()
# y holds the corresponding target values
y = np.array(subjects)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, test_size=None)
# print(
#     "Array shapes:\n X_train = {}\n y_train = {}\n X_test = {}\n y_test = {}".format(
#         X_train.shape, y_train.shape, X_test.shape, y_test.shape
#     )
# )
clf = LogisticRegressionCV(
    Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=300
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print("1")