import networkx as nx
import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE

from keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt

dataset = datasets.Cora()
display(HTML(dataset.description))
G, node_subjects = dataset.load()

train_subjects, test_subjects = model_selection.train_test_split(
    node_subjects, train_size=0.1, test_size=None, stratify=node_subjects
)
target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
test_targets = target_encoding.transform(test_subjects)
batch_size = 50
num_samples = [10, 5]
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)
train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)
graphsage_model = GraphSAGE(
    layer_sizes=[32, 32], generator=generator, bias=True, dropout=0.5,
)
x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

model = Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.005),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)
test_gen = generator.flow(test_subjects.index, test_targets)
history = model.fit(
    train_gen, epochs=1, validation_data=test_gen, verbose=2, shuffle=False
)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

all_nodes = node_subjects.index
all_targets = target_encoding.fit_transform(node_subjects)
all_mapper = generator.flow(all_nodes, all_targets)

embedding_model = Model(inputs=x_inp, outputs=x_out)

emb = embedding_model.predict(all_mapper)
print(emb.shape)