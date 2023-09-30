import stellargraph
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, HinSAGE, link_classification

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar
from stellargraph import datasets
from IPython.display import display, HTML
import pandas as pd
from stellargraph.data import UnsupervisedSampler
from stellargraph.data import BiasedRandomWalk
from stellargraph.mapper import Node2VecLinkGenerator
import numpy as np
import os

from sklearn.metrics import roc_auc_score
import tensorflow as tf
my_var = tf.Variable(0.0, name='my_var')
def roc_auc(y_true, y_pred):
    auc = tf.keras.metrics.AUC(num_thresholds=200, curve='ROC')
    auc.update_state(y_true, y_pred)
    return auc.result()
batch_size = 20
epochs = 1
num_samples = [20, 10]
layer_sizes = [20, 20]

folder_path = r"C:\Users\Penguin\Desktop\2\as-733"
anth=0
model=None
for root, dirs, files in os.walk(folder_path):

    files=files[::10]
    for file_name in files:
        if anth == 0:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.readlines()[4:]
                    for i in range(len(content)):
                        content[i] = content[i].replace('\t', ',').replace('\n', '').split(",")
                    df = pd.DataFrame(content, columns=['source', 'target'])
                    a = pd.concat([df['source'], df['target']], ignore_index=True).unique()
                    node_features = np.random.randn(len(a), 128)
                    node_features = pd.DataFrame(node_features, index=a)
                    G = sg.StellarGraph(nodes=node_features, edges=df)
                    edge_splitter_test = EdgeSplitter(G)
                    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
                        p=0.1, method="global", keep_connected=True
                    )
                    edge_splitter_train = EdgeSplitter(G_test)
                    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
                        p=0.1, method="global", keep_connected=True
                    )
                    train_gen = GraphSAGELinkGenerator(G, batch_size, num_samples)
                    train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)
                    test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
                    test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

                    layer_sizes = [20, 20]
                    graphsage = GraphSAGE(
                        layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3
                    )
                    x_inp, x_out = graphsage.in_out_tensors()
                    prediction = link_classification(
                        output_dim=1, output_act="relu", edge_embedding_method="ip"
                    )(x_out)
                    model = keras.Model(inputs=x_inp, outputs=prediction)
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                        loss=keras.losses.binary_crossentropy,
                        metrics=[keras.metrics.AUC(num_thresholds=200, curve='ROC')],
                    )

                    history = model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=2)


                    anth=anth+1
                    print("第", anth, "个时间训练完成")
        else:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.readlines()[4:]
                    for i in range(len(content)):
                        content[i] = content[i].replace('\t', ',').replace('\n', '').split(",")
                    df = pd.DataFrame(content, columns=['source', 'target'])
                    a = pd.concat([df['source'], df['target']], ignore_index=True).unique()
                    node_features = np.random.randn(len(a), 128)
                    node_features = pd.DataFrame(node_features, index=a)
                    G = sg.StellarGraph(nodes=node_features, edges=df)
                    edge_splitter_test = EdgeSplitter(G)
                    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
                        p=0.1, method="global", keep_connected=True
                    )

                    edge_splitter_train = EdgeSplitter(G_test)
                    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
                        p=0.1, method="global", keep_connected=True
                    )
                    train_gen = GraphSAGELinkGenerator(G, batch_size, num_samples)
                    train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)
                    test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
                    test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

                    layer_sizes = [20, 20]
                    history = model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=2)
                    anth = anth + 1
                    print("第", anth, "个时间训练完成")

# f=open(r"C:\Users\Penguin\Desktop\2\as20000102.txt")
# line = f.readlines()[4:]
# for i in range(len(line)):
#     line[i]=line[i].replace('\t',',').replace('\n','').split(",")
# df = pd.DataFrame(line,columns=['source', 'target'])
# a = pd.concat([df['source'], df['target']], ignore_index=True).unique()
# node_features = np.random.randn(len(a), 128)
# node_features = pd.DataFrame(node_features, index=a)
# G = sg.StellarGraph(nodes=node_features,edges=df)
# edge_splitter_test = EdgeSplitter(G)
# G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
#     p=0.9, method="global", keep_connected=True
# )
# test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
# test_flow = test_gen.flow(edge_ids_test, edge_labels_test)
# a=model.evaluate(test_flow)
# a_id=np.array(G.edges()).tolist()
# b=np.ones((len(a_id),)).tolist()
# gen = GraphSAGELinkGenerator(G, batch_size, num_samples)
# flow = gen.flow(a_id, b)
# metrics = model.predict(flow).tolist()
# m = keras.metrics.AUC(num_thresholds=200)
# m.update_state(b, metrics)
# asdf=m.result().numpy()
# train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)
#
# print("\nMetrics of the final model:")
# for name, val in zip(model.metrics_names, metrics):
#     print("\t{}: {:0.4f}".format(name, val))



