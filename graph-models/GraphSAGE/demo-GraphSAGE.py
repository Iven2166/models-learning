import os
import pandas as pd
import networkx as nx
import stellargraph as sg
from stellargraph.data import UnsupervisedSampler
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
import keras

cora_dir = '../../../dataset/cora'
cora_sites = os.path.join(cora_dir, 'cora.cites')
cora_features = os.path.join(cora_dir, 'cora.content')

# with open(cora_sites, "rb") as f:
#     for i, line in enumerate(f.readlines()):
#         print(line)
#         if i == 10:
#             break

edgelist = pd.read_csv(cora_sites, sep='\t', header=None, names=['target', 'source'])
edgelist['label'] = 'cites'  # set the edge type
Gnx = nx.from_pandas_edgelist(edgelist, edge_attr='label')

# Add content features
feature_names = ["w_{}".format(ii) for ii in range(1433)]
column_names = feature_names + ['subject']
node_data = pd.read_csv(cora_features, sep='\t', header=None, names=column_names)
node_features = node_data[feature_names]
# Create StellarGraph object
G = sg.StellarGraph(Gnx, node_features=node_features)

# Model Training
"""
UnsupervisedSampler class to sample a number of walks of given length
from the graph.
GraphSAGELinkGenerator to generate the edges needed in the loss function
"""
nodes = list(G.nodes())
number_of_walks = 1
length = 5
batch_size = 50
epochs = 4
num_samples = [10, 5]

unsuperviesd_samples = UnsupervisedSampler(G, nodes=nodes, length=length, number_of_walks=number_of_walks)
train_gen = GraphSAGELinkGenerator(G, batch_size, num_samples).flow(unsuperviesd_samples)

# define GraphSAGE model
layer_sizes = [50,50]
graphsage = GraphSAGE(layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.0, normalize='l2')
x_inp, x_out = graphsage.build()
prediction = link_classification(output_dim=1, output_act='sigmoid', edge_embedding_method='ip')(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)
history = model.fit_generator(
    train_gen,
    epochs=epochs,
    verbose=1,
    use_multiprocessing=False,
    workers=4,
    shuffle=True,
)


x_inp_src = x_inp[0::2]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
node_ids = node_data.index
node_gen = GraphSAGENodeGenerator(G, batch_size,num_samples).flow(node_ids)
node_embeddings = embedding_model.predict_generator(node_gen, workers=4, verbose=1)