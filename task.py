import pandas as pd
import numpy as np
import pydot
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from causica.lightning.data_modules.basic_data_module import BasicDECIDataModule
from causica.datasets.causica_dataset_format import Variable

from pytorch_lightning.callbacks import TQDMProgressBar

from causica.distributions import ContinuousNoiseDist
from causica.lightning.modules.deci_module import DECIModule
from causica.sem.sem_distribution import SEMDistributionModule
from causica.training.auglag import AugLagLRConfig
from causica.datasets.variable_types import VariableTypeEnum
import os
import argparse
import logging
import base64


parser = argparse.ArgumentParser(description='PyTorch Causal Training')
parser.add_argument('--data_csv', dest='data_csv',
                    type=str, help='Storage location of training CSV')
parser.add_argument('--model-dir', dest='model_dir',
                    default=os.getenv('AIP_MODEL_DIR'), type=str, help='Model directory')
parser.add_argument('--model-id', dest='model_id', type=str, help='Model id')                    
parser.add_argument('--batch_size', dest='batch_size',
                    type=int, default=128, help='Batch size')
parser.add_argument('--epochs', dest='epochs',
                    type=int, default=300, help='Number of epochs')
parser.add_argument('--lr', dest='lr',
                    type=float, default=1e-3, help='Learning rate')
parser.add_argument('--dtype', dest='dtype',
                    type=str, default=None, help='Data type for all columns in order, ex. CccBc for Categorical, Continuous Continuous Binary Continuous')                    
parser.add_argument('--constraint', dest='constraint',
                    type=str, default='', help='Edge Constraint, {"P": [(0, 1), (2, 3)], "N":[(4, 5)]} means there are must edges between column 2 and 3, column 0 and 1. There are must no edge between column 4 and 5.')
args = parser.parse_args()


# Load data
logging.info('importing training data')
gs_prefix = 'gs://'
gcsfuse_prefix = '/gcs/'

if args.data_csv.startswith(gs_prefix):
    args.data_csv.replace(gs_prefix, gcsfuse_prefix)

data = pd.read_csv(args.data_csv)
data = data.astype(float)
n_nodes = data.shape[-1]

# Building constraint_matrix 
# np.nan -> no constraint
# 0.0 -> Must not related
# 1.0 -> Must related
logging.info('loading constraints')
num_nodes = len(data.keys())

if len(args.constraint) == 0:
    constraint_matrix = np.full((num_nodes, num_nodes), np.nan, dtype=np.float32)
else:
    r = base64.decodebytes(bytes(args.constraint, "utf-8"))
    constraint_matrix = np.frombuffer(r, dtype=np.float32).reshape(num_nodes, num_nodes)
    constraint_matrix = constraint_matrix.copy()
    constraint_matrix[constraint_matrix == -1] = np.nan

# Building Variables Spec
# group_name -> node
# name -> for categorical data
# type -> VariableTypeEnum.CONTINUOUS, VariableTypeEnum.BINARY, VariableTypeEnum.CATEGORICAL 
# The CATEGORICAL is for higher level API of causica. For categorical data, please use group_name + name and BIANRY.
logging.info('Build dtype dictionary.')
C2TYPE = {
    "C": VariableTypeEnum.CATEGORICAL,
    "c": VariableTypeEnum.CONTINUOUS,
    "B": VariableTypeEnum.BINARY,
}

variables_spec = []

if args.dtype is None:
    variables_spec = [{"group_name": name, "name": name, "type": VariableTypeEnum.CONTINUOUS} for name, _ in zip(data.columns ,list(range(n_nodes)))]
else:
    variables_spec = [{"group_name": name, "name": name, "type": C2TYPE[x]} for name, x in zip(data.columns, args.dtype)]
    
data_module = BasicDECIDataModule(
    data,  # remove ground truth from dataframe
    variables=[Variable.from_dict(d) for d in variables_spec], # Same order as the data.columns
    batch_size=args.batch_size,
    normalize=True, # Should normalize all CONTINUOUS or not
)
normalizer = data_module.normalizer    
pl.seed_everything(seed=1) # Fix random seed

lightning_module = DECIModule(
    noise_dist=ContinuousNoiseDist.GAUSSIAN, # GAUSSIAN or SPLINE, SPLINE is pretty slow.
    prior_sparsity_lambda=43.0,
    init_rho=30.0,
    init_alpha=0.20,
    auglag_config=AugLagLRConfig(
        max_inner_steps=3400,
        max_outer_steps=8,
        lr_init_dict={
            "icgnn": 0.00076,
            "vardist": 0.0098,
            "functional_relationships": 3e-4,
            "noise_dist": 0.0070,
        },
    ),
)

# Inject the constraint_matrix
lightning_module.constraint_matrix = torch.tensor(constraint_matrix)

logging.info(f'Start Training.')
trainer = pl.Trainer(
    accelerator="auto", # gpu, cpu or auto
    max_epochs=args.epochs,
    fast_dev_run=False,
    callbacks=[TQDMProgressBar(refresh_rate=19)],
    enable_checkpointing=False,
)

trainer.fit(lightning_module, datamodule=data_module)

if args.model_dir.startswith(gs_prefix):
    args.model_dir = args.model_dir.replace(gs_prefix, gcsfuse_prefix)
    dirpath = os.path.split(args.model_dir)[0]
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

SEM_MODULE = lightning_module.sem_module()

REMOVE_CYCLE = True

def h(W):
    n = W.shape[0]
    Wn = W
    for i in range(n):
        if np.trace(Wn) != 0:
            return True
        Wn = Wn@W
    return False

if REMOVE_CYCLE:
    # Get the probability and reform the adj-matrix from the edge with highest probability. Skip the edge if it will cause a cycle.
    logits = SEM_MODULE._adjacency_dist.dist._get_independent_bernoulli_logits()
    probability = torch.sigmoid(logits).detach().cpu().numpy()
    probability[np.eye(num_nodes, dtype=bool)] = 0.0
    # print(np.sort(probability.flatten())[::-1])
    # exit()
    weight = probability[probability > 0.54]
    from_, to_ = np.where(probability > 0.54)
    w_order = np.argsort(weight)[::-1]
    from_ = from_[w_order]
    to_ = to_[w_order]
    
    new_adj = np.zeros_like(probability)
    for f, t in zip(from_, to_):
        new_adj[f, t] = 1
        if h(new_adj):
            new_adj[f, t] = 0
            continue
            
    new_adj = new_adj.astype(np.float32)
    
    # Inject the constraint.
    new_adj = 1.0 - (1.0 - new_adj * SEM_MODULE._adjacency_dist.negative_constraints.cpu().numpy()) * (~SEM_MODULE._adjacency_dist.positive_constraints.cpu().numpy())

    new_adj = torch.as_tensor(new_adj)
            
else:
    new_adj = SEM_MODULE.mode.graph

torch.save({"model": lightning_module.sem_module, "adj": new_adj, "normalizer": normalizer}, os.path.join(args.model_dir, args.model_id) + '.pt')
logging.info(f'Model is saved to {args.model_dir}, model_id is {args.model_id}')
pyd_graph = pydot.Dot('my_graph', graph_type='digraph', bgcolor='white')

rows, cols = np.where(new_adj.detach().cpu().numpy() != 0)
edges = list(zip(rows.tolist(), cols.tolist()))

labels = list(data.columns)
for name in labels:
    pyd_graph.add_node(pydot.Node(name, label=name, fontname="Microsoft YeHei"))
   
for r, c in edges:
    pyd_graph.add_edge(pydot.Edge(labels[r], labels[c]))
   
logging.info("writing!")
pyd_graph.write_png(os.path.join(args.model_dir, args.model_id) + '.png', encoding="utf8")

