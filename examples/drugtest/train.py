import torch
import pickle
from torchdrug import core, models, tasks
from torch import nn, optim

print("Pulling in Mol data")
with open("/home/azureuser/zinc250k.pkl", "rb") as fin:
    dataset = pickle.load(fin)


model = models.RGCN(input_dim=dataset.node_feature_dim,
                    num_relation=dataset.num_bond_type,
                    hidden_dims=[256, 256, 256, 256], batch_norm=False)
task = tasks.GCPNGeneration(model, dataset.atom_types, max_edge_unroll=12,
                            max_node=38, criterion="nll")



optimizer = optim.Adam(task.parameters(), lr = 1e-3)
solver = core.Engine(task, dataset, None, None, optimizer,
                     gpus=(0,), batch_size=128, log_interval=10)

solver.train(num_epoch=1)

results = task.generate(num_sample=32, max_resample=5)

print(results)