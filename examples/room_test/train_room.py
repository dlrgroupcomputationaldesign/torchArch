import torch
import pickle
from torchdrug import core, models, tasks
from torchdrug.data.dataset import RoomLayoutDataset
from torch import nn, optim

print("Pulling in Room data")
dataset = RoomLayoutDataset()
dataset = dataset.load_blobs_dataset(blob_container_path="near-room-graph-classroom-dist-5", 
                                               export_id="20240606171120")


model = models.RGCN(input_dim=dataset.node_feature_dim,
                    num_relation=dataset.num_bond_type,
                    hidden_dims=[256, 256, 256, 256], batch_norm=False)
task = tasks.GCPNGeneration(model, dataset.item_types, max_edge_unroll=12,
                            max_node=50, criterion="nll", node_type="item")



optimizer = optim.Adam(task.parameters(), lr = 1e-3)
solver = core.Engine(task, dataset, None, None, optimizer,
                     gpus=(0,), batch_size=8, log_interval=10)

solver.train(num_epoch=30)

results = task.generate(num_sample=32, max_resample=5, arch=True, max_step=300)
results.display_info()