import torch
from torchdrug import datasets
import pickle

dataset = datasets.ZINC250k("~/molecule-datasets/", kekulize=True,
                            atom_feature="symbol")



with open("zinc50k3.pkl", "wb") as fout:
    pickle.dump(dataset, fout)