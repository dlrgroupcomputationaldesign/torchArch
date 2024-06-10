import unittest

import torch

from torchdrug import data


class RoomTest(unittest.TestCase):

    def setUp(self):
        self.item_feature = torch.tensor([[1],[1],[2]], dtype=torch.float32)
        self.adj_matrix = torch.tensor([[0, 1, 0],
                                        [1, 0, 1],
                                        [0, 1, 0]], dtype=torch.float32)

    def test_room_definition(self):
        room = data.Room.from_room_graph_definition(self.item_feature, self.adj_matrix)
        self.assertTrue(room.adjacency.shape == torch.Size([3,3,1]), "Incorrect adjacency matrix")
        self.assertTrue(room.item_feature.shape == torch.Size([3,1]), "Incorrect item feature shape")
        self.assertTrue(room.edge_list.shape == torch.Size([4,3]))

    # def test_feature(self):
    #     mol = data.Molecule.from_smiles(self.smiles, mol_feature="ecfp")
    #     self.assertTrue((mol.graph_feature > 0).any(), "Incorrect ECFP feature")

class RoomDatasetTest(unittest.TestCase):
    def setUp(self):
        self.blob_path = "near-room-graph-classroom-dist-5"
        self.export_id = "20240530144326"
    def test_load(self):
        rooms_dataset = data.dataset.RoomLayoutDataset()
        rooms_dataset = rooms_dataset.load_blobs_dataset(self.blob_path, self.export_id)
        self.assertTrue(rooms_dataset.num_rooms == 13)

if __name__ == "__main__":
    unittest.main()