import os.path
import sys
import torch
import torchdrug
from torchdrug import data, datasets, utils
from torchdrug import core, models, tasks
from torch.utils import data as torch_data
import torchdrug.utils.comm as dist
import rdkit

data_path = "./dataset"
uspto_train_csv_path = os.path.join(data_path, "selected_train.csv")
uspto_valid_csv_path = os.path.join(data_path, "selected_eval.csv")
uspto_test_csv_path = os.path.join(data_path, "selected_test.csv")

reaction_dataset_train = datasets.USPTO50k(uspto_train_csv_path,
                                     atom_feature="center_identification", as_synthon=False,
                                     with_hydrogen=False, kekulize=True, verbose=1)

reaction_dataset_valid = datasets.USPTO50k(uspto_valid_csv_path,
                                     atom_feature="center_identification", as_synthon=False,
                                     with_hydrogen=False, kekulize=True, verbose=1)

reaction_dataset_test = datasets.USPTO50k(uspto_test_csv_path,
                                     atom_feature="center_identification", as_synthon=False,
                                     with_hydrogen=False, kekulize=True, verbose=1)

reaction_model = models.RGCN(input_dim=reaction_dataset_train.node_feature_dim,
                             hidden_dims=[512, 512, 512],
                             batch_norm=False, short_cut=True,
                             num_relation=reaction_dataset_train.num_bond_type,
                             concat_hidden=False)

dual_model = models.RGCN(input_dim=reaction_dataset_train.node_feature_dim,
                         hidden_dims=[512, 512, 512],
                         batch_norm=False, short_cut=True,
                         num_relation=reaction_dataset_train.num_bond_type,
                         concat_hidden=False)

reaction_task = tasks.CenterIdentification(reaction_model, dual_model,
                                           feature=("graph", "atom", "bond"))

reaction_optimizer = torch.optim.Adam(reaction_task.parameters(), lr=1e-4, weight_decay=0)
reaction_solver = core.Engine(reaction_task, reaction_train, reaction_valid,
                              reaction_test, reaction_optimizer, gpus=[0],
                              batch_size=64, gradient_interval=1, log_interval=100)

reaction_solver.train(num_epoch=50)
reaction_solver.evaluate("valid")
reaction_solver.evaluate("test")
reaction_solver.save("./model/reaction_center_model_wo_class.pth")
