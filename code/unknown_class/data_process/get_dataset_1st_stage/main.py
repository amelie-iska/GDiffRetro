import pandas as pd
from rdkit.Chem import PandasTools
import numpy as np
import torch
import torch
import sys
from rdkit import Chem


from rdkit.Chem import PandasTools
from sklearn.model_selection import train_test_split
from torchdrug import data, datasets, utils
from torchdrug import core, models, tasks
from torch.utils import data as torch_data
import torchdrug.utils.comm as dist
from torchdrug import core, tasks, data, metrics, transforms
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug import layers
import rdkit
from rdkit import rdBase
rdBase.rdkitVersion
import logging
logger = logging.getLogger(__name__)
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
import inspect
from collections import deque
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torch_data
from torch_scatter import scatter_max, scatter_add

mode = "train"

reaction_dataset = datasets.USPTO50k(   "~/molecule-datasets/",
                                        outputpath = "~/molecule-datasets/index_product_part1_" + mode + ".csv",
                                        used_path = "./cmp_data/selected_" + mode + ".csv",
                                        prepare_dataset=False, 
                                        atom_feature="center_identification",
                                        kekulize=True
                                    )

reaction_dataset = datasets.USPTO50k(   "~/molecule-datasets/",
                                        outputpath = "~/molecule-datasets/index_product_part1_" + mode + ".csv",
                                        used_path = "./cmp_data/selected_" + mode + ".csv",
                                        prepare_dataset=True, 
                                        atom_feature="center_identification",
                                        kekulize=True
                                    )

synthon_dataset = datasets.USPTO50k(    "~/molecule-datasets/", 
                                        outputpath = "~/molecule-datasets/index_product_part2_" + mode + ".csv",
                                        used_path = "./cmp_data/selected_" + mode + ".csv",
                                        as_synthon=True, 
                                        prepare_dataset=True,
                                        atom_feature="synthon_completion",
                                        kekulize=True
                                    )

df1 = pd.read_csv('~/molecule-datasets/index_product_part1_' + mode + '.csv', sep=',', header=None, usecols=[0, 2, 3])
df2 = pd.read_csv('~/molecule-datasets/index_product_part2_' + mode + '.csv', sep=',', header=None)

merged_df = pd.merge(df2[1:], df1[1:], left_on = 0, right_on = 0, how = 'left')
merged_df.columns = ['index', 'molecule', 'fragments', 'linker', 'product', 'center1', 'center2']
merged_df = merged_df.reindex(columns=['index', 'molecule', 'fragments', 'linker', 'product', 'center1', 'center2'])

NAN_list = merged_df[merged_df.iloc[:,-1].isna()].index.tolist()
print('Detect #NaN', len(NAN_list))
merged_df = merged_df.drop(index=NAN_list)

print('After Delete #NaN', len(merged_df[merged_df.iloc[:,-1].isna()].index.tolist()))
merged_df = merged_df.drop('index', axis=1)
merged_df.to_csv('./molecule-datasets/merge_final_' + mode + '.csv', index=True, index_label='id')

table = pd.read_csv('./molecule-datasets/merge_final_' + mode + '.csv', index_col = False)
table["anchors"] = 0
table["linksize"] = 0
for i in range(len(table)):
    print(i)
    data = table.loc[i]
    molecule_smi = data["molecule"]
    frag_smi = data["fragments"]
    center1 = int(data['center1'])
    center2 = int(data['center2'])

    frag = Chem.MolFromSmiles(frag_smi)
    molecule = Chem.MolFromSmiles(molecule_smi)
    table.loc[i, "linksize"] = molecule.GetNumHeavyAtoms() - frag.GetNumHeavyAtoms()

    for atom in frag.GetAtoms():
            if atom.GetAtomMapNum()==center1 or atom.GetAtomMapNum()==center2:
                table.loc[i, "anchors"] = atom.GetIdx()
                break   
table.to_csv("./molecule-datasets/merge_final_" + mode + ".csv", index=False)



