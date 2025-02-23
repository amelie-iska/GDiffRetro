import csv
import math
import logging
import warnings
from collections import defaultdict

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch.utils import data as torch_data

from torchdrug import core, data, utils
#from torchdrug.utils import doc


logger = logging.getLogger(__name__)


class MoleculeDataset(torch_data.Dataset, core.Configurable):
    """
    Molecule dataset.

    Each sample contains a molecule graph, and any number of prediction targets.
    """

    #@doc.copy_args(data.Molecule.from_molecule)
    @utils.copy_args(data.Molecule.from_molecule)
    def load_smiles(self, smiles_list, targets, transform=None, lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from SMILES and targets.

        Parameters:
            smiles_list (list of str): SMILES strings
            targets (dict of list): prediction targets
            transform (Callable, optional): data transformation function
            lazy (bool, optional): if lazy mode is used, the molecules are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = len(smiles_list)
        if num_sample > 1000000:
            warnings.warn("Preprocessing molecules of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_smiles(lazy=True) to construct molecules in the dataloader instead.")
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError("Number of target `%s` doesn't match with number of molecules. "
                                 "Expect %d but found %d" % (field, num_sample, len(target_list)))

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.smiles_list = []
        self.data = []
        self.targets = defaultdict(list)

        if verbose:
            smiles_list = tqdm(smiles_list, "Constructing molecules from SMILES")
        
        for i, smiles in enumerate(smiles_list):
            if not self.lazy or len(self.data) == 0:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    logger.debug("Can't construct molecule from SMILES `%s`. Ignore this sample." % smiles)
                    continue
                mol = data.Molecule.from_molecule(mol, **kwargs)
            else:
                mol = None
            self.data.append(mol)
            self.smiles_list.append(smiles)
            for field in targets:
                self.targets[field].append(targets[field][i])

    #@doc.copy_args(load_smiles)
    @utils.copy_args(load_smiles)
    def load_csv(self, csv_file, smiles_field="smiles", target_fields=None, verbose=0, **kwargs):
        """
        Load the dataset from a csv file.

        Parameters:
            csv_file (str): file name
            smiles_field (str, optional): name of SMILES column in the table.
                Use ``None`` if there is no SMILES column.
            target_fields (list of str, optional): name of target columns in the table.
                Default is all columns other than the SMILES column.
            verbose (int, optional): output verbose level
            **kwargs
        """
        if target_fields is not None:
            target_fields = set(target_fields)

        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))
            fields = next(reader)
            smiles = []
            targets = defaultdict(list)
            for values in reader:
                if not any(values):
                    continue
                if smiles_field is None:
                    smiles.append("")
                for field, value in zip(fields, values):
                    if field == smiles_field:
                        smiles.append(value)
                    elif target_fields is None or field in target_fields:
                        value = utils.literal_eval(value)
                        if value == "":
                            value = math.nan
                        targets[field].append(value)

        self.load_smiles(smiles, targets, verbose=verbose, **kwargs)

    def _standarize_index(self, index, count):
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = range(start, stop, step)
        elif not isinstance(index, list):
            raise ValueError("Unknown index `%s`" % index)
        return index

    def get_item(self, index):
        if getattr(self, "lazy", False):
            item = {"graph": data.Molecule.from_smiles(self.smiles_list[index], **self.kwargs)}
        else:
            item = {"graph": self.data[index]}
        item.update({k: v[index] for k, v in self.targets.items()})
        if self.transform:
            item = self.transform(item)
        return item

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item(index)

        index = self._standarize_index(index, len(self))
        return [self.get_item(i) for i in index]

    @property
    def tasks(self):
        """List of tasks."""
        return list(self.targets.keys())

    @property
    def node_feature_dim(self):
        """Dimension of node features."""
        return self.data[0].node_feature.shape[-1]

    @property
    def edge_feature_dim(self):
        """Dimension of edge features."""
        return self.data[0].edge_feature.shape[-1]

    @property
    def num_atom_type(self):
        """Number of different atom types."""
        return len(self.atom_types)

    @property
    def num_bond_type(self):
        """Number of different bond types."""
        return len(self.bond_types)

    @utils.cached_property
    def atom_types(self):
        """All atom types."""
        atom_types = set()

        if getattr(self, "lazy", False):
            warnings.warn("Calling this function for dataset with lazy=True may take a large amount of time.")             
            for smiles in self.smiles_list:
                graph = data.Molecule.from_smiles(smiles, **self.kwargs)
                atom_types.update(graph.atom_type.tolist())
        else:
            for graph in self.data:
                atom_types.update(graph.atom_type.tolist())

        return sorted(atom_types)

    @utils.cached_property
    def bond_types(self):
        """All bond types."""
        bond_types = set()

        if getattr(self, "lazy", False):
            warnings.warn("Calling this function for dataset with lazy=True may take a large amount of time.")             
            for smiles in self.smiles_list:
                graph = data.Molecule.from_smiles(smiles, **self.kwargs)
                bond_types.update(graph.edge_list[:, 2].tolist())
        else:
            for graph in self.data:
                bond_types.update(graph.edge_list[:, 2].tolist())

        return sorted(bond_types)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: %d" % len(self.tasks),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))


class ReactionDataset(MoleculeDataset, core.Configurable):
    """
    Chemical reaction dataset.

    Each sample contains two molecule graphs, and any number of prediction targets.
    """

    #@doc.copy_args(data.Molecule.from_molecule)
    @utils.copy_args(data.Molecule.from_molecule)
    def load_smiles(self, smiles_list, targets, transform=None, verbose=0, **kwargs):
        """
        Load the dataset from SMILES and targets.

        Parameters:
            smiles_list (list of str): SMILES strings
            targets (dict of list): prediction targets
            transform (Callable, optional): data transformation function
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = len(smiles_list)
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError("Number of target `%s` doesn't match with number of molecules. "
                                 "Expect %d but found %d" % (field, num_sample, len(target_list)))

        self.smiles_list = smiles_list
        self.data = []
        self.targets = defaultdict(list)
        if verbose:
            smiles_list = tqdm(smiles_list, "Constructing molecules from SMILES")
        for i, smiles in enumerate(smiles_list):
            smiles_reactant, agent, smiles_product = smiles.split(">")
            mols = []
            for _smiles in [smiles_reactant, smiles_product]:
                mol = Chem.MolFromSmiles(_smiles)
                if not mol:
                    logger.debug("Can't construct molecule from SMILES `%s`. Ignore this sample." % _smiles)
                    break
                mol = data.Molecule.from_molecule(mol, **kwargs)
                mols.append(mol)
            else:
                self.data.append(mols)
                for field in targets:
                    self.targets[field].append(targets[field][i])
        self.transform = transform

    @property
    def node_feature_dim(self):
        """Dimension of node features."""
        return self.data[0][0].node_feature.shape[-1]

    @property
    def edge_feature_dim(self):
        """Dimension of edge features."""
        return self.data[0][0].edge_feature.shape[-1]

    @property
    def num_atom_type(self):
        """Number of different atom types."""
        return len(self.atom_types)

    @property
    def num_bond_type(self):
        """Number of different bond types."""
        return len(self.bond_types)

    @utils.cached_property
    def atom_types(self):
        """All atom types."""
        atom_types = set()
        for graphs in self.data:
            for graph in graphs:
                atom_types.update(graph.atom_type.tolist())
        return sorted(atom_types)

    @utils.cached_property
    def bond_types(self):
        """All bond types."""
        bond_types = set()
        for graphs in self.data:
            for graph in graphs:
                bond_types.update(graph.edge_list[:, 2].tolist())
        return sorted(bond_types)

    def __len__(self):
        return len(self.data)



def key_split(dataset, keys, lengths=None, key_lengths=None):

    def round_to_boundary(i):
        for j in range(min(i, len(dataset) - i)):
            if keys[indexes[i - j]] != keys[indexes[i - j - 1]]:
                return i - j
            if keys[indexes[i + j]] != keys[indexes[i + j - 1]]:
                return i + j
        if i < len(dataset) - i:
            return 0
        else:
            return len(dataset)

    keys = torch.as_tensor(keys)
    #print("key",keys)
    key_set, keys = torch.unique(keys, return_inverse=True)
    perm = torch.randperm(len(key_set))
    #print("lol", perm)
    keys = perm[keys]
    indexes = keys.argsort().tolist()

    if key_lengths is not None:
        assert lengths is None
        key2count = keys.bincount()
        key_offset = 0
        lengths = []
        for key_length in key_lengths:
            lengths.append(key2count[key_offset: key_offset + key_length].sum().item())
            key_offset += key_length

    offset = 0
    offsets = [offset]
    for length in lengths:
        offset = round_to_boundary(offset + length)
        offsets.append(offset)
    offsets[-1] = len(dataset)
    return [torch_data.Subset(dataset, indexes[offsets[i]: offsets[i + 1]]) for i in range(len(lengths))]


def scaffold_split(dataset, lengths):
    """
    Randomly split a dataset into new datasets with non-overlapping scaffolds.

    Parameters:
        dataset (Dataset): dataset to split
        lengths (list of int): expected length for each split.
            Note the results may be different in length due to rounding.
    """

    scaffold2id = {}
    keys = []
    for sample in dataset:
        scaffold = sample["graph"].to_scaffold()
        if scaffold not in scaffold2id:
            id = len(scaffold2id)
            scaffold2id[scaffold] = id
        else:
            id = scaffold2id[scaffold]
        keys.append(id)

    return key_split(dataset, keys, lengths)


def ordered_scaffold_split(dataset, lengths, chirality=True):
    """
    Split a dataset into new datasets with non-overlapping scaffolds and sorted w.r.t. number of each scaffold.

    Parameters:
        dataset (Dataset): dataset to split
        lengths (list of int): expected length for each split.
            Note the results may be different in length due to rounding.
    """
    frac_train, frac_valid, frac_test = 0.8, 0.1, 0.1

    scaffold2id = defaultdict(list)
    for idx, smiles in enumerate(dataset.smiles_list):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=chirality)
        scaffold2id[scaffold].append(idx)

    scaffold2id = {key: sorted(value) for key, value in scaffold2id.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffold2id.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    return torch_data.Subset(dataset, train_idx), torch_data.Subset(dataset, valid_idx), torch_data.Subset(dataset, test_idx)
