import os
import numpy as np
import pandas as pd
import pickle
import torch

from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src import const


from pdb import set_trace


def read_sdf(sdf_path):
    with Chem.SDMolSupplier(sdf_path, sanitize=False) as supplier:
        for molecule in supplier:
            yield molecule


def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot


def parse_molecule(mol, is_geom):
    one_hot = []
    charges = []
    atom2idx = const.ATOM2IDX
    charges_dict = const.CHARGES
    for atom in mol.GetAtoms():
        one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
        charges.append(charges_dict[atom.GetSymbol()])
    positions = mol.GetConformer().GetPositions()
    return positions, np.array(one_hot), np.array(charges)


class ZincDataset(Dataset):
    def __init__(self, data_path, prefix, device):
        dataset_path = os.path.join(data_path, f'{prefix}.pt')
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f'Preprocessing dataset with prefix {prefix}')
            self.data = ZincDataset.preprocess(data_path, prefix, device)
            torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def preprocess(data_path, prefix, device):
        data = []
        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        fragments_path = os.path.join(data_path, f'{prefix}_frag.sdf')
        linkers_path = os.path.join(data_path, f'{prefix}_link.sdf')

        is_geom = ('geom' in prefix) or ('MOAD' in prefix)
        # is_multifrag = 'multifrag' in prefix

        table = pd.read_csv(table_path)
        generator = tqdm(zip(table.iterrows(), read_sdf(fragments_path), read_sdf(linkers_path)), total=len(table))
        for (_, row), fragments, linker in generator:
            uuid = row['uuid']
            name = row['molecule']
            frag_pos, frag_one_hot, frag_charges = parse_molecule(fragments, is_geom=is_geom)
            link_pos, link_one_hot, link_charges = parse_molecule(linker, is_geom=is_geom)

            if ''!=Chem.MolToSmiles(linker):
                positions = np.concatenate([frag_pos, link_pos], axis=0) #pocket_pos
                one_hot = np.concatenate([frag_one_hot, link_one_hot], axis=0) #pocket_one_hot
            else:
                positions = np.concatenate([frag_pos], axis=0) #pocket_pos
                one_hot = np.concatenate([frag_one_hot], axis=0) #pocket_one_hot
                link_one_hot = [[0.] * len(frag_one_hot[0])]          
            charges = np.concatenate([frag_charges, link_charges], axis=0) #pocket_charges
            anchors = np.zeros_like(charges)

            

            fragment_mask = np.concatenate([np.ones_like(frag_charges), np.zeros_like(link_charges)])
            linker_mask = np.concatenate([np.zeros_like(frag_charges), np.ones_like(link_charges)])

            # fragment_only_mask = np.concatenate([
            #     np.ones_like(frag_charges),
            #     # np.zeros_like(pocket_charges),
            #     np.zeros_like(link_charges)
            # ])
            # pocket_mask = np.concatenate([
            #     np.zeros_like(frag_charges),
            #     # np.ones_like(pocket_charges),
            #     np.zeros_like(link_charges)
            # ])
            linker_mask = np.concatenate([
                np.zeros_like(frag_charges),
                # np.zeros_like(pocket_charges),
                np.ones_like(link_charges)
            ])
            fragment_mask = np.concatenate([
                np.ones_like(frag_charges),
                # np.ones_like(pocket_charges),
                np.zeros_like(link_charges)
            ])
            anchors[row['anchors']] = 1
            data.append({
                'uuid': uuid,
                'name': name,
                'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
                'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
                'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
                'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device),
                'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
                'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device),
                'num_atoms': len(positions),
            })
        return data




def collate(batch):
    out = {}
    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')

    atom_mask = (out['fragment_mask'].bool() | out['linker_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]

    batch_size, n_nodes = atom_mask.size()

    # In case of MOAD edge_mask is batch_idx
    # if 'pocket_mask' in batch[0].keys():
    # batch_mask = torch.cat([
    #         torch.ones(n_nodes, dtype=const.TORCH_INT) * i
    #         for i in range(batch_size)
    #     ]).to(atom_mask.device)
    # out['edge_mask'] = batch_mask
    # else:
    edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
    diag_mask = 1 - torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=atom_mask.device).unsqueeze(0)
    edge_mask *= diag_mask
    out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out


def collate_with_fragment_edges(batch):
    out = {}
    # Filter out big molecules
    # batch = [data for data in batch if data['num_atoms'] <= 50]

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)
    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')

    frag_mask = out['fragment_mask']
    edge_mask = frag_mask[:, None, :] * frag_mask[:, :, None]
    diag_mask = 1 - torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=frag_mask.device).unsqueeze(0)
    edge_mask *= diag_mask

    batch_size, n_nodes = frag_mask.size()
    out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    # Building edges and covalent bond values
    rows, cols, bonds = [], [], []
    for batch_idx in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i + batch_idx * n_nodes)
                cols.append(j + batch_idx * n_nodes)

    edges = [torch.LongTensor(rows).to(frag_mask.device), torch.LongTensor(cols).to(frag_mask.device)]
    out['edges'] = edges
    atom_mask = (out['fragment_mask'].bool() | out['linker_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out


def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle)


def create_template(tensor, fragment_size, linker_size, fill=0):
    values_to_keep = tensor[:fragment_size]
    values_to_add = torch.ones(linker_size, tensor.shape[1], dtype=values_to_keep.dtype, device=values_to_keep.device)
    values_to_add = values_to_add * fill
    return torch.cat([values_to_keep, values_to_add], dim=0)


def create_templates_for_linker_generation(data, linker_sizes):
    """
    Takes data batch and new linker size and returns data batch where fragment-related data is the same
    but linker-related data is replaced with zero templates with new linker sizes
    """
    decoupled_data = []
    for i, linker_size in enumerate(linker_sizes):
        data_dict = {}
        fragment_mask = data['fragment_mask'][i].squeeze()
        fragment_size = fragment_mask.sum().int()
        for k, v in data.items():
            if k == 'num_atoms':
                # Computing new number of atoms (fragment_size + linker_size)
                data_dict[k] = fragment_size + linker_size
                continue
            if k in const.DATA_LIST_ATTRS:
                # These attributes are written without modification
                data_dict[k] = v[i]
                continue
            if k in const.DATA_ATTRS_TO_PAD:
                # Should write fragment-related data + (zeros x linker_size)
                fill_value = 1 if k == 'linker_mask' else 0
                template = create_template(v[i], fragment_size, linker_size, fill=fill_value)
                if k in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                    template = template.squeeze(-1)
                data_dict[k] = template

        decoupled_data.append(data_dict)

    return collate(decoupled_data)
