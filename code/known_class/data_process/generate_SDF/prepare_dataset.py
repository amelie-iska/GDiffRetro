import argparse
import itertools
import numpy as np
import pandas as pd
from rdkit import Chem, Geometry
from tqdm import tqdm
from pdb import set_trace
from rdkit.Chem import Draw
from rdkit import Chem
from torchdrug import data as dt
from torchdrug import datasets, utils
from torchdrug.datasets import  uspto50k
from torchdrug import core, models, tasks
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

def _get_difference(reactant, product):
        product2id = product.atom_map
        id2reactant = torch.zeros(product2id.max() + 1, dtype=torch.long)
        id2reactant[reactant.atom_map] = torch.arange(reactant.num_node)
        prod2react = id2reactant[product2id]
        product = product.directed()
        mapped_edge = product.edge_list.clone()
        mapped_edge[:, :2] = prod2react[mapped_edge[:, :2]]
        is_same_index = mapped_edge.unsqueeze(0) == reactant.edge_list.unsqueeze(1)
        has_typed_edge = is_same_index.all(dim=-1).any(dim=0)
        has_edge = is_same_index[:, :, :2].all(dim=-1).any(dim=0)
        is_added = ~has_edge
        is_modified = has_edge & ~has_typed_edge
        edge_added = product.edge_list[is_added, :2]
        edge_modified = product.edge_list[is_modified, :2]
        
        return edge_added, edge_modified, prod2react


def _get_reaction_center(reactant, product):
        edge_added, edge_modified, prod2react = _get_difference(reactant, product)
        edge_label = torch.zeros(product.num_edge, dtype=torch.long)
        node_label = torch.zeros(product.num_node, dtype=torch.long)

        if len(edge_added) > 0:
            print('edge_added')
            if len(edge_added) == 1:
                any = -torch.ones(1, 1, dtype=torch.long)
                pattern = torch.cat([edge_added, any], dim=-1)
                index, num_match = product.match(pattern)
                assert num_match.item() == 1
                edge_label[index] = 1
                h, t = edge_added[0]
                reaction_center = torch.tensor([product.atom_map[h], product.atom_map[t]])
            else:
                print('several_edges')
        else:
            if len(edge_modified) == 1:
                print('edge modified')
                h, t = edge_modified[0]
                if product.degree_in[h] == 1:
                    node_label[h] = 1
                    reaction_center = torch.tensor([product.atom_map[h], 0])

                elif product.degree_in[t] == 1:
                    node_label[t] = 1
                    reaction_center = torch.tensor([product.atom_map[t], 0])
                    
                else:
                    node_label[h] = 1
                    reaction_center = torch.tensor([product.atom_map[h], 0])
                    
            else:
                print('atom modified')
                product_hs = torch.tensor([atom.GetTotalNumHs() for atom in product.to_molecule().GetAtoms()])
                reactant_hs = torch.tensor([atom.GetTotalNumHs() for atom in reactant.to_molecule().GetAtoms()])
                atom_modified = (product_hs != reactant_hs[prod2react]).nonzero().flatten()

                if len(atom_modified)==1:
                    node_label[atom_modified] = 1
                    reaction_center = torch.tensor([product.atom_map[atom_modified[0]], 0])
                elif len(atom_modified)>1:
                    node_label[atom_modified] = 1
                    reaction_center = torch.tensor([product.atom_map[atom_modified[0]], 0])
                else: print('no changes')          

        if edge_label.sum() + node_label.sum() == 0:
            return [], []
            
        with product.edge():
            product.edge_label = edge_label
        with product.node():
            product.node_label = node_label
        with reactant.graph():
            reactant.reaction_center = reaction_center
        with product.graph():
            product.reaction_center = reaction_center
        
        return [reactant], [product]
        
def get_exits(mol):
    """
    Returns atoms marked as exits in DeLinker data
    """
    exits = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == '*':
            exits.append(atom)
    return exits


def set_anchor_flags(mol, anchor_idx):
    """
    Sets property _Anchor to all atoms in a molecule
    """
    for atom in mol.GetAtoms():
        if atom.GetIdx() == anchor_idx:
            atom.SetProp('_Anchor', '1')
        else:
            atom.SetProp('_Anchor', '0')


def get_anchors_idx(mol):
    anchors_idx = []
    for atom in mol.GetAtoms():
        if atom.GetProp('_Anchor') == '1':
            anchors_idx.append(atom.GetIdx())

    return anchors_idx


def update_fragment(frag):
    """
    Removes exit atoms with corresponding bonds and sets _Anchor property
    """
    exits = get_exits(frag)
    if len(exits) > 1:
        raise Exception('Found more than one exits in fragment')
    exit = exits[0]

    bonds = exit.GetBonds()
    if len(bonds) > 1:
        raise Exception('Exit atom has more than 1 bond')
    bond = bonds[0]

    exit_idx = exit.GetIdx()
    source_idx = bond.GetBeginAtomIdx()
    target_idx = bond.GetEndAtomIdx()
    anchor_idx = source_idx if target_idx == exit_idx else target_idx
    set_anchor_flags(frag, anchor_idx)

    efragment = Chem.EditableMol(frag)
    efragment.RemoveBond(source_idx, target_idx)
    efragment.RemoveAtom(exit_idx)

    return efragment.GetMol()


def update_linker(linker):
    """
    Removes exit atoms with corresponding bonds
    """
    exits = get_exits(linker)
    if len(exits) > 2:
        raise Exception('Found more than two exits in linker')

    exits = sorted(exits, key=lambda e: e.GetIdx(), reverse=True)
    elinker = Chem.EditableMol(linker)

    for exit in exits:
        bonds = exit.GetBonds()
        if len(bonds) > 1:
            raise Exception('Exit atom has more than 1 bond')
        bond = bonds[0]
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        elinker.RemoveBond(source_idx, target_idx)

    for exit in exits:
        elinker.RemoveAtom(exit.GetIdx())

    return elinker.GetMol()


def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer

def transfer_conformers(frag, lin, mol):
    """
    Computes coordinates from molecule to fragment (for all matchings)
    """

    molecule = dt.Molecule.from_molecule(mol)
    fragment = dt.Molecule.from_molecule(frag)
    _, _, prod2react1 = _get_difference(molecule, fragment)
    matches_lin = mol.GetSubstructMatches(lin)
    matches_frag = (tuple(prod2react1.tolist()),)
    if len(matches_lin) < 1:
        print("suka", match)
        raise Exception('Could not find fragment or linker matches')

    match2conf_frag = {}
    for match in matches_frag:
        mol_coords = mol.GetConformer().GetPositions()
        frag_coords = mol_coords[np.array(match)]
        frag_conformer = create_conformer(frag_coords)
        match2conf_frag[match] = frag_conformer

    match2conf_lin = {}
    for match in matches_lin:
        mol_coords = mol.GetConformer().GetPositions()
        frag_coords = mol_coords[np.array(match)]
        frag_conformer = create_conformer(frag_coords)
        match2conf_lin[match] = frag_conformer
    return match2conf_frag, match2conf_lin

def transfer_conformers_empty_lin(frag, mol, mol_smi):
    """
    Computes coordinates from molecule to fragment (for all matchings)
    """
    molecule = dt.Molecule.from_smiles(mol_smi)
    fragment = dt.Molecule.from_molecule(frag)

    _, _, prod2react1 = _get_difference(molecule, fragment)
    
    matches_frag = (tuple(prod2react1.tolist()),)
    match2conf_frag = {}
    for match in matches_frag:
        mol_coords = mol.GetConformer().GetPositions()
        frag_coords = mol_coords[np.array(match)]
        frag_conformer = create_conformer(frag_coords)
        match2conf_frag[match] = frag_conformer

    return match2conf_frag

def find_non_intersecting_matches(matches1, matches3):
    """
    Checks all possible triplets and selects only non-intersecting matches
    """
    triplets = list(itertools.product(matches1, matches3))
    non_intersecting_matches = set()
    for m1,  m3 in triplets:
        m1m3 = set(m1) & set(m3)
        if len( m1m3) == 0:
            non_intersecting_matches.add((m1, m3))
    return list(non_intersecting_matches)


def find_non_intersecting_matches1(matches1):
    """
    Checks all possible triplets and selects only non-intersecting matches
    """
    triplets = list(itertools.product(matches1))
    non_intersecting_matches = set()
    for m1 in triplets:
        non_intersecting_matches.add((m1))
    return list(non_intersecting_matches)


def find_matches_with_linker_in_the_middle(non_intersecting_matches, mol):
    """
    Selects only matches where linker is between fragments
    I.e. each fragment should have one atom that is included in the set of neighbors of all linker atoms
    """
    matches_with_linker_in_the_middle = []
    for m1, m2, lm in non_intersecting_matches:
        neighbors = set()
        for atom_idx in lm:
            atom_neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            for neighbor in atom_neighbors:
                neighbors.add(neighbor.GetIdx())

        conn1 = set(m1) & neighbors
        conn2 = set(m2) & neighbors
        if len(conn1) == 1 and len(conn2) == 1:
            matches_with_linker_in_the_middle.append((m1, m2, lm))

    return matches_with_linker_in_the_middle


def find_correct_matches(matches_frag1, matches_linker, mol):
    """
    Finds all correct fragments and linker matches
    """
    non_intersecting_matches = find_non_intersecting_matches(matches_frag1, matches_linker)
    if len(non_intersecting_matches) == 1:
        return non_intersecting_matches

    return find_matches_with_linker_in_the_middle(non_intersecting_matches, mol)


def find_correct_matches1(matches_frag1, mol):
    """
    Finds all correct fragments and linker matches
    """
    non_intersecting_matches = find_non_intersecting_matches1(matches_frag1)
    if len(non_intersecting_matches) == 1:
        return non_intersecting_matches

    return find_matches_with_linker_in_the_middle(non_intersecting_matches, mol)


def prepare_fragments_and_linker(frag_smi, linker_smi, mol):
    """
    Fixed for one fragment!!!!!!!!!

    Given a molecule and SMILES string of fragments from DeLinker data,
    creates fragment and linker conformers according to the molecule conformer,
    removes exit atoms and sets _Anchor property to all fragment atoms
    """

    frag = Chem.MolFromSmiles(frag_smi)
    linker = Chem.MolFromSmiles(linker_smi)
    match2conf_frag1, match2conf_linker = transfer_conformers(frag, linker, mol)

    correct_matches = find_correct_matches(
        match2conf_frag1.keys(),
        match2conf_linker.keys(),
        mol,
    )

    if len(correct_matches) > 2:
        raise Exception('Found more than two fragment matches')
    conf_frag1 = match2conf_frag1[correct_matches[0][0]]
    conf_linker = match2conf_linker[correct_matches[0][1]]    
    frag.AddConformer(conf_frag1)
    linker.AddConformer(conf_linker)

    return frag, linker

def prepare_fragments_with_empty_linker(frag_smi, mol):
    """
    Fixed for one fragment!!!!!!!!!

    Given a molecule and SMILES string of fragments from DeLinker data,
    creates fragment and linker conformers according to the molecule conformer,
    removes exit atoms and sets _Anchor property to all fragment atoms
    """

    frag = Chem.MolFromSmiles(frag_smi)

    match2conf_frag = transfer_conformers_empty_lin(frag, mol)

    correct_matches = find_correct_matches1(
        match2conf_frag.keys(),
        mol,
    )
    
    if len(correct_matches) > 2:
        raise Exception('Found more than two fragment matches')
    conf_frag1 = match2conf_frag[correct_matches[0][0]]
    frag.AddConformer(conf_frag1)

    return frag

def process_sdf(sdf_path, table):
    supplier = Chem.SDMolSupplier(sdf_path)
    out_table = []
    uuid = 0
    index_csv = 0
    index_sdf = 0
    while index_csv<len(table):
        data = table.loc[index_csv]
        mol_smi_csv = data['molecule']
        frag_smi_csv = data['fragments']
        link_smi_csv = data['linker']
        product_smi_csv = data["product"]
        center1 = data['center1']
        center2 = data['center2']
        anchors = data['anchors']
        linksize = data['linksize']
        while index_sdf<len(supplier) and supplier[index_sdf].GetProp('_Name')==mol_smi_csv:
            out_table.append({
                'uuid': uuid,
                'molecule': mol_smi_csv,
                'fragments': frag_smi_csv,
                'linker':  link_smi_csv,
                'product': product_smi_csv,
                'center1': center1,
                'center2': center2,
                'energy': supplier[index_sdf].GetProp('_Energy'),
                'anchors': anchors,
                'linksize': linksize,
                })
            index_sdf += 1
            uuid += 1
        index_csv += 1
        print('ok')       
    return  pd.DataFrame(out_table)


def processmin__sdf(sdf_path, table):
    supplier = Chem.SDMolSupplier(sdf_path)
    out_table = []
    molecules = []
    index_start = 0
    while index_start<len(table):
        data = table.loc[index_start]
        mol_smi_csv = data['molecule']
        frag_smi_csv = data['fragments']
        link_smi_csv = data['linker']
        product_smi_csv = data['product']
        center1 = data['center1']
        center2 = data['center2']
        anchors = data['anchors']
        linksize = data['linksize']
        index_finish = index_start
        dict_indices = {}
        
        while index_finish<len(table) and supplier[index_finish].GetProp('_Name')==mol_smi_csv:
            key = supplier[index_finish].GetProp('_Energy')
            dict_indices[key] = index_finish
            index_finish += 1

        min_energy = min(dict_indices)
        index = dict_indices[min_energy]
        molecules.append(supplier[index])
        out_table.append({
                'uuid': table.loc[index, 'uuid'],
                'molecule': mol_smi_csv,
                'fragments': frag_smi_csv,
                'linker':  link_smi_csv,
                "product": product_smi_csv,
                'center1': center1,
                'center2': center2,
                'energy': min_energy,
                'anchors': anchors,
                'linksize': linksize,
                'anchors': anchors,
                'linksize': linksize,
                })
        index_start = index_finish 
        print('ok')       
    return  molecules, pd.DataFrame(out_table)  


def mol_link_frag_sdf(sdf_path, table):
    supplier = Chem.SDMolSupplier(sdf_path)
    table['linker'][pd.isna(table.linker)] = ""
    molecules = []
    fragments = []
    linkers = []
    for i in range(len(supplier)):
        mol = supplier[i]
        frags_smi = table.loc[i, "fragments"]
        linker_smi = table.loc[i, "linker"]
        if  linker_smi!="":
            frag, linker = prepare_fragments_and_linker(frags_smi, linker_smi, mol)
        else: 
            frag = mol
            linker = Chem.MolFromSmiles('')

        molecules.append(mol)
        fragments.append(frag)
        linkers.append(linker)   
    return  molecules, fragments, linkers

def run_substructures_sdf(table_path: str, sdf_path: str, out_mol_path:str, out_frag_path: str, out_link_path:str, out_table_path):
    table = pd.read_csv(table_path, sep=',', index_col=False)
    molecules, fragments, linkers = mol_link_frag_sdf(sdf_path, table)
    table.to_csv(out_table_path, index=False)

    writer = Chem.SDWriter(out_mol_path)
    for mol in molecules:
            writer.write(mol)
    writer.close()

    writer = Chem.SDWriter(out_frag_path)
    writer.SetKekulize(False)
    for frags in fragments:
            writer.write(frags)
    writer.close()

    writer = Chem.SDWriter(out_link_path)
    writer.SetKekulize(False)
    for linker in linkers:
            writer.write(linker)
    writer.close()


def run_processmin__sdf(table_path: str, sdf_path: str, out_mol_path: str, out_table_path: str):
    table = pd.read_csv(table_path, sep=',', index_col=False)
    molecules, out_table = processmin__sdf(sdf_path, table)
    out_table.to_csv(out_table_path, index=False)
    writer = Chem.SDWriter(out_mol_path)
    for mol in molecules:
            writer.write(mol)
    writer.close()

def run(table_path, sdf_path, out_mol_path, out_frag_path, out_link_path, out_table_path):
    table = pd.read_csv(table_path, sep=',', index_col = False)

    out_table = process_sdf(sdf_path, table) 
    out_table.to_csv(out_table_path, index=False)

    # writer = Chem.SDWriter(out_mol_path)
    # for mol in molecules:
    #         writer.write(mol)
    # writer.close()

    # writer = Chem.SDWriter(out_frag_path)
    # writer.SetKekulize(False)
    # for frags in fragments:
    #         writer.write(frags)
    # writer.close()

    # writer = Chem.SDWriter(out_link_path)
    # writer.SetKekulize(False)
    # for linker in linkers:
    #         writer.write(linker)
    # writer.close()

def run_get_anchors(table_path):
    table = pd.read_csv(table_path, index_col = False)
    table["anchors"] = 0
    for i in range(len(table)):
        data = table.loc[i]
        frag_smi = data["fragments"]
        center1 = int(data['center1'])
        center2 = int(data['center2'])
        
        frag = Chem.MolFromSmiles(frag_smi)
        for atom in frag.GetAtoms():
                if atom.GetAtomMapNum()==center1 or atom.GetAtomMapNum()==center2:
                    table["anchors"].loc[i] = atom.GetIdx()
                    break

    table.to_csv("./table1_predsize.csv", index=False)








