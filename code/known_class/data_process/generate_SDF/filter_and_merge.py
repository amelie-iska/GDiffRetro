import argparse
import sys
sys.path.append('../../')

import os
import pandas as pd

from rdkit import Chem
from tqdm import tqdm


def read_sdf(sdf_path):
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
    for molecule in supplier:
        yield molecule

def run(input_dir, output_dir, template, n, mol_path, frag_path, link_path, table_path):

    os.makedirs(output_dir, exist_ok=True)
    out_table_path = os.path.join(output_dir, f'{template}_table.csv')
    out_mol_path = os.path.join(output_dir, f'{template}_mol.sdf')
    full_table = []
    full_molecules = []
    mol_path = os.path.join(input_dir, f'{template}_mol.sdf')
    frag_path = os.path.join(input_dir, f'{template}_frag.sdf')
    link_path = os.path.join(input_dir, f'{template}_link.sdf')
    table_path = os.path.join(input_dir, f'{template}_table.csv')
    table = pd.read_csv(table_path)
    table['idx'] = table.index
    mol = set(table['molecule'])

    grouped_table = (
            table
            .groupby(['molecule'])
            .min()
            .reset_index()
            .sort_values(by='idx')
        )
    
    idx_to_keep = set(grouped_table['idx'].unique())
    table['keep'] = table['idx'].isin(idx_to_keep)

    generator = tqdm(zip(table.iterrows(), read_sdf(mol_path)), total=len(table))

    for (_, row), molecule, in generator:
            if row['keep']:
                if molecule.GetProp('_Name') != row['molecule']:
                    print('Molecule _Name:', molecule.GetProp('_Name'), row['molecule'])
                    continue
                full_table.append(row)
                full_molecules.append(molecule)

    full_table = pd.DataFrame(full_table)
    full_table.to_csv(out_table_path, index=False)
    


    with Chem.SDWriter(open(out_mol_path, 'w')) as writer:
        i = 0
        for mol in tqdm(full_molecules):
            writer.write(mol)
            
    # with Chem.SDWriter(open(out_frag_path, 'w')) as writer:
    #     writer.SetKekulize(False)
    #     i=0
    #     for frags in tqdm(full_fragments):
    #         # i += 1
    #         # if i!=18667 and i!=28917: 
    #             writer.write(frags)
    # with Chem.SDWriter(open(out_link_path, 'w')) as writer:
    #     i=0
    #     writer.SetKekulize(False)
    #     for linker in tqdm(full_linkers):
    #         # i += 1
    #         # if i!=18667 and i!=28917: 
    #             writer.write(linker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', action='store', type=str, required=True)
    parser.add_argument('--out-dir', action='store', type=str, required=True)
    parser.add_argument('--template', action='store', type=str, required=True)
    parser.add_argument('--number-of-files', action='store', type=int, required=True)
    args = parser.parse_args()

    run(
        input_dir=args.in_dir,
        output_dir=args.out_dir,
        template=args.template,
        n=args.number_of_files,
    )
