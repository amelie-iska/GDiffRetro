import pandas as pd

# mode = eval, train, test
mode = 'eval'
use_unique = False

input_path = './final_data/merge_final_' + mode + '.csv'
unique_mol_path = './molecule_unique_' + mode + '.txt'

table = pd.read_csv(input_path, sep=',', names=['id', 'molecule', 'fragments', 'linker', 'product', 'center1', 'center2', 'anchors', 'linksize'])
if use_unique:
    smiles = table.molecule.unique()
else:
    smiles = table.molecule

with open(unique_mol_path, 'w') as f:
    for smi in smiles:
        if smi == 'molecule':
            continue
        f.write(f'{smi}\n')

cores = 1
output_template = './sdfdir/conformers_' + mode
import rdkit_conf_parallel
import os
from rdkit import Chem
from tqdm import tqdm
rdkit_conf_parallel.main(input_file=unique_mol_path, output_template=output_template, cores=cores)

if cores != 1:
    merged_out_mol_path = './sdfdir/merged_mol_' + mode + '.sdf'
    def read_sdf(sdf_path):
        with Chem.SDMolSupplier(sdf_path, sanitize=False) as supplier:
            for molecule in supplier:
                yield molecule

    full_molecules = []
    for idx in range(cores):
        mol_path = output_template + '_' + str(idx) + '.sdf'
        for mol_read in read_sdf(mol_path):
            full_molecules.append(mol_read)

    with Chem.SDWriter(open(merged_out_mol_path, 'w')) as writer:
        for mol in tqdm(full_molecules):
            writer.write(mol)
else:
    merged_out_mol_path = './sdfdir/conformers_' + mode + '_0.sdf'

import prepare_dataset
prepare_dataset.run(table_path = input_path, sdf_path = merged_out_mol_path, out_mol_path = './final/mol_' + mode + '.sdf', out_frag_path = './final/frag_' + mode + '.sdf', \
    out_link_path = './final/link_' + mode + '.sdf', out_table_path = './final/test_final_' + mode + '.csv')

prepare_dataset.run_processmin__sdf(table_path = './final/test_final_' + mode + '.csv', sdf_path = merged_out_mol_path, out_mol_path = './final/mol_' + mode + '.sdf', \
    out_table_path = './final/test_final_2_' + mode + '.csv')

repare_dataset.run_substructures_sdf(table_path = './final/test_final_2_' + mode + '.csv', sdf_path = './final/mol_' + mode + '.sdf', out_mol_path = './final/mol_' + mode + '.sdf', out_frag_path = './final/frag_' + mode + '.sdf', \
    out_link_path = './final/link_' + mode + '.sdf', out_table_path = './final/final_' + mode + '.csv')
