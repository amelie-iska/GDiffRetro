import argparse
import os.path

import pandas as pd
import torch
import torchdrug
from torchdrug import data, datasets, utils
from torchdrug import core, models, tasks
from rdkit import Chem
from rdkit.Chem import AllChem
import csv
from torchdrug import models
from tqdm import tqdm
import pickle
torch.manual_seed(1)

def main(args):
    uspto_test_csv_path = args.uspto_test_csv_path
    uspto_preprocess_csv_path = args.uspto_preprocess_csv_path
    checkpoint = args.checkpoint
    checkpoint_name = os.path.basename(checkpoint)
    result_dir = './result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, f"{checkpoint_name[:-4]}_{args.sample_times}.pkl")

    reaction_preprocess_dataset = datasets.USPTO50k(uspto_preprocess_csv_path,
                                               atom_feature="center_identification",
                                               kekulize=True)

    reaction_dataset = datasets.USPTO50k(uspto_test_csv_path,
                                         atom_feature="center_identification",
                                         kekulize=True)
    reaction_model = models.RGCN(input_dim=reaction_dataset.node_feature_dim,
                                 hidden_dims=[512, 512, 512],
                                 batch_norm=False, short_cut=True,
                                 num_relation=reaction_dataset.num_bond_type,
                                 concat_hidden=False)
    dual_model = models.RGCN(input_dim=reaction_dataset.node_feature_dim,
                             hidden_dims=[512, 512, 512],
                             batch_norm=False, short_cut=True,
                             num_relation=reaction_dataset.num_bond_type,
                             concat_hidden=False)
    feature = ("graph", "atom", "bond", "reaction") if args.with_reaction else ("graph", "atom", "bond")
    reaction_task = tasks.CenterIdentification(reaction_model, dual_model, feature=feature)
    reaction_optimizer = torch.optim.Adam(reaction_task.parameters(), lr=1e-4, weight_decay=0)

    reaction_solver = core.Engine(reaction_task, reaction_preprocess_dataset, reaction_dataset,
                                  reaction_dataset, reaction_optimizer,
                                  gpus=[0],
                                  batch_size=128, gradient_interval=1, log_interval=100)

    reaction_solver.load(checkpoint, load_optimizer=False)
    reaction_solver.evaluate("test")

    product_right_count_dict = dict()
    batch_size = args.batch_size

    dataloader = data.DataLoader(reaction_dataset, batch_size)
    for batch in tqdm(dataloader):
        batch = utils.cuda(batch, device=reaction_task.device)
        reactant, product = batch["graph"]
        pred_reaction_center_k = reaction_solver.model.sample_center_reaction(batch, k=args.sample_times)
        product_reaction_center = (product.reaction_center).unsqueeze(1)
        batch_product_right_count = (pred_reaction_center_k == product_reaction_center).all(dim=-1).sum(dim=-1)
        batch_product_smiles = product.to_smiles()
        for i in range(batch_product_right_count.size(0)):
            product_right_count_dict[batch_product_smiles[i]] = batch_product_right_count[i].item()

    with open(result_path, 'wb') as fp:
        pickle.dump(product_right_count_dict, fp)
        fp.close()
        print("Stage 1 result saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Result Statistics for Stage 1.')
    parser.add_argument('--checkpoint', type=str, default="model/reaction_center_model_wo_class.pth",
                        help='Path to ReactionCenter model')
    parser.add_argument('--uspto_test_csv_path', type=str, default="dataset/selected_test.csv",
                        help='Path to test dataset')
    parser.add_argument('--uspto_preprocess_csv_path', type=str, default="dataset/preprocess_dataset.csv",
                        help='Path to preprocess dataset')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--with_reaction', action="store_true")
    parser.add_argument('--sample_times', type=int, default=300)

    args = parser.parse_args()
    main(args)