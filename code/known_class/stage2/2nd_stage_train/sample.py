import argparse
import os
import torch
import yaml

from tqdm import tqdm
from pdb import set_trace
from rdkit import Chem

from src import utils
from src.lightning import DDPM
from src.linker_size_lightning import SizeClassifier
from src.visualizer import save_xyz_file, visualize_chain
from src.datasets import collate, collate_with_fragment_edges

from pytorch_lightning import seed_everything


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/uspto_sample.yml')
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--samples', action='store', type=str)
parser.add_argument('--data', action='store', type=str, required=False, default=None)
parser.add_argument('--prefix', action='store', type=str)
parser.add_argument('--n_samples', action='store', type=int)
parser.add_argument('--n_steps', action='store', type=int, required=False, default=None)
parser.add_argument('--sample_seed', type=int, default=0)
parser.add_argument('--linker_size_model', type=str, default='./models/uspto_size_gnn/uspto_size_gnnbest_epoch=599.ckpt')
parser.add_argument('--device', action='store', type=str)
args = parser.parse_args()

if args.config:
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list) and key != 'normalize_factors':
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value
    args.config = args.config.name
else:
        config_dict = {}

seed_everything(args.sample_seed, workers=True)
experiment_name = args.checkpoint.split('/')[-1].replace('.ckpt', '')

print('checkpoint: ', args.checkpoint)
print('path_linker_size_pred: ', args.linker_size_model)

if args.linker_size_model is None:
    output_dir = os.path.join(args.samples, args.prefix, experiment_name)
else:
    linker_size_name = args.linker_size_model.split('/')[-1].replace('.ckpt', '')
    output_dir = os.path.join(args.samples, args.prefix, 'sampled_size', linker_size_name, experiment_name)

os.makedirs(output_dir, exist_ok=True)

def check_if_generated(_output_dir, _uuids, n_samples):
    generated = True
    starting_points = []
    for _uuid in _uuids:
        uuid_dir = os.path.join(_output_dir, _uuid)
        numbers = []
        for fname in os.listdir(uuid_dir):
            try:
                num = int(fname.split('_')[0])
                numbers.append(num)
            except:
                continue
        if len(numbers) == 0 or max(numbers) != n_samples - 1:
            generated = False
            if len(numbers) == 0:
                starting_points.append(0)
            else:
                starting_points.append(max(numbers) - 1)

    if len(starting_points) > 0:
        starting = min(starting_points)
    else:
        starting = None

    return generated, starting


collate_fn = collate
sample_fn = None
if args.linker_size_model is not None:
    size_nn = SizeClassifier.load_from_checkpoint(args.linker_size_model, map_location=args.device)
    size_nn = size_nn.eval().to(args.device)

    collate_fn = collate_with_fragment_edges

    def sample_fn(_data):
        output, _ = size_nn.forward(_data)
        probabilities = torch.softmax(output, dim=1)
        distribution = torch.distributions.Categorical(probs=probabilities)
        samples = distribution.sample()
        sizes = []
        for label in samples.detach().cpu().numpy():
            sizes.append(size_nn.linker_id2size[label])
        sizes = torch.tensor(sizes, device=samples.device, dtype=torch.long)
        return sizes

model = DDPM.load_from_checkpoint(args.checkpoint, map_location=args.device)
model.val_data_prefix = args.prefix

if args.data is not None:
    model.data_path = args.data

if args.n_steps is not None:
    model.edm.T = args.n_steps
print(args.n_steps)

model = model.eval().to(args.device)
model.torch_device = model.device
model.setup(stage='val')

dataloader = model.val_dataloader(collate_fn=collate_fn)
print(f'Dataloader contains {len(dataloader)} batches')

for batch_idx, data in enumerate(dataloader):
    if batch_idx <= 1:
        continue
    uuids = []
    true_names = []
    frag_names = []

    for uuid in data['uuid']:
        uuid = str(uuid)
        uuids.append(uuid)
        true_names.append(f'{uuid}/true')
        frag_names.append(f'{uuid}/frag')
        os.makedirs(os.path.join(output_dir, uuid), exist_ok=True)

    generated, starting_point = check_if_generated(output_dir, uuids, args.n_samples)
    if generated:
        print(f'Already generated batch={batch_idx}, max_uuid={max(uuids)}')
        continue
    if starting_point > 0:
        print(f'Generating {args.n_samples - starting_point} for batch={batch_idx}')

    h, x, node_mask, frag_mask = data['one_hot'], data['positions'], data['atom_mask'], data['fragment_mask']
    if model.inpainting:
        center_of_mass_mask = node_mask
    elif model.center_of_mass == 'fragments':
        center_of_mass_mask = data['fragment_mask']
    elif model.center_of_mass == 'anchors':
        center_of_mass_mask = data['anchors']
    else:
        raise NotImplementedError(model.center_of_mass)
    x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
    utils.assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask)

    save_xyz_file(output_dir, h, x, node_mask, true_names, is_geom=model.is_geom)
    save_xyz_file(output_dir, h, x, frag_mask, frag_names, is_geom=model.is_geom)

    for i in tqdm(range(starting_point, args.n_samples), desc=str(batch_idx)):
        chain, node_mask = model.sample_chain(data, sample_fn=sample_fn, keep_frames=1)
        x = chain[0][:, :, :model.n_dims]
        h = chain[0][:, :, model.n_dims:]
        pred_names = [f'{uuid}/{i}' for uuid in uuids]
        save_xyz_file(output_dir, h, x, node_mask, pred_names, is_geom=model.is_geom)

