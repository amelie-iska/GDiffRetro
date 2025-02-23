import sys
from datetime import datetime
import pandas as pd
import torch
import numpy as np

class Logger(object):
    def __init__(self, logpath, syspart=sys.stdout):
        self.terminal = syspart
        self.log = open(logpath, "a")

    def write(self, message):

        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def log(*args):
    print(f'[{datetime.now()}]', *args)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask):
    """
    Subtract center of mass of fragments from coordinates of all atoms
    """
    x_masked = x * center_of_mass_mask
    N = center_of_mass_mask.sum(1, keepdims=True)
    mean = torch.sum(x_masked, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def tab_process(count_dict, current_path, n_samples):

    result_dir = current_path
    merge_result_path = os.path.join(result_dir, 'merged_result.csv')
    df = pd.read_csv(merge_result_path, sep='\t', header=None)
    
    uuid_df = df.iloc[:, :1]
    sel_df = df.iloc[:, 1: n_samples+1]
    target_df = df.iloc[:, n_samples+1: n_samples+2]
    product_df = df.iloc[:, n_samples+2: n_samples+3]

    miss = 0
    for i, row in sel_df.iterrows():
        product_i = product_df.iloc[i, 0]
        try:
            stage1_right_num = count_dict[product_i]

            if stage1_right_num < n_samples:
                row[stage1_right_num:] = "MISSING"
                sel_df.loc[i] = row
        except:
            miss += 1
            continue

    if miss > 0:
        raise Exception(f"Wrong. Miss {miss} products.")

    top10_df = sel_df.apply(process_row, axis=1, result_type='expand')
    bool_df = pd.DataFrame(index=top10_df.index, columns=top10_df.columns)
    target_df = target_df.iloc[:, 0]

    for col in top10_df.columns:
        bool_df[col] = (top10_df[col] == target_df)

    all_num = bool_df.shape[0]
    
    for k in range(10):
        right_num = bool_df.iloc[:, :k+1].any(axis=1).sum()
        print(f"Top-{k+1}: {right_num / all_num}")

def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    x_masked = x * center_of_mass_mask
    largest_value = x_masked.abs().max().item()
    error = torch.sum(x_masked, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Partial mean is not zero, relative_error {rel_error}'


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def center_gravity_zero_gaussian_log_likelihood(x):
    assert len(x.size()) == 3
    B, N, D = x.size()
    assert_mean_zero(x)

    r2 = sum_except_batch(x.pow(2))

    degrees_of_freedom = (N-1) * D

    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)
    x_projected = remove_mean(x)
    return x_projected

def process_row(row):
    counts = Counter(row)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    categories, freqs = zip(*sorted_counts)
    result = list(categories) + [""] * (10 - len(categories))
    return result

def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)

    r2 = sum_except_batch(x.pow(2))

    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def standard_gaussian_log_likelihood(x):
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2*np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked


def concatenate_features(x, h):
    xh = torch.cat([x, h['categorical']], dim=2)
    if 'integer' in h:
        xh = torch.cat([xh, h['integer']], dim=2)
    return xh


def split_features(z, n_dims, num_classes, include_charges):
    assert z.size(2) == n_dims + num_classes + include_charges
    x = z[:, :, 0:n_dims]
    h = {'categorical': z[:, :, n_dims:n_dims+num_classes]}
    if include_charges:
        h['integer'] = z[:, :, n_dims+num_classes:n_dims+num_classes+1]

    return x, h


class Queue:
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} while allowed {max_grad_norm:.1f}')
    return grad_norm

def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')

def merge_res(current_path, n_samples, result_path, save_merged_result_path, uspto_final_test_table):
    dict_path = "../../../../code/known_class/stage1/result/reaction_center_model_w_class_300.pkl"

    def is_sorted(lst):
        return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

    if not is_sorted(uspto_final_test_table['uuid'].to_list()):
        raise ValueError("The uuid in the uspto_final_test_table.csv should be sorted!")

    grouped_by_product = uspto_final_test_table[['uuid', 'product']].groupby('product')['uuid'].apply(list).reset_index()
    product_uuidlist = dict(zip(grouped_by_product['product'], grouped_by_product['uuid']))

    result_table = pd.read_csv(result_path).sort_values(by='uuid')
    result_uuid_list = result_table['uuid'].to_list()
    agg_dict = {}
    for value in result_uuid_list:
        for key, values_list in product_uuidlist.items():
            if value in values_list:
                if key in agg_dict:
                    agg_dict[key].append(value)
                else:
                    agg_dict[key] = [value]
    result_table.set_index('uuid', inplace=True)
    result_df = pd.DataFrame(index=range(len(agg_dict.keys())))
    
    for i, (product_smiles, uuids) in enumerate(agg_dict.items()):
        result_df.loc[i, 'uuid'] = '.'.join(map(str, uuids))
        for k in range(1, n_samples + 2):
            result_df.loc[i, str(k)] = '.'.join(result_table.loc[uuids, str(k)])
        result_df.loc[i, "product"] = product_smiles
    
        result_df.reset_index(drop=True, inplace=True)
        result_df.to_csv(save_merged_result_path, sep='\t', index=False, header=False)

    with open(dict_path, 'rb') as fp:
        count_dict = pickle.load(fp)
        fp.close()
    
    tab_process(count_dict, current_path, n_samples)

class FoundNaNException(Exception):
    def __init__(self, x, h):
        x_nan_idx = self.find_nan_idx(x)
        h_nan_idx = self.find_nan_idx(h)

        self.x_h_nan_idx = x_nan_idx & h_nan_idx
        self.only_x_nan_idx = x_nan_idx.difference(h_nan_idx)
        self.only_h_nan_idx = h_nan_idx.difference(x_nan_idx)

    @staticmethod
    def find_nan_idx(z):
        idx = set()
        for i in range(z.shape[0]):
            if torch.any(torch.isnan(z[i])):
                idx.add(i)
        return idx


def get_batch_idx_for_animation(batch_size, batch_idx):
    batch_indices = []
    mol_indices = []
    for idx in [0, 110, 360]:
        if idx // batch_size == batch_idx:
            batch_indices.append(idx % batch_size)
            mol_indices.append(idx)
    return batch_indices, mol_indices


# Rotation data augmntation
def random_rotation(x):
    bs, n_nodes, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2
    if n_dims == 2:
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_row0 = torch.cat([cos_theta, -sin_theta], dim=2)
        R_row1 = torch.cat([sin_theta, cos_theta], dim=2)
        R = torch.cat([R_row0, R_row1], dim=1)

        x = x.transpose(1, 2)
        x = torch.matmul(R, x)
        x = x.transpose(1, 2)

    elif n_dims == 3:

        # Build Rx
        Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1:2, 1:2] = cos
        Rx[:, 1:2, 2:3] = sin
        Rx[:, 2:3, 1:2] = - sin
        Rx[:, 2:3, 2:3] = cos

        # Build Ry
        Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0:1, 0:1] = cos
        Ry[:, 0:1, 2:3] = -sin
        Ry[:, 2:3, 0:1] = sin
        Ry[:, 2:3, 2:3] = cos

        # Build Rz
        Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0:1, 0:1] = cos
        Rz[:, 0:1, 1:2] = sin
        Rz[:, 1:2, 0:1] = -sin
        Rz[:, 1:2, 1:2] = cos

        x = x.transpose(1, 2)
        x = torch.matmul(Rx, x)
        x = torch.matmul(Ry, x)
        x = torch.matmul(Rz, x)
        x = x.transpose(1, 2)
    else:
        raise Exception("Not implemented Error")

    return x.contiguous()