#!/bin/bash

function handle_sigint() {
    echo "SIGINT received, killing all subprocesses..."
    pkill -P $$
}

trap 'handle_sigint' SIGINT


gnn_size_version=599
diffusion_version=2282
n_samples=300
num_parts=32
sample_path="./sample/uspto_final_test/sampled_size/uspto_size_gnnbest_epoch=${gnn_size_version}/latent_epoch=${diffusion_version}/"

python sample.py \
	--linker_size_model "./models/uspto_size_gnn/uspto_size_gnnbest_epoch=${gnn_size_version}.ckpt" \
	--n_samples ${n_samples} \
	--sample_seed 0 \
	--n_steps 100

python xyz_split.py --sample_path $sample_path --num_parts $num_parts

for idx in $(seq 1 $num_parts)
do
python vis_get_result.py \
	--used_path "${sample_path}part_${idx}" \
	--n_samples ${n_samples} &
done

wait

python merge_result.py --used_path ${sample_path} --n_samples ${n_samples}