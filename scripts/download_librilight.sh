#!/bin/sh

dataset_root=${1:-data}

wget -t 0 -c -P ${dataset_root}/librilight https://dl.fbaipublicfiles.com/librilight/data/small.tar
wget -t 0 -c -P ${dataset_root}/librilight https://dl.fbaipublicfiles.com/librilight/data/medium.tar
wget -t 0 -c -P ${dataset_root}/librilight https://dl.fbaipublicfiles.com/librilight/data/large.tar

mkdir ${dataset_root}/_librilight
tar xf ${dataset_root}/librilight/small.tar -C ${dataset_root}/_librilight
tar xf ${dataset_root}/librilight/medium.tar -C ${dataset_root}/_librilight
tar xf ${dataset_root}/librilight/large.tar -C ${dataset_root}/_librilight

cd src/libri-light/data_preparation
python cut_by_vad.py --target_len_sec 25 --input_dir ../../../${dataset_root}/_librilight/small --output_dir ../../../${dataset_root}/librilight/small
python cut_by_vad.py --target_len_sec 25 --input_dir ../../../${dataset_root}/_librilight/medium --output_dir ../../../${dataset_root}/librilight/medium
python cut_by_vad.py --target_len_sec 25 --input_dir ../../../${dataset_root}/_librilight/large --output_dir ../../../${dataset_root}/librilight/large

rm -r ../../../${dataset_root}/_librilight