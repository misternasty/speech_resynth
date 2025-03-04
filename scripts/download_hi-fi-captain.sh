#!/bin/sh

dataset_root=${1:-data}

wget -t 0 -c -P ${dataset_root} https://ast-astrec.nict.go.jp/release/hi-fi-captain/hfc_en-US_F.zip
wget -t 0 -c -P ${dataset_root} https://ast-astrec.nict.go.jp/release/hi-fi-captain/hfc_en-US_M.zip

cd ${dataset_root}
unzip hfc_en-US_F.zip
unzip hfc_en-US_M.zip