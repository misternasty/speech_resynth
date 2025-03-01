# Textless Speech Resynthesis using Conditional Flow Matching and HuBERT units

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org)
[![model](https://img.shields.io/badge/%F0%9F%A4%97-Models-blue)](https://huggingface.co/ryota-komatsu/flow_matching_with_hifigan)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-blue)](https://huggingface.co/datasets/ryota-komatsu/libritts-r-mhubert-2000units)

## Setup

```shell
sudo apt install git-lfs  # for UTMOS

conda create -y -n py39 python=3.9.21 pip=24.0
conda activate py39
pip install -r requirements/requirements.txt

sh scripts/setup.sh  # download textlesslib and UTMOS

cd src/textlesslib
pip install -e .
cd -

```

## Usage: Sampling multi-speaker speech from self-supervised discrete units

```python
import torchaudio
from textless.data.speech_encoder import SpeechEncoder

from src.flow_matching.models import ConditionalFlowMatchingWithHifiGan

wav_path = "/path/to/wav"

encoder = SpeechEncoder.by_name(
    dense_model_name="mhubert-base-vp_mls_cv_8lang",
    quantizer_model_name="kmeans-expresso",
    vocab_size=2000,
    deduplicate=False,
    need_f0=False,
).cuda()

# download a pretrained model from hugging face hub
decoder = ConditionalFlowMatchingWithHifiGan.from_pretrained("ryota-komatsu/flow_matching_with_hifigan").cuda()

# load a waveform
waveform, sr = torchaudio.load(wav_path)
waveform = torchaudio.functional.resample(waveform, sr, 16000)

# encode a waveform into pseudo-phonetic units
units = encoder(waveform.cuda())["units"]
units = units.unsqueeze(0) + 1  # 0: pad

# resynthesis
audio_values = decoder(units)
```

## Demo

Jupyter notebook demo is found [here](demo.ipynb).

## Data Preparation

If you already have LibriTTS-R, you can use it by editing [a config file](configs/resynth/mhubert-expresso-2000.yaml#L6);
```yaml
dataset:
  wav_dir_orig: "/path/to/LibriTTS-R" # ${dataset.wav_dir_orig}/train-clean-100, train-clean-360, ...
```

otherwise you can download the new one under `dataset_root`.
```shell
dataset_root=data

sh scripts/download_libritts.sh ${dataset_root}
```

## Training

```shell
python main.py
```