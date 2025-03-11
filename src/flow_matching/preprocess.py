import json
from pathlib import Path
from typing import List, Optional

import librosa
import torch
import torchaudio
from tqdm import tqdm

from ..hifigan.data import mel_spectrogram
from .utils.textless import load_encoder


def preprocess(config):
    resample(config)
    tokenize(config)
    extract_features(config)


def resample(config):
    wav_dir_orig = Path(config.dataset.wav_dir_orig)
    wav_dir = Path(config.dataset.wav_dir)
    wav_paths = list(wav_dir_orig.glob("**/*" + config.dataset.ext_audio))

    for wav_path in tqdm(wav_paths):
        wav_name = wav_path.relative_to(wav_dir_orig)
        wav_path = str(wav_path)

        wav, sr = torchaudio.load(wav_path)
        wav = torchaudio.functional.resample(wav, sr, 16000)

        if config.dataset.vad:
            wav = wav.numpy()
            wav, _ = librosa.effects.trim(wav, top_db=20)
            wav = torch.from_numpy(wav)

        wav_path = wav_dir / wav_name
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path = str(wav_path)  # for sox backend
        torchaudio.save(wav_path, wav, 16000)


def tokenize(config):
    wav_dir = Path(config.dataset.wav_dir)
    txt_dir = Path(config.dataset.wav_dir_orig)
    train_paths = sorted(wav_dir.glob("train-*/**/*" + config.dataset.ext_audio))
    dev_paths = sorted(wav_dir.glob("dev-clean/**/*" + config.dataset.ext_audio))

    encoder = load_encoder(
        config.flow_matching.dense_model_name,
        config.flow_matching.quantizer_model_name,
        config.flow_matching.vocab_size,
        config.flow_matching.predict_duration,
    )

    _tokenize(encoder, config.dataset.train_file, train_paths, wav_dir)
    _tokenize(encoder, config.dataset.dev_file, dev_paths, wav_dir, txt_dir)


def _tokenize(encoder, file, wav_paths: List[str], wav_dir: Path, txt_dir: Optional[Path] = None):
    dataset = dict()

    for wav_path in tqdm(wav_paths):
        wav_name = wav_path.relative_to(wav_dir)
        wav_name = wav_name.with_suffix("")
        wav_path = str(wav_path)

        waveform, sr = torchaudio.load(wav_path)
        outputs = encoder(waveform.cuda())
        units = outputs["units"].tolist()
        durations = outputs["durations"].tolist()

        if txt_dir is not None:
            txt_path = txt_dir / wav_name
            txt_path = txt_path.with_suffix(".normalized.txt")

            with open(txt_path) as g:
                txt = g.read().rstrip()

            dataset[wav_name] = {"units": units, "durations": durations, "transcript": txt}
        else:
            dataset[wav_name] = {"units": units, "durations": durations, "transcript": ""}

    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as f:
        json.dump(dataset, f)


def extract_features(config):
    wav_dir = Path(config.dataset.wav_dir)
    spectrogram_dir = Path(config.dataset.spectrogram_dir)
    wav_paths = list(wav_dir.glob("**/*" + config.dataset.ext_audio))

    for wav_path in tqdm(wav_paths):
        wav_name = wav_path.relative_to(wav_dir).with_suffix("")
        spectrogram_path = spectrogram_dir / wav_name.with_suffix(".pt")
        if spectrogram_path.is_file():
            continue
        spectrogram_path.parent.mkdir(parents=True, exist_ok=True)

        wav_path = str(wav_path)
        wav, sr = torchaudio.load(wav_path)
        wav = wav.cuda()
        wav = wav / wav.abs().max() * 0.95

        spectrogram_labels = mel_spectrogram(wav)  # (1, 80, len)
        spectrogram_labels = spectrogram_labels.transpose(1, 2)  # (1, len, 80)
        spectrogram_labels = spectrogram_labels.cpu()

        torch.save(spectrogram_labels, spectrogram_path)
