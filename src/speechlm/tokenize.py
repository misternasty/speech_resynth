import glob
from pathlib import Path

import torch
from textless.data.speech_encoder import SpeechEncoder
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

from .data import SpeechDataset
from .utils import shift_unit


def tokenize(config):
    Path(config.s2u.tokenizer_path).parent.mkdir(parents=True, exist_ok=True)

    files = glob.glob(config.dataset.unicode_train + "*")
    initial_alphabet = [chr(shift_unit(unit)) for unit in range(config.s2u.vocab_size)]
    trainer = BpeTrainer(vocab_size=config.model.vocab_size, initial_alphabet=initial_alphabet)
    tokenizer = Tokenizer(BPE())
    tokenizer.train(files=files, trainer=trainer)
    tokenizer.save(config.s2u.tokenizer_path)

    train_paths = glob.glob(config.dataset.unicode_train + "*")
    dev_paths = glob.glob(config.dataset.unicode_dev + "*")

    _tokenize(tokenizer, config.dataset.train_file, train_paths)
    _tokenize(tokenizer, config.dataset.dev_file, dev_paths)


def _tokenize(tokenizer, file, paths):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as f:
        for path in paths:
            with open(path) as g:
                for unicodes in g:
                    unicodes = unicodes.rstrip()
                    units = tokenizer.encode(unicodes).ids
                    units = " ".join(str(u) for u in units)

                    f.write(f"{units}\n")


def encode(config, spk_ids: str = "1-9"):
    wav_dir_train = Path(config.dataset.wav_dir_train)
    wav_dir_dev = Path(config.dataset.wav_dir_dev)

    train_paths = wav_dir_train.glob(f"*/[{spk_ids}]*/**/*" + config.dataset.ext_audio)
    dev_paths = wav_dir_dev.glob(f"dev-clean/[{spk_ids}]*/**/*" + config.dataset.ext_audio)

    train_set = SpeechDataset(train_paths)
    dev_set = SpeechDataset(dev_paths)

    train_loader = torch.utils.data.DataLoader(train_set, num_workers=config.s2u.num_workers)
    dev_loader = torch.utils.data.DataLoader(dev_set, num_workers=config.s2u.num_workers)

    encoder = SpeechEncoder.by_name(
        dense_model_name=config.s2u.dense_model_name,
        quantizer_model_name=config.s2u.quantizer_model_name,
        vocab_size=config.s2u.vocab_size,
        deduplicate=True,
        need_f0=False,
    ).cuda()

    _encode(encoder, config.dataset.unicode_train + f"{spk_ids}", train_loader)
    _encode(encoder, config.dataset.unicode_dev + f"{spk_ids}", dev_loader)


def _encode(encoder: SpeechEncoder, file, data_loader: torch.utils.data.DataLoader):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as f:
        for item in tqdm(data_loader):
            outputs = encoder(item["input_values"].cuda())
            units = "".join(chr(shift_unit(u)) for u in outputs["units"].tolist())

            f.write(f"{units}\n")
