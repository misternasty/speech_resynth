import random
from typing import Any, Dict

import torch
import torchaudio


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, wav_paths):
        self.wav_paths = list(wav_paths)

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, n: int) -> Dict[str, Any]:
        wav_path = self.wav_paths[n]
        name = wav_path.stem
        wav_path = str(wav_path)
        input_values, sr = torchaudio.load(wav_path)
        return {"input_values": input_values, "name": name}

    @staticmethod
    def collate_fn(batch):
        return batch


class UnitDataset(torch.utils.data.Dataset):
    def __init__(self, path, units_per_sample: int = 704, num_special_tokens: int = 1):
        self.input_ids = []
        with open(path) as f:
            for units in f:
                units = units.rstrip().split()
                units = torch.tensor([int(u) + num_special_tokens for u in units])
                self.input_ids.append(units)

        self.units_per_sample = units_per_sample

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, n: int) -> Dict[str, torch.Tensor]:
        input_ids = self.input_ids[n]
        attention_mask = torch.ones_like(input_ids)

        diff = len(input_ids) - self.units_per_sample

        if diff > 0:
            start = random.randrange(diff)
            input_ids = input_ids[start : start + self.units_per_sample]
            attention_mask = attention_mask[start : start + self.units_per_sample]
        else:
            input_ids = torch.nn.functional.pad(input_ids, (0, -diff))
            attention_mask = torch.nn.functional.pad(attention_mask, (0, -diff))

        labels = input_ids.masked_fill(input_ids.eq(0), -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
