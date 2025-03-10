import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchaudio
from tqdm import tqdm


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, src_dir, tgt_dir, ext_audio: str = ".flac"):
        self.src_dir = Path(src_dir)
        self.tgt_dir = Path(tgt_dir)

        self.src_paths = list(self.src_dir.glob("**/*" + ext_audio))
        self.tgt_paths = list()

        for src_path in self.src_paths:
            src_name = src_path.relative_to(src_dir)
            tgt_path = self.tgt_dir / src_name
            self.tgt_paths.append(tgt_path)

    def __len__(self) -> int:
        return len(self.src_paths)

    def __getitem__(self, n: int) -> Dict[str, Any]:
        src_path = self.src_paths[n]
        tgt_path = self.tgt_paths[n]

        src_path = str(src_path)
        tgt_path = str(tgt_path)

        input_values, sr = torchaudio.load(src_path)
        input_values = torchaudio.functional.resample(input_values, sr, 16000)

        return {"input_values": input_values, "tgt_path": tgt_path}

    @staticmethod
    def collate_fn(batch):
        return batch


class LibriSpeech(SpeechDataset):
    def __getitem__(self, n: int) -> Dict[str, Any]:
        item = super().__getitem__(n)

        # transcript
        src_name = Path(item["tgt_path"]).relative_to(self.tgt_dir)
        src_name = src_name.with_suffix("")
        src_name = str(src_name)

        split, speaker_id, chap_id, utterance_id = src_name.split("/")
        file = self.src_dir / split / speaker_id / chap_id / f"{speaker_id}-{chap_id}.trans.txt"

        with open(file) as f:
            for line in f:
                id, transcript = line.rstrip().split(" ", maxsplit=1)
                if id == utterance_id:
                    break

        item["transcript"] = transcript

        return item


class UnitDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file,
        wav_dir=None,
        spectrogram_dir=None,
        frames_per_seg: Optional[int] = None,
        ext_audio: str = ".wav",
    ):
        self.input_ids = []
        self.spectrogram_labels = []
        self.transcripts = []
        self.names = []
        self.input_values = []

        with open(file) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                name, units, transcript = line.split("\t", maxsplit=2)

                input_ids = torch.tensor([int(u) + 1 for u in units.split()])  # 0: pad

                if spectrogram_dir is not None:
                    spectrogram_path = os.path.join(spectrogram_dir, name + ".pt")
                    spectrogram_labels = torch.load(spectrogram_path, "cpu", weights_only=True)
                    spectrogram_labels = spectrogram_labels.squeeze(0)  # (len, 80)
                else:
                    spectrogram_labels = torch.zeros(1, 80)

                if wav_dir is not None:
                    wav_path = os.path.join(wav_dir, name + ext_audio)
                    input_values, sr = torchaudio.load(wav_path)
                    input_values = input_values.squeeze(0)  # (len,)
                else:
                    input_values = torch.zeros(1)

                self.input_ids.append(input_ids)
                self.spectrogram_labels.append(spectrogram_labels)
                self.transcripts.append(transcript)
                self.names.append(name)
                self.input_values.append(input_values)

        self.frames_per_seg = frames_per_seg

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, n: int) -> Dict[str, Any]:
        input_ids = self.input_ids[n]
        spectrogram_labels = self.spectrogram_labels[n]
        transcripts = self.transcripts[n]
        names = self.names[n]
        input_values = self.input_values[n]

        if self.frames_per_seg is not None:
            diff = len(input_ids) - self.frames_per_seg

            if diff > 0:
                start = random.randrange(diff)
                input_ids = input_ids[start : start + self.frames_per_seg]
                spectrogram_labels = spectrogram_labels[start : start + self.frames_per_seg]
            else:
                input_ids = torch.nn.functional.pad(input_ids, (0, -diff))
                spectrogram_labels = torch.nn.functional.pad(spectrogram_labels, (0, 0, 0, -diff), value=-100)

        return {
            "input_ids": input_ids,
            "spectrogram_labels": spectrogram_labels,
            "transcripts": transcripts,
            "names": names,
            "input_values": input_values,
        }
