from pathlib import Path

import torch
import torch.nn.functional as F
from textless.data.speech_encoder import SpeechEncoder
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import LlamaForCausalLM

from .data import SpeechDataset
from .utils import shift_unit


def evaluate(config):
    encoder = SpeechEncoder.by_name(
        dense_model_name=config.s2u.dense_model_name,
        quantizer_model_name=config.s2u.quantizer_model_name,
        vocab_size=config.s2u.vocab_size,
        deduplicate=True,
        need_f0=False,
    ).cuda()
    tokenizer = Tokenizer.from_file(config.s2u.tokenizer_path)
    model = LlamaForCausalLM.from_pretrained(config.model.path).cuda()

    swuggy_dir = Path(config.dataset.swuggy_dir).expanduser()
    sblimp_dir = Path(config.dataset.sblimp_dir).expanduser()

    swuggy_dev_paths = list(swuggy_dir.glob("dev/*.wav"))
    sblimp_dev_paths = list(sblimp_dir.glob("dev/*.wav"))
    swuggy_test_paths = list(swuggy_dir.glob("test/*.wav"))
    sblimp_test_paths = list(sblimp_dir.glob("test/*.wav"))

    swuggy_dev_set = SpeechDataset(swuggy_dev_paths)
    sblimp_dev_set = SpeechDataset(sblimp_dev_paths)
    swuggy_test_set = SpeechDataset(swuggy_test_paths)
    sblimp_test_set = SpeechDataset(sblimp_test_paths)

    swuggy_dev_loader = torch.utils.data.DataLoader(
        swuggy_dev_set, config.dataloader.batch_size, collate_fn=SpeechDataset.collate_fn
    )
    sblimp_dev_loader = torch.utils.data.DataLoader(
        sblimp_dev_set, config.dataloader.batch_size, collate_fn=SpeechDataset.collate_fn
    )
    swuggy_test_loader = torch.utils.data.DataLoader(
        swuggy_test_set, config.dataloader.batch_size, collate_fn=SpeechDataset.collate_fn
    )
    sblimp_test_loader = torch.utils.data.DataLoader(
        sblimp_test_set, config.dataloader.batch_size, collate_fn=SpeechDataset.collate_fn
    )

    _eval(encoder, tokenizer, model, config.dataset.swuggy_dev, swuggy_dev_loader)
    _eval(encoder, tokenizer, model, config.dataset.sblimp_dev, sblimp_dev_loader)
    _eval(encoder, tokenizer, model, config.dataset.swuggy_test, swuggy_test_loader)
    _eval(encoder, tokenizer, model, config.dataset.sblimp_test, sblimp_test_loader)


@torch.inference_mode()
def _eval(
    encoder: SpeechEncoder,
    tokenizer: Tokenizer,
    model: LlamaForCausalLM,
    file,
    data_loader: torch.utils.data.DataLoader,
):
    with open(file, "w") as f:
        for batch in tqdm(data_loader):
            # Speech encoder
            batch_unicodes = []
            for item in batch:
                units = encoder(item["input_values"].cuda())["units"].tolist()
                unicodes = "".join(chr(shift_unit(u)) for u in units)
                batch_unicodes.append(unicodes)

            # BPE
            encoded = tokenizer.encode_batch(batch_unicodes)

            # Pad
            max_len = max(len(item.ids) for item in encoded)
            input_ids = [
                F.pad(torch.tensor(item.ids, device="cuda") + 1, (0, max_len - len(item.ids))) for item in encoded
            ]
            input_ids = torch.stack(input_ids)
            labels = input_ids.masked_fill(input_ids.eq(0), -100)

            # Speech LM
            logits = model(input_ids=input_ids, labels=labels).logits.transpose(1, 2)

            labels = F.pad(labels, (0, 1), value=-100)
            shifted_labels = labels[:, 1:]

            scores = -F.cross_entropy(logits, shifted_labels, reduction="none")
            scores = scores.sum(dim=1) / scores.ne(0).sum(dim=1)
            scores = scores.tolist()

            for item, score in zip(batch, scores):
                f.write(f"{item['name']} {score}\n")
