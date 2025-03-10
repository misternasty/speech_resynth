from pathlib import Path

import jiwer
import pandas as pd
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .data import LibriSpeech, SpeechDataset
from .models import ConditionalFlowMatchingWithHifiGan
from .utils.misc import cer_transform, wer_transform
from .utils.textless import load_encoder


@torch.inference_mode()
def synthesize(config):
    dataset = SpeechDataset(config.dataset.src_dir, config.dataset.tgt_dir, config.dataset.ext_audio_src)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        config.flow_matching_with_hifigan.batch_size,
        collate_fn=SpeechDataset.collate_fn,
    )

    encoder = load_encoder(
        config.flow_matching.dense_model_name,
        config.flow_matching.quantizer_model_name,
        config.flow_matching.vocab_size,
    )

    decoder = ConditionalFlowMatchingWithHifiGan.from_pretrained(config.flow_matching_with_hifigan.name).cuda()

    for batch in tqdm(dataloader):
        input_ids = []
        for item in batch:
            units = encoder(item["input_values"].cuda())["units"]
            units = units + 1  # 0: pad
            input_ids.append(units)

        input_ids = pad_sequence(input_ids, batch_first=True)

        audio_values = decoder(input_ids, config.flow_matching.dt, config.flow_matching.truncation_value)

        for item, hyp_wav in zip(batch, audio_values):
            hyp_wav = hyp_wav.cpu()

            Path(item["tgt_path"]).parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(item["tgt_path"], hyp_wav, 16000)


@torch.inference_mode()
def synthesize_librispeech(config):
    dataset = LibriSpeech(config.dataset.src_dir, config.dataset.tgt_dir, config.dataset.ext_audio_src)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        config.flow_matching_with_hifigan.batch_size,
        collate_fn=LibriSpeech.collate_fn,
    )

    encoder = load_encoder(
        config.flow_matching.dense_model_name,
        config.flow_matching.quantizer_model_name,
        config.flow_matching.vocab_size,
    )

    decoder = ConditionalFlowMatchingWithHifiGan.from_pretrained(config.flow_matching_with_hifigan.name).cuda()

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.asr.name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(config.asr.name)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=asr,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
    )

    transcripts = []
    hyps = []
    refs = []

    for batch in tqdm(dataloader):
        input_ids = []
        for item in batch:
            units = encoder(item["input_values"].cuda())["units"]
            units = units + 1  # 0: pad
            input_ids.append(units)

        input_ids = pad_sequence(input_ids, batch_first=True)

        audio_values = decoder(input_ids, config.flow_matching.dt, config.flow_matching.truncation_value)

        hyp_wavs = []
        ref_wavs = []

        for item, hyp_wav in zip(batch, audio_values):
            hyp_wav = hyp_wav.cpu()
            ref_wav = item["input_values"]

            Path(item["tgt_path"]).parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(item["tgt_path"], hyp_wav, 16000)

            hyp_wavs.append(hyp_wav.squeeze(0).numpy())
            ref_wavs.append(ref_wav.squeeze(0).numpy())

        batch_hyps = pipe(hyp_wavs, generate_kwargs={"language": "english"}, return_timestamps=True)
        batch_refs = pipe(ref_wavs, generate_kwargs={"language": "english"}, return_timestamps=True)

        batch_hyps = [hyp["text"] for hyp in batch_hyps]
        batch_refs = [ref["text"] for ref in batch_refs]

        for item, hyp, ref in zip(batch, batch_hyps, batch_refs):
            transcripts.append(item["transcript"])
            hyps.append(hyp)
            refs.append(ref)

    wer_hyp = jiwer.wer(transcripts, hyps, wer_transform, wer_transform)
    cer_hyp = jiwer.cer(transcripts, hyps, cer_transform, cer_transform)

    wer_ref = jiwer.wer(transcripts, refs, wer_transform, wer_transform)
    cer_ref = jiwer.cer(transcripts, refs, cer_transform, cer_transform)

    Path(config.dataset.result_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [wer_hyp, cer_hyp, wer_ref, cer_ref],
        index=["WER (hyp)", "CER (hyp)", "WER (ref)", "CER (ref)"],
    ).to_csv(config.dataset.result_path)
