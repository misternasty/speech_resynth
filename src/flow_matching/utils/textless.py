import torch
from textless import dispatch_quantizer
from textless.data.speech_encoder import SpeechEncoder


def load_encoder(
    dense_model_name: str = "mhubert-base-vp_mls_cv_8lang",
    quantizer_model_name: str = "kmeans-expresso",
    vocab_size: int = 2000,
) -> SpeechEncoder:
    return SpeechEncoder.by_name(
        dense_model_name=dense_model_name,
        quantizer_model_name=quantizer_model_name,
        vocab_size=vocab_size,
        deduplicate=False,
        need_f0=False,
    ).cuda()


def embedding(
    dense_model_name: str = "mhubert-base-vp_mls_cv_8lang",
    quantizer_model_name: str = "kmeans-expresso",
    vocab_size: int = 2000,
):
    quantizer = dispatch_quantizer(dense_model_name, quantizer_model_name, vocab_size)
    embedding = torch.from_numpy(quantizer.kmeans_model.cluster_centers_)
    embedding = torch.cat((torch.zeros(1, embedding.shape[1]), embedding))
    return torch.nn.Embedding.from_pretrained(embedding, freeze=True, padding_idx=0)
