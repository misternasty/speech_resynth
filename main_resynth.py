import fire
from omegaconf import OmegaConf

from src.flow_matching.preprocess import extract_features, preprocess, resample, tokenize
from src.flow_matching.synthesize import synthesize, synthesize_librispeech
from src.flow_matching.train import train_flow_matching
from src.hifigan.train import train_hifigan


class TaskRunner:
    def resample(self, config: str = "configs/resynth/mhubert-expresso-2000.yaml"):
        config = OmegaConf.load(config)
        resample(config)

    def tokenize(self, config: str = "configs/resynth/mhubert-expresso-2000.yaml"):
        config = OmegaConf.load(config)
        tokenize(config)

    def extract_features(self, config: str = "configs/resynth/mhubert-expresso-2000.yaml"):
        config = OmegaConf.load(config)
        extract_features(config)

    def train_hifigan(self, config: str = "configs/resynth/mhubert-expresso-2000.yaml"):
        config = OmegaConf.load(config)
        train_hifigan(config)

    def train_flow_matching(self, config: str = "configs/resynth/mhubert-expresso-2000.yaml"):
        config = OmegaConf.load(config)
        train_flow_matching(config)

    def synthesize(self, config: str = "configs/resynth/mhubert-expresso-2000.yaml"):
        config = OmegaConf.load(config)
        synthesize(config)

    def synthesize_librispeech(self, config: str = "configs/resynth/mhubert-expresso-2000.yaml"):
        config = OmegaConf.load(config)
        synthesize_librispeech(config)

    def __call__(self, config: str = "configs/resynth/mhubert-expresso-2000.yaml"):
        config = OmegaConf.load(config)
        preprocess(config)
        train_hifigan(config)
        train_flow_matching(config)


if __name__ == "__main__":
    fire.Fire(TaskRunner)
