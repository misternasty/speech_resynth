import argparse

from omegaconf import OmegaConf

from src.flow_matching.preprocess import preprocess
from src.flow_matching.train import train_flow_matching
from src.hifigan.train import train_hifigan

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resynth/mhubert-expresso-2000.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    preprocess(config)
    # train_hifigan(config)  # use a pretrained model
    train_flow_matching(config)
