import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
import pprint
import tensorflow as tf

from model import EncryptNet
from tools import summarize_results


@hydra.main(config_path="config", config_name="run_configs")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)

    print("Running config:")
    pprint.pprint(cfg, width=1)

    print("Running model")
    results = EncryptNet(cfg["net_params"], **cfg["other_params"]).train()
    summarize_results(results)
    pickle.dump(results, open("results.p", "wb"))


if __name__ == "__main__":
    main()
