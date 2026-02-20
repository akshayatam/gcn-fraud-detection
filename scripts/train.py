from argparse import ArgumentParser

from omegaconf import OmegaConf

from fraud_detection.training.trainer import Trainer


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/gcn.yaml",
        help="Path to training config",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    trainer = Trainer(config)

    trainer.train()
    trainer.save(config.name)


if __name__ == "__main__":
    main()
