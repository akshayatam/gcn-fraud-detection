from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

from fraud_detection.data.elliptic import EllipticDataset
from fraud_detection.models.gcn import GCN
from fraud_detection.training.trainer import Trainer


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/gcn.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="The timestamp step to visualize predictions",
    )
    parser.add_argument(
        "--weights_file",
        default=None,
        help="Path to PyTorch weights file. If not provided, uses config.train.save_dir + config.name",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # Dataset (wrapper object, not the PyG Data object)
    dataset = EllipticDataset(config.dataset)
    config.model.input_dim = dataset.pyg_dataset().num_node_features

    # Model
    model = GCN(config.model).to(config.train.device)

    # Weights path
    if args.weights_file is None:
        save_dir = getattr(config.train, "save_dir", "artifacts/weights")
        weights_file = f"{save_dir}/{config.name}.pt"
    else:
        weights_file = args.weights_file

    state = torch.load(weights_file, map_location=config.train.device)
    model.load_state_dict(state)
    model = model.float()
    # Trainer (we override its model to avoid retraining/re-init differences)
    trainer = Trainer(config)
    trainer.model = model

    # Visualization output path
    viz_dir = getattr(config.train, "viz_dir", "artifacts/visualizations")
    out_path = f"{viz_dir}/{config.name}/{args.step}.png"

    trainer.visualize(
        dataset,
        time_step=args.step,
        save_to=out_path,
    )


if __name__ == "__main__":
    main()
