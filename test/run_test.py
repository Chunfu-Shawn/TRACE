"""CPU smoke test for the current sequence-only TRACE model."""

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.base_model import BaseModel
from model.prediction_heads import PsiteDensityHead, TERegressionHead


def main():
    model = BaseModel.from_config(
        str(SRC_DIR / "config/base_model_384d_16h_12l_64env_16ad.yaml")
    )
    model.add_head("count", PsiteDensityHead.create_from_model(model, d_pred_h=384))
    model.add_head("te", TERegressionHead.create_from_model(model))

    expression_dict = torch.load(
        SRC_DIR / "config/human_expression_dict.pt", map_location="cpu"
    )
    model.load_expression_dict(expression_dict)
    cell_type = next(iter(expression_dict))
    outputs = model.predict(
        seq_batch=["AUGCCGAUGCAG", "AUGCCG"],
        species=["human", "human"],
        cell_type=[cell_type, cell_type],
        head_names=["count", "te"],
    )

    assert outputs["count"].shape == (2, 12, 1)
    assert outputs["te"].shape == (2, 1)
    assert torch.count_nonzero(outputs["count"][1, 6:]) == 0
    assert torch.isfinite(outputs["count"]).all()
    assert torch.isfinite(outputs["te"]).all()
    print("TRACE CPU smoke test passed")


if __name__ == "__main__":
    main()
