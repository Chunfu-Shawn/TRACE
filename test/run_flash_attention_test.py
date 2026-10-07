"""Verify native CUDA FlashAttention and run the three-transcript FASTA demo."""

import argparse
import pickle
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
CHECKPOINT_NAME = (
    "base_model_384d_16h_12l_64env_16ad_bs-PsiteDensityHead."
    "hs_22c_18c_26c_rm_4c_mm_3c_6k_depth0.1_cov0.1_rpm1_"
    "e50_a1_b0_exp_aug_i03_m15.150_0.001.best_profile.pt"
)
DEMO_LENGTHS = {
    "ENST00000412698": 1810,
    "ENST00000005995": 1091,
    "ENST00000518804": 612,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "checkpoint" / CHECKPOINT_NAME)
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index.")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "flash_attention_demo")
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
    for package in ("flash-attn", "ninja", "packaging", "psutil"):
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            print(f"{package}: not installed")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Native FlashAttention requires a supported GPU environment.")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}; pass --checkpoint with its actual path.")

    import model.flash_multi_headed_attention as flash_module
    from model.base_model import BaseModel
    from model.model_modules import MultiHeadedAttention
    from model.prediction_heads import PsiteDensityHead
    from model.translation_predictor import TranslationProfilePredictor

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}; capability: {torch.cuda.get_device_capability(device)}")
    torch.manual_seed(42)
    model = BaseModel.from_config(
        str(PROJECT_ROOT / "src/config/base_model_384d_16h_12l_64env_16ad_bs.yaml")
    )
    model.add_head("count", PsiteDensityHead.create_from_model(model, d_pred_h=384))
    loaded = model.load_pretrained_weights(str(args.checkpoint), strict=False, map_location="cpu")
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {loaded}")
    model.load_expression_dict(torch.load(
        PROJECT_ROOT / "src/config/human_expression_dict.pt", map_location="cpu"
    ))
    model.to(device).eval()

    layers = model.encoder.encoder_layers
    if not all(isinstance(layer.multi_headed_attention, flash_module.FlashMultiHeadedAttention)
               for layer in layers):
        raise RuntimeError("The model did not initialize every layer with FlashMultiHeadedAttention.")
    print(f"FlashAttention layers: {len(layers)}; head dimension: {model.d_model // model.n_heads}")

    native_kernel = flash_module.flash_attn_varlen_qkvpacked_func
    native_calls = 0

    def counted_native_kernel(*kernel_args, **kernel_kwargs):
        nonlocal native_calls
        output = native_kernel(*kernel_args, **kernel_kwargs)
        native_calls += 1
        return output

    # Disable fallback only in this test, so a native-kernel failure cannot pass silently.
    with patch.object(flash_module, "flash_attn_varlen_qkvpacked_func", counted_native_kernel), \
         patch.object(flash_module.FlashMultiHeadedAttention, "_standard_attention",
                      side_effect=RuntimeError("Standard-attention fallback was attempted; native FlashAttention failed.")):
        flash_layer = layers[0].multi_headed_attention
        standard_layer = MultiHeadedAttention(model.d_model, model.n_heads).to(device).eval()
        standard_loaded = standard_layer.load_state_dict(flash_layer.state_dict(), strict=False)
        if standard_loaded.missing_keys or standard_loaded.unexpected_keys != ["softmax_scale"]:
            raise RuntimeError(f"Attention weight mismatch: {standard_loaded}")
        tokens = torch.randn(2, 96, model.d_model, device=device)
        lengths = torch.tensor([96, 61], device=device)
        mask = torch.arange(96, device=device)[None, :] < lengths[:, None]
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            flash_output = flash_layer(tokens, tokens, attention_mask=mask)
            standard_output = standard_layer(tokens, tokens, attention_mask=mask)
        torch.testing.assert_close(flash_output[mask].float(), standard_output[mask].float(),
                                   rtol=0.02, atol=0.02)
        max_error = (flash_output[mask].float() - standard_output[mask].float()).abs().max().item()
        print(f"Native/standard attention agreement passed; max absolute error: {max_error:.6g}")
        del standard_layer, tokens, flash_output, standard_output

        with torch.amp.autocast("cuda", dtype=torch.float16):
            prediction = model.predict(
                seq_batch=["AUGCCGAUGCAG", "AUGCCG"], species="human",
                cell_type="liver", head_names=["count"],
            )["count"]
        assert prediction.shape == (2, 12, 1)
        assert torch.isfinite(prediction).all()
        assert torch.count_nonzero(prediction[1, 6:]) == 0
        del prediction

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started_at = time.perf_counter()
        predictor = TranslationProfilePredictor(
            model, str(PROJECT_ROOT / "test/gencode.v43.pc_transcripts.test_2000.fa")
        )
        output_path = predictor.run(
            species="human", cell_type="liver",
            cell_expr_vector=model.cell_expr_dict["liver"].cpu().numpy(),
            target_tids=list(DEMO_LENGTHS), out_dir=str(args.out_dir),
            suffix="liver_flash_demo", min_len=200, max_len=10000,
            batch_size=1, num_workers=0,
        )
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started_at

    with open(output_path, "rb") as handle:
        profiles = pickle.load(handle)["liver"]
    assert set(profiles) == set(DEMO_LENGTHS)
    for transcript_id, length in DEMO_LENGTHS.items():
        profile = profiles[transcript_id]
        assert profile.shape == (length,) and profile.dtype == np.float16
        assert np.isfinite(profile).all() and (profile >= 0).all()
        print(f"{transcript_id}: shape={profile.shape}, dtype={profile.dtype}")
    expected_calls = 1 + len(layers) * (1 + len(DEMO_LENGTHS))
    assert native_calls == expected_calls, (native_calls, expected_calls)
    print(f"Successful native FlashAttention kernel calls: {native_calls}")
    print(f"FASTA demo elapsed time: {elapsed_seconds:.2f} s")
    print(f"Peak PyTorch GPU memory allocated: {torch.cuda.max_memory_allocated(device) / 2**30:.3f} GiB")
    print(f"Predictions saved to: {output_path}")
    print("TRACE native FlashAttention demo passed (no standard-attention fallback).")


if __name__ == "__main__":
    main()
