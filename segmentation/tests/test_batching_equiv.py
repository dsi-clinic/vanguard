"""CPU equivalence test for the batched vessel inference.

Proves the batched path (`predict_fast.predict_vessel_batched`) is *bit-identical*
to the original streaming loop (`model_utils.pred_and_save_masks_3d_divided`) when
using the same float16 accumulator and the same weights.

Tiny synthetic data + a deterministic Conv3d model — runs in a few seconds on
CPU, safe for the login node. No GPU, no real data, no large memory.

Run:  python segmentation/tests/test_batching_equiv.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # segmentation/

import predict_fast  # noqa: E402  (also puts the submodule on sys.path)
import torchio as tio  # noqa: E402
from dataset_3d import Dataset3DDivided  # noqa: E402
from model_utils import pred_and_save_masks_3d_divided  # noqa: E402


class TinyDVNet(nn.Module):
    """Deterministic 2->3 channel, shape-preserving model (stand-in for UNet3D)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(2, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the deterministic conv layer."""
        return self.conv(x)


def _write_subject(
    image_dir: Path,
    add_dir: Path,
    name: str,
    shape: tuple[int, int, int],
    rng: np.random.Generator,
) -> None:
    np.save(image_dir / f"{name}.npy", rng.standard_normal(shape).astype(np.float32))
    np.save(add_dir / f"{name}.npy", rng.random(shape).astype(np.float32))


def main() -> int:
    """Run the batched-vs-streaming vessel inference equivalence check."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image_dir = tmp / "images"
        add_dir = tmp / "breast"
        out_orig = tmp / "out_orig"
        out_fast = tmp / "out_fast"
        for d in (image_dir, add_dir, out_orig, out_fast):
            d.mkdir(parents=True)

        # Two subjects of different size to exercise per-subject accumulators.
        _write_subject(image_dir, add_dir, "subjA", (40, 40, 30), rng)
        _write_subject(image_dir, add_dir, "subjB", (36, 44, 28), rng)

        # Save weights the way the loaders expect (DataParallel-prefixed keys).
        ckpt = nn.DataParallel(TinyDVNet())
        ckpt_path = tmp / "tiny.pth"
        torch.save(ckpt.state_dict(), ckpt_path)

        def make_dataset() -> Dataset3DDivided:
            return Dataset3DDivided(
                image_dir=str(image_dir),
                mask_dir=None,
                additional_input_dir=str(add_dir),
                input_dim=16,
                x_y_divisions=3,
                z_division=2,
                transforms=tio.Compose([]),
                one_hot_mask=True,
                image_only=True,
            )

        # --- Original streaming implementation -----------------------------
        pred_and_save_masks_3d_divided(
            TinyDVNet(),
            str(ckpt_path),
            make_dataset(),
            3,
            str(out_orig),
            num_workers=0,
        )

        # --- Batched implementation (float16 accum, AMP off) ---------------
        unet = predict_fast.load_model(TinyDVNet(), str(ckpt_path), device)
        predict_fast.predict_vessel_batched(
            unet,
            make_dataset(),
            n_classes=3,
            save_masks_dir=str(out_fast),
            device=device,
            batch_size=7,
            num_workers=0,
            use_amp=False,
            accum_dtype=np.float16,
        )

        # --- Compare --------------------------------------------------------
        ok = True
        for name in ("subjA", "subjB"):
            a = np.load(out_orig / f"{name}.npz")["vessel"]
            b = np.load(out_fast / f"{name}.npz")["vessel"]
            identical = np.array_equal(a, b)
            max_abs = float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
            print(
                f"{name}: shape={a.shape} bit_identical={identical} max_abs_diff={max_abs:g}"
            )
            ok = ok and identical

        if ok:
            print(
                "\nPASS: batched vessel inference is bit-identical to the original loop."
            )
            return 0
        print("\nFAIL: outputs differ.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
