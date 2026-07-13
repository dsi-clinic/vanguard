"""Translation-only motion correction for archived breast DCE-MRI exams."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import math
import os
import shutil
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from preprocessing.spgr import (
    ExamRecord,
    PreprocessingContractError,
    load_mask,
    load_nifti_member,
)


@dataclass(frozen=True)
class MotionSettings:
    """Tracked translation-registration settings."""

    downsample_xyz: tuple[int, int, int] = (4, 4, 2)
    upsample_factor: int = 4
    max_translation_mm: float = 30.0
    minimum_correlation_delta: float = 0.0
    mask_dilation_iterations: int = 5
    maximum_correlation_voxels: int = 250_000


DEFAULT_MOTION_SETTINGS = MotionSettings()
MINIMUM_CORRELATION_VOXELS = 32
MINIMUM_REGISTRATION_STANDARD_DEVIATION = 1e-6


class _HashingWriter:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        return self.stream.write(data)

    def flush(self) -> None:
        self.stream.flush()

    def tell(self) -> int:
        return self.stream.tell()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _safe_slug(value: str) -> str:
    slug = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(value).strip()
    ).strip("._")
    if not slug:
        raise ValueError(f"could not make a safe path component from {value!r}")
    return slug[:180]


def motion_shard_dir(output_root: Path, record: ExamRecord) -> Path:
    """Return the deterministic output directory for one manifest row."""
    name = f"{record.row_index:04d}_{_safe_slug(record.exam_id)}"
    return Path(output_root) / "shards" / name


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = int(size)
    info.mode = 0o640
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _add_payload(archive: tarfile.TarFile, payload: bytes, member_name: str) -> None:
    import io

    archive.addfile(_tar_info(member_name, len(payload)), io.BytesIO(payload))


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    partial.replace(path)


def _robust_registration_image(
    volume: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise PreprocessingContractError("registration input contains nonfinite signal")
    if mask is not None and np.any(mask):
        values = array[mask]
    else:
        values = array[array > 0]
    if values.size < MINIMUM_CORRELATION_VOXELS:
        raise PreprocessingContractError("registration support contains too few voxels")
    lower, upper = np.percentile(values, [1.0, 99.5])
    if not np.isfinite(upper) or upper <= lower:
        raise PreprocessingContractError(
            "registration support has degenerate intensity"
        )
    clipped = np.clip(array, float(lower), float(upper))
    normalized_values = clipped[mask] if mask is not None and np.any(mask) else clipped
    mean = float(np.mean(normalized_values))
    standard_deviation = float(np.std(normalized_values))
    if (
        not np.isfinite(standard_deviation)
        or standard_deviation <= MINIMUM_REGISTRATION_STANDARD_DEVIATION
    ):
        raise PreprocessingContractError("registration support has degenerate variance")
    return np.asarray((clipped - mean) / standard_deviation, dtype=np.float32)


def correlation_in_support(
    first: np.ndarray,
    second: np.ndarray,
    support: np.ndarray | None,
    *,
    maximum_voxels: int,
) -> float:
    """Compute Pearson correlation on deterministic support samples."""
    if support is None or not np.any(support):
        support = (first > 0) & (second > 0)
    indices = np.flatnonzero(np.asarray(support, dtype=bool).ravel())
    if indices.size < MINIMUM_CORRELATION_VOXELS:
        return float("nan")
    if indices.size > int(maximum_voxels):
        stride = int(math.ceil(indices.size / int(maximum_voxels)))
        indices = indices[::stride]
    first_values = np.asarray(first, dtype=np.float32).ravel()[indices]
    second_values = np.asarray(second, dtype=np.float32).ravel()[indices]
    finite = np.isfinite(first_values) & np.isfinite(second_values)
    if int(finite.sum()) < MINIMUM_CORRELATION_VOXELS:
        return float("nan")
    first_centered = first_values[finite] - float(first_values[finite].mean())
    second_centered = second_values[finite] - float(second_values[finite].mean())
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    if denominator <= 0:
        return float("nan")
    return float(np.dot(first_centered, second_centered) / denominator)


def correct_phase(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    support: np.ndarray | None,
    spacing_xyz_mm: np.ndarray,
    settings: MotionSettings = DEFAULT_MOTION_SETTINGS,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Propose a translation and keep it only when correlation does not worsen."""
    fixed_array = np.asarray(fixed, dtype=np.float32)
    moving_array = np.asarray(moving, dtype=np.float32)
    if fixed_array.shape != moving_array.shape:
        raise PreprocessingContractError("fixed and moving phase shapes differ")
    if np.any(fixed_array < 0) or np.any(moving_array < 0):
        raise PreprocessingContractError(
            "motion correction requires nonnegative signal"
        )

    fixed_registration = _robust_registration_image(fixed_array, support)
    moving_registration = _robust_registration_image(moving_array, support)
    downsample = tuple(max(1, int(value)) for value in settings.downsample_xyz)
    fixed_small = fixed_registration[
        :: downsample[0],
        :: downsample[1],
        :: downsample[2],
    ]
    moving_small = moving_registration[
        :: downsample[0],
        :: downsample[1],
        :: downsample[2],
    ]
    shift_small, error, difference_phase = phase_cross_correlation(
        fixed_small,
        moving_small,
        upsample_factor=max(1, int(settings.upsample_factor)),
        normalization=None,
    )
    proposed_shift = np.asarray(shift_small, dtype=np.float64) * np.asarray(
        downsample,
        dtype=np.float64,
    )
    proposed_translation_mm = proposed_shift * np.asarray(
        spacing_xyz_mm,
        dtype=np.float64,
    )
    proposed_translation_norm_mm = float(np.linalg.norm(proposed_translation_mm))
    if proposed_translation_norm_mm > float(settings.max_translation_mm):
        message = (
            f"implausible proposed translation: {proposed_translation_norm_mm:.3f} mm"
        )
        raise PreprocessingContractError(message)

    proposed = ndimage.shift(
        moving_array,
        shift=tuple(float(value) for value in proposed_shift),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)
    if not np.all(np.isfinite(proposed)) or np.any(proposed < 0):
        message = "linear motion resampling produced invalid or negative signal"
        raise PreprocessingContractError(message)

    raw_correlation = correlation_in_support(
        fixed_array,
        moving_array,
        support,
        maximum_voxels=settings.maximum_correlation_voxels,
    )
    proposed_correlation = correlation_in_support(
        fixed_array,
        proposed,
        support,
        maximum_voxels=settings.maximum_correlation_voxels,
    )
    proposed_delta = (
        float(proposed_correlation - raw_correlation)
        if np.isfinite(raw_correlation) and np.isfinite(proposed_correlation)
        else float("nan")
    )
    accepted = bool(
        np.isfinite(proposed_delta)
        and proposed_delta >= float(settings.minimum_correlation_delta)
    )
    corrected = proposed if accepted else moving_array
    accepted_shift = proposed_shift if accepted else np.zeros(3, dtype=np.float64)
    accepted_translation_mm = (
        proposed_translation_mm if accepted else np.zeros(3, dtype=np.float64)
    )
    accepted_correlation = proposed_correlation if accepted else raw_correlation
    metrics: dict[str, object] = {
        "raw_corr_to_phase0": raw_correlation,
        "proposed_registered_corr_to_phase0": proposed_correlation,
        "proposed_corr_delta": proposed_delta,
        "registered_corr_to_phase0": accepted_correlation,
        "corr_delta": (
            float(accepted_correlation - raw_correlation)
            if np.isfinite(raw_correlation) and np.isfinite(accepted_correlation)
            else float("nan")
        ),
        "transform_accepted": accepted,
        "transform_rejection_reason": (
            "" if accepted else "nonfinite_or_corr_delta_below_minimum"
        ),
        "proposed_translation_voxels": proposed_shift.tolist(),
        "proposed_translation_mm": proposed_translation_mm.tolist(),
        "proposed_translation_norm_mm": proposed_translation_norm_mm,
        "translation_voxels": accepted_shift.tolist(),
        "translation_mm": accepted_translation_mm.tolist(),
        "translation_norm_mm": float(np.linalg.norm(accepted_translation_mm)),
        "phasecorr_error": float(error),
        "phasecorr_difference_phase": float(difference_phase),
    }
    return np.asarray(corrected, dtype=np.float32), proposed, metrics


def _encode_nifti(
    array: np.ndarray,
    *,
    reference: nib.Nifti1Image,
) -> bytes:
    header = reference.header.copy()
    header.set_data_shape(array.shape)
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(
        np.ascontiguousarray(array, dtype=np.float32),
        reference.affine,
        header=header,
    )
    image.set_qform(reference.affine, code=int(reference.header["qform_code"]))
    image.set_sform(reference.affine, code=int(reference.header["sform_code"]))
    return gzip.compress(image.to_bytes(), compresslevel=6, mtime=0)


def _qc_slices(
    fixed: np.ndarray,
    moving: np.ndarray,
    proposed: np.ndarray,
    corrected: np.ndarray,
) -> tuple[list[np.ndarray], str]:
    axis = int(np.argmin(fixed.shape))
    reduction_axes = tuple(index for index in range(3) if index != axis)
    score = np.abs(moving - fixed).sum(axis=reduction_axes)
    slice_index = int(np.argmax(score))
    slices = [
        np.take(array, slice_index, axis=axis).T
        for array in (fixed, moving, proposed, corrected)
    ]
    return slices, f"{('x', 'y', 'z')[axis]}={slice_index}"


def _scale_qc_images(images: list[np.ndarray]) -> list[np.ndarray]:
    values = np.concatenate(
        [np.asarray(image, dtype=np.float32).ravel() for image in images]
    )
    finite_positive = values[np.isfinite(values) & (values > 0)]
    if finite_positive.size:
        lower, upper = np.percentile(finite_positive, [1.0, 99.5])
    else:
        lower, upper = 0.0, 1.0
    if not np.isfinite(upper) or upper <= lower:
        upper = lower + 1.0
    return [
        (
            np.clip((np.asarray(image) - lower) / (upper - lower), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        for image in images
    ]


def _render_qc_panel(
    path: Path,
    *,
    exam_id: str,
    phase_index: int,
    slice_description: str,
    images: list[np.ndarray],
    metrics: dict[str, object],
) -> None:
    scaled = _scale_qc_images(images)
    labels = ("phase 0", "raw", "proposed", "saved")
    tile_height, tile_width = scaled[0].shape
    header_height = 86
    label_height = 28
    sheet = Image.new(
        "RGB",
        (tile_width * len(scaled), header_height + label_height + tile_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 8),
        f"{exam_id} | phase {phase_index} | {slice_description}",
        fill=(0, 0, 0),
    )
    draw.text(
        (8, 32),
        (
            f"raw corr={float(metrics['raw_corr_to_phase0']):.4f} | "
            f"proposed corr={float(metrics['proposed_registered_corr_to_phase0']):.4f} | "
            f"accepted={bool(metrics['transform_accepted'])}"
        ),
        fill=(0, 0, 0),
    )
    draw.text(
        (8, 54),
        (
            f"proposed shift={float(metrics['proposed_translation_norm_mm']):.3f} mm | "
            f"saved shift={float(metrics['translation_norm_mm']):.3f} mm"
        ),
        fill=(0, 0, 0),
    )
    for index, (label, array) in enumerate(zip(labels, scaled, strict=True)):
        x_position = index * tile_width
        draw.text((x_position + 6, header_height + 6), label, fill=(0, 0, 0))
        sheet.paste(
            Image.fromarray(array).convert("RGB"),
            (x_position, header_height + label_height),
        )
    sheet.save(path)


def motion_correct_exam_to_shard(
    record: ExamRecord,
    *,
    output_root: Path,
    settings: MotionSettings = DEFAULT_MOTION_SETTINGS,
) -> dict[str, object]:
    """Motion-correct one exam and atomically publish its restartable shard."""
    if record.motion_correction_applied:
        raise PreprocessingContractError(
            "refusing to motion-correct an already corrected exam"
        )
    if record.mask_archive_path is None:
        raise PreprocessingContractError(
            "motion correction requires a whole-breast mask"
        )

    final_directory = motion_shard_dir(Path(output_root), record)
    metadata_path = final_directory / "metadata.json"
    shard_path = final_directory / "phase_images.tar"
    if final_directory.exists():
        if not metadata_path.is_file() or not shard_path.is_file():
            message = f"incomplete existing shard: {final_directory}"
            raise PreprocessingContractError(message)
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("status") != "complete"
            or metadata.get("exam_id") != record.exam_id
        ):
            message = f"invalid existing shard metadata: {metadata_path}"
            raise PreprocessingContractError(message)
        return dict(metadata)

    final_directory.parent.mkdir(parents=True, exist_ok=True)
    partial_directory = final_directory.with_name(
        f".{final_directory.name}.partial.{os.getpid()}"
    )
    if partial_directory.exists():
        shutil.rmtree(partial_directory)
    partial_directory.mkdir(parents=True)
    partial_shard = partial_directory / "phase_images.tar"

    output_members: list[str] = []
    output_hashes: list[str] = []
    output_sizes: list[int] = []
    metrics_rows: list[dict[str, object]] = []
    best_qc: tuple[float, int, list[np.ndarray], str, dict[str, object]] | None = None
    with tarfile.open(record.phase_archive_path, mode="r") as source_archive:
        fixed_image, fixed_payload = load_nifti_member(
            source_archive,
            record.phase_archive_members[0],
            expected_sha256=record.phase_member_sha256[0],
        )
        fixed = np.asarray(fixed_image.dataobj, dtype=np.float32)
        if not np.all(np.isfinite(fixed)) or np.any(fixed < 0):
            raise PreprocessingContractError(
                f"invalid phase 0 signal: {record.exam_id}"
            )
        affine = np.asarray(fixed_image.affine, dtype=np.float64)
        spacing_xyz_mm = np.linalg.norm(affine[:3, :3], axis=0)
        if not np.all(np.isfinite(spacing_xyz_mm)) or np.any(spacing_xyz_mm <= 0):
            raise PreprocessingContractError(f"invalid image spacing: {record.exam_id}")
        breast_mask = load_mask(record, shape=fixed.shape, affine=affine)
        if breast_mask is None or not np.any(breast_mask > 0):
            raise PreprocessingContractError(f"empty breast support: {record.exam_id}")
        support = breast_mask > 0
        if int(settings.mask_dilation_iterations) > 0:
            support = ndimage.binary_dilation(
                support,
                iterations=int(settings.mask_dilation_iterations),
            )

        with partial_shard.open("wb") as shard_stream:
            hashing_stream = _HashingWriter(shard_stream)
            with tarfile.open(fileobj=hashing_stream, mode="w|") as output_archive:
                for phase_index in range(record.n_phases):
                    member_name = (
                        f"motion_corrected_phase_images/{_safe_slug(record.dataset)}/"
                        f"{_safe_slug(record.exam_id)}/phase_{phase_index:04d}.nii.gz"
                    )
                    if phase_index == 0:
                        payload = fixed_payload
                    else:
                        moving_image, _ = load_nifti_member(
                            source_archive,
                            record.phase_archive_members[phase_index],
                            expected_sha256=record.phase_member_sha256[phase_index],
                        )
                        moving = np.asarray(moving_image.dataobj, dtype=np.float32)
                        if moving.shape != fixed.shape or not np.allclose(
                            moving_image.affine,
                            fixed_image.affine,
                            rtol=0.0,
                            atol=1e-5,
                        ):
                            message = (
                                f"within-exam geometry mismatch: {record.exam_id}, "
                                f"phase {phase_index}"
                            )
                            raise PreprocessingContractError(message)
                        corrected, proposed, metrics = correct_phase(
                            fixed,
                            moving,
                            support=support,
                            spacing_xyz_mm=spacing_xyz_mm,
                            settings=settings,
                        )
                        metrics = {
                            "exam_id": record.exam_id,
                            "dataset": record.dataset,
                            "phase_index": phase_index,
                            "source_archive_member": record.phase_archive_members[
                                phase_index
                            ],
                            "registered_archive_member": member_name,
                            **metrics,
                        }
                        metrics_rows.append(metrics)
                        payload = _encode_nifti(corrected, reference=fixed_image)
                        qc_score = float(
                            metrics["proposed_translation_norm_mm"]
                        ) + 10.0 * abs(
                            float(metrics["proposed_corr_delta"])
                            if np.isfinite(float(metrics["proposed_corr_delta"]))
                            else 0.0
                        )
                        if best_qc is None or qc_score > best_qc[0]:
                            slices, slice_description = _qc_slices(
                                fixed,
                                moving,
                                proposed,
                                corrected,
                            )
                            best_qc = (
                                qc_score,
                                phase_index,
                                slices,
                                slice_description,
                                metrics,
                            )
                    _add_payload(output_archive, payload, member_name)
                    output_members.append(member_name)
                    output_hashes.append(hashlib.sha256(payload).hexdigest())
                    output_sizes.append(len(payload))
            hashing_stream.flush()
            os.fsync(shard_stream.fileno())
            shard_sha256 = hashing_stream.digest.hexdigest()

    qc_path = partial_directory / "motion_qc.png"
    qc_score = 0.0
    qc_phase_index = 0
    if best_qc is not None:
        qc_score, qc_phase_index, images, slice_description, qc_metrics = best_qc
        _render_qc_panel(
            qc_path,
            exam_id=record.exam_id,
            phase_index=qc_phase_index,
            slice_description=slice_description,
            images=images,
            metrics=qc_metrics,
        )

    accepted = sum(bool(row["transform_accepted"]) for row in metrics_rows)
    metadata: dict[str, object] = {
        "status": "complete",
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "row_index": record.row_index,
        "exam_id": record.exam_id,
        "dataset": record.dataset,
        "n_phases": record.n_phases,
        "source_phase_archive_path": str(record.phase_archive_path),
        "source_phase_archive_members": list(record.phase_archive_members),
        "output_phase_archive_members": output_members,
        "output_phase_member_sha256": output_hashes,
        "output_phase_member_bytes": output_sizes,
        "shard_archive_sha256": shard_sha256,
        "motion_correction_applied": True,
        "motion_correction_method": "translation_phasecorr_phase0_corr_gate",
        "reference_phase_index": 0,
        "settings": asdict(settings),
        "voxel_spacing_xyz_mm": spacing_xyz_mm.tolist(),
        "transforms_accepted": accepted,
        "transforms_rejected": len(metrics_rows) - accepted,
        "registration_metrics": metrics_rows,
        "qc_score": float(qc_score),
        "qc_phase_index": int(qc_phase_index),
        "qc_panel": str(final_directory / "motion_qc.png") if best_qc else "",
    }
    _atomic_write_json(partial_directory / "metadata.json", metadata)
    partial_directory.replace(final_directory)
    return metadata
