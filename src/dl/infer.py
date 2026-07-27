from pathlib import Path
from shutil import rmtree

import cv2
import hydra
import numpy as np
from numbers import Integral

from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.dl.dataset import read_image_hwc
from src.dl.utils import (
    Visualizer,
    abs_xyxy_to_norm_xywh,
    get_latest_experiment_name,
    overlay_sem_seg,
    sem_seg_palette,
)
from src.infer.byte_track import ByteTrack, Detection
from src.infer.torch_model import Torch_model

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def figure_input_type(folder_path: Path):
    video_types = ["mp4", "avi", "mov", "mkv"]
    # .tif/.tiff intentionally excluded: cv2.imread mangles 4-channel TIFFs
    # (alpha pre-multiplication + photometric-tag swap). Convert to .npy first
    # (see src/etl/preprocess.py for a PIL-based TIFF->JPG path for 3-channel).
    img_types = ["jpg", "png", "jpeg", "npy"]

    for f in folder_path.iterdir():
        if f.suffix[1:].lower() in video_types:
            data_type = "video"
            break
        elif f.suffix[1:].lower() in img_types:
            data_type = "image"
            break
    logger.info(
        f"Inferencing on data type: {data_type}, path: {folder_path}",
    )
    return data_type


def visualize(img, boxes, labels, scores, output_path, img_path, label_to_name, masks=None):
    output_path.mkdir(parents=True, exist_ok=True)
    results = {"boxes": boxes, "labels": labels, "scores": scores}
    if masks is not None:
        results["masks"] = masks
    vis = Visualizer(n_classes=max(label_to_name.keys()) + 1, class_names=label_to_name)
    img = vis.draw(img, results)
    if len(boxes):
        cv2.imwrite((str(f"{output_path / Path(img_path).stem}.jpg")), img)


def save_yolo_annotations(res, output_path, img_path, img_shape):
    output_path.mkdir(parents=True, exist_ok=True)

    if len(res["boxes"]) == 0:
        return

    has_polys = "polys" in res and res["polys"] is not None and len(res["polys"]) > 0

    with open(output_path / f"{Path(img_path).stem}.txt", "a") as f:
        for idx, (class_id, box) in enumerate(zip(res["labels"], res["boxes"])):
            if has_polys:
                # YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
                poly = res["polys"][idx]
                if len(poly) >= 3:  # Need at least 3 points for a valid polygon
                    norm_coords = []
                    for point in poly:
                        norm_coords.append(f"{point[0]:.6f}")
                        norm_coords.append(f"{point[1]:.6f}")
                    f.write(f"{int(class_id)} {' '.join(norm_coords)}\n")
            else:
                # YOLO detection format: class_id x_center y_center width height
                norm_box = abs_xyxy_to_norm_xywh(box[None], img_shape[0], img_shape[1])[0]
                f.write(
                    f"{int(class_id)} {norm_box[0]:.6f} {norm_box[1]:.6f} {norm_box[2]:.6f} {norm_box[3]:.6f}\n"
                )


def crops(or_img, res, paddings, output_path, output_stem):
    if isinstance(paddings["w"], float):
        paddings["w"] = int(or_img.shape[1] * paddings["w"])
    if isinstance(paddings["h"], float):
        paddings["h"] = int(or_img.shape[0] * paddings["h"])

    for crop_id, box in enumerate(res["boxes"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = or_img[
            max(y1 - paddings["h"], 0) : min(y2 + paddings["h"], or_img.shape[0]),
            max(x1 - paddings["w"], 0) : min(x2 + paddings["w"], or_img.shape[1]),
        ]

        (output_path / "crops").mkdir(parents=True, exist_ok=True)
        cv2.imwrite((str(output_path / "crops" / f"{output_stem}_{crop_id}.jpg")), crop)


def run_images(
    torch_model, folder_path, output_path, label_to_name, to_crop, paddings, conf_thresh
):
    batch = 0
    imag_paths = [img.name for img in folder_path.iterdir() if not str(img).startswith(".")]
    labels = set()
    for img_path in tqdm(imag_paths):
        img = read_image_hwc(folder_path / img_path)
        if img is None:
            logger.warning(f"Skipping unreadable image: {img_path}")
            continue
        or_img = img.copy()
        is_npy = Path(img_path).suffix.lower() == ".npy"
        raw_res = torch_model(img, bgr=not is_npy)

        # Convert torch tensors to numpy for saving/visualization
        res = {
            "boxes": raw_res[batch]["boxes"].cpu().numpy(),
            "labels": raw_res[batch]["labels"].cpu().numpy(),
            "scores": raw_res[batch]["scores"].cpu().numpy(),
        }
        if "masks" in raw_res[0]:
            res["masks"] = raw_res[batch]["masks"].cpu()
            res["polys"] = torch_model.mask2poly(res["masks"], img.shape)

        # visualization / crops only support 3-channel; slice for N>3.
        # cv2 saves in BGR; .npy stacks are RGB(+extras) by convention.
        vis_img = img[:, :, :3] if img.shape[2] > 3 else img
        crop_img = or_img[:, :, :3] if or_img.shape[2] > 3 else or_img
        if is_npy:
            vis_img = np.ascontiguousarray(vis_img[..., ::-1])
            crop_img = np.ascontiguousarray(crop_img[..., ::-1])

        visualize(
            img=vis_img,
            boxes=res["boxes"],
            labels=res["labels"],
            scores=res["scores"],
            output_path=output_path / "images",
            img_path=img_path,
            label_to_name=label_to_name,
            masks=res.get("masks", None),
        )

        for class_id in res["labels"]:
            labels.add(class_id)

        save_yolo_annotations(
            res=res, output_path=output_path / "labels", img_path=img_path, img_shape=img.shape
        )

        if to_crop:
            crops(crop_img, res, paddings, output_path, Path(img_path).stem)

    with open(output_path / "labels.txt", "w") as f:
        for class_id in labels:
            f.write(f"{label_to_name[int(class_id)]}\n")


def _normalize_tile_size(tile_size):
    if isinstance(tile_size, Integral) and not isinstance(tile_size, bool):
        tile_height = tile_width = int(tile_size)
    elif isinstance(tile_size, (tuple, list)) and len(tile_size) == 2:
        tile_height, tile_width = tile_size
        if not all(
            isinstance(value, Integral) and not isinstance(value, bool)
            for value in (tile_height, tile_width)
        ):
            raise ValueError("tile_size values must be integers")
        tile_height, tile_width = int(tile_height), int(tile_width)
    else:
        raise ValueError("tile_size must be an int or a (height, width) pair")

    if tile_height <= 0 or tile_width <= 0:
        raise ValueError("tile_size values must be greater than zero")

    return tile_height, tile_width


def _tile_starts(length, tile_length, overlap):
    if length <= 0:
        raise ValueError("image dimensions must be greater than zero")
    if tile_length <= 0:
        raise ValueError("tile dimensions must be greater than zero")
    if not 0 <= overlap < 1:
        raise ValueError("tile_overlap must be in the range [0, 1)")
    if tile_length >= length:
        return [0]

    stride = max(1, int(round(tile_length * (1 - overlap))))
    starts = list(range(0, length - tile_length + 1, stride))
    final_start = length - tile_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _to_numpy(prediction):
    if hasattr(prediction, "detach"):
        prediction = prediction.detach()
    if hasattr(prediction, "cpu"):
        prediction = prediction.cpu()
    if hasattr(prediction, "numpy"):
        prediction = prediction.numpy()
    return np.asarray(prediction)


def _run_sliced_sem_seg(
    torch_model,
    image,
    *,
    bgr,
    num_classes,
    use_instance_segmentation_dataset=False,
    tile_size=1024,
    tile_overlap=0.2,
    img_path=None
):
    """Infer overlapping tiles and merge them into one full-size class map."""
    tile_height, tile_width = _normalize_tile_size(tile_size)
    if not 0 <= tile_overlap < 1:
        raise ValueError("tile_overlap must be in the range [0, 1)")
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero")

    image_height, image_width = image.shape[:2]
    y_starts = _tile_starts(image_height, tile_height, tile_overlap)
    x_starts = _tile_starts(image_width, tile_width, tile_overlap)

    score_sum = None
    score_count = None
    class_votes = None
    output_rank = None

    for y1 in y_starts:
        y2 = min(y1 + tile_height, image_height)
        for x1 in x_starts:
            x2 = min(x1 + tile_width, image_width)
            
            tile = image[y1:y2, x1:x2]
            prediction = _to_numpy(torch_model(tile, bgr=bgr)[0]["sem_seg"])

            # debug_tiles_dir = Path("debug_tiles")
            # debug_tiles_dir.mkdir(parents=True, exist_ok=True)

            # tile = image[y1:y2, x1:x2]

            # # OpenCV expects BGR when saving.
            # debug_tile = tile[..., :3]
            # if not bgr:
            #     debug_tile = cv2.cvtColor(debug_tile, cv2.COLOR_RGB2BGR)

            # debug_path = debug_tiles_dir / (
            #     f"{Path(img_path).stem}_x{x1}-{x2}_y{y1}-{y2}.png"
            # )

            # if not cv2.imwrite(str(debug_path), debug_tile):
            #     logger.warning(f"Failed to save debug tile: {debug_path}")

            # prediction = _to_numpy(
            #     torch_model(tile, bgr=bgr)[0]["sem_seg"]
            # )

            if prediction.ndim not in (2, 3):
                raise ValueError(
                    "sem_seg must have shape (H, W) or (C, H, W); "
                    f"received {prediction.shape}"
                )
            if prediction.shape[-2:] != tile.shape[:2]:
                raise ValueError(
                    "sem_seg spatial size must match its input tile; "
                    f"received {prediction.shape[-2:]} for tile {tile.shape[:2]}"
                )
            if output_rank is None:
                output_rank = prediction.ndim
            elif prediction.ndim != output_rank:
                raise ValueError("sem_seg output rank changed between tiles")

            if prediction.ndim == 3 or use_instance_segmentation_dataset:
                if not use_instance_segmentation_dataset and prediction.shape[0] != num_classes:
                    raise ValueError(
                        f"Expected {num_classes} logit channels, "
                        f"received {prediction.shape[0]}"
                    )
                if score_sum is None:
                    if use_instance_segmentation_dataset:
                        score_sum = np.zeros(
                            (image_height, image_width),
                            dtype=np.float32,
                        )
                    else:
                        score_sum = np.zeros(
                            (num_classes, image_height, image_width),
                            dtype=np.float32,
                        )
                    score_count = np.zeros(
                        (image_height, image_width),
                        dtype=np.uint16,
                    )
                if use_instance_segmentation_dataset:
                    score_sum[y1:y2, x1:x2] += prediction.astype(
                        np.float32, copy=False
                    )
                else:
                    score_sum[:, y1:y2, x1:x2] += prediction.astype(
                        np.float32, copy=False
                    )
                score_count[y1:y2, x1:x2] += 1
            else:
                if not np.all(np.isfinite(prediction)):
                    raise ValueError("Class-map predictions contain non-finite values")
                rounded_prediction = np.rint(prediction)
                if not np.allclose(prediction, rounded_prediction):
                    raise ValueError(
                        "A 2D sem_seg output must contain integer class IDs"
                    )
                prediction = rounded_prediction.astype(np.int64, copy=False)
                if prediction.min() < 0 or prediction.max() >= num_classes:
                    raise ValueError(
                        f"Class IDs must be between 0 and {num_classes - 1}"
                    )
                if class_votes is None:
                    class_votes = np.zeros(
                        (num_classes, image_height, image_width),
                        dtype=np.uint16,
                    )
                for class_id in np.unique(prediction):
                    class_votes[class_id, y1:y2, x1:x2] += (
                        prediction == class_id
                    )

    if use_instance_segmentation_dataset:
        average_scores = score_sum / score_count
        return (average_scores > 0.5)*1
    if output_rank == 3:
        average_scores = score_sum / score_count[None, :, :]
        return np.argmax(average_scores, axis=0)
    return np.argmax(class_votes, axis=0)


def run_images_sem_seg(
    torch_model,
    folder_path,
    output_path,
    label_to_name,
    use_instance_segmentation_dataset: bool = False,
    slice_inference: bool = False,
    tile_size=1024,
    tile_overlap: float = 0.2,
):
    """Write overlay images and raw label maps, optionally using tiled inference."""
    palette = sem_seg_palette(len(label_to_name))
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)
    labels = set()
    img_paths = [
        image.name for image in folder_path.iterdir() if not image.name.startswith(".")
    ]

    for img_path in tqdm(img_paths):
        img = read_image_hwc(folder_path / img_path)
        if img is None:
            logger.warning(f"Skipping unreadable image: {img_path}")
            continue

        is_npy = Path(img_path).suffix.lower() == ".npy"
        if slice_inference:
            label_map = _run_sliced_sem_seg(
                torch_model,
                img,
                bgr=not is_npy,
                num_classes=len(label_to_name),
                use_instance_segmentation_dataset=use_instance_segmentation_dataset,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                img_path=img_path
            )
        else:
            label_map = _to_numpy(
                torch_model(img, bgr=not is_npy)[0]["sem_seg"]
            )

        vis_img = img[:, :, :3] if img.shape[2] > 3 else img
        if is_npy:
            vis_img = np.ascontiguousarray(vis_img[..., ::-1])

        cv2.imwrite(
            str(output_path / "images" / f"{Path(img_path).stem}.jpg"),
            overlay_sem_seg(
                vis_img,
                label_map,
                palette,
                binary_overlay=use_instance_segmentation_dataset,
            ),
        )

        save_label = (
            (label_map * 255).astype(np.uint8)
            if use_instance_segmentation_dataset
            else label_map.astype(np.uint8)
        )
        cv2.imwrite(
            str(output_path / "labels" / f"{Path(img_path).stem}.png"),
            save_label,
        )
        labels.update(np.unique(label_map).tolist())

    with open(output_path / "labels.txt", "w") as file:
        for class_id in sorted(labels):
            file.write(f"{label_to_name[int(class_id)]}\n")


def run_videos_sem_seg(torch_model, folder_path, output_path, label_to_name):
    """Per-frame overlay written to <stem>_sem_seg.mp4; tracking is box-based -> skipped."""
    palette = sem_seg_palette(len(label_to_name))
    output_path.mkdir(parents=True, exist_ok=True)
    video_files = sorted(
        f
        for f in folder_path.iterdir()
        if f.suffix.lower() in VIDEO_EXTS and not f.name.startswith(".")
    )
    for video_path in video_files:
        vid = cv2.VideoCapture(str(video_path))
        if not vid.isOpened():
            logger.warning(f"Could not open {video_path}, skipping")
            continue
        fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = output_path / f"{video_path.stem}_sem_seg.mp4"
        out_vid = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        pbar = tqdm(total=total_frames, desc=video_path.name, unit="frame")
        success, frame = vid.read()
        while success:
            label_map = torch_model(frame)[0]["sem_seg"].cpu().numpy()
            out_vid.write(overlay_sem_seg(frame, label_map, palette))
            pbar.update(1)
            success, frame = vid.read()
        pbar.close()
        vid.release()
        out_vid.release()
        logger.info(f"Output video saved: {out_path}")


def run_videos(
    torch_model, folder_path, output_path, label_to_name, to_crop, paddings, conf_thresh
):
    batch = 0
    vid_paths = [vid.name for vid in folder_path.iterdir() if not str(vid.name).startswith(".")]
    labels = set()
    for vid_path in vid_paths:
        vid = cv2.VideoCapture(str(folder_path / vid_path))
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        pbar = tqdm(total=total_frames, desc=vid_path, unit="frame")
        success, img = vid.read()
        idx = 0
        while success:
            idx += 1
            raw_res = torch_model(img)

            # Convert torch tensors to numpy for saving/visualization
            res = {
                "boxes": raw_res[batch]["boxes"].cpu().numpy(),
                "labels": raw_res[batch]["labels"].cpu().numpy(),
                "scores": raw_res[batch]["scores"].cpu().numpy(),
            }
            if "masks" in raw_res[0]:
                res["masks"] = raw_res[batch]["masks"].cpu()
                res["polys"] = torch_model.mask2poly(res["masks"], img.shape)

            frame_name = f"{Path(vid_path).stem}_frame_{idx}"
            visualize(
                img=img,
                boxes=res["boxes"],
                labels=res["labels"],
                scores=res["scores"],
                output_path=output_path / "images",
                img_path=frame_name,
                label_to_name=label_to_name,
                masks=res.get("masks", None),
            )

            for class_id in res["labels"]:
                labels.add(class_id)

            save_yolo_annotations(
                res=res,
                output_path=output_path / "labels",
                img_path=frame_name,
                img_shape=img.shape,
            )

            if to_crop:
                crops(img, res, paddings, output_path, frame_name)

            pbar.update(1)
            success, img = vid.read()
        pbar.close()
        vid.release()

    with open(output_path / "labels.txt", "w") as f:
        for class_id in labels:
            f.write(f"{label_to_name[int(class_id)]}\n")


def _run_video_tracked(torch_model, tracker, visualizer, video_path, output_path):
    vid = cv2.VideoCapture(str(video_path))
    if not vid.isOpened():
        logger.warning(f"Could not open {video_path}, skipping")
        return

    fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_vid = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    pbar = tqdm(total=total_frames, desc=video_path.name, unit="frame")
    success, frame = vid.read()
    while success:
        raw_res = torch_model(frame)
        res = raw_res[0]

        boxes = res["boxes"].cpu().numpy()
        labels = res["labels"].cpu().numpy()
        scores = res["scores"].cpu().numpy()

        detections = [
            Detection(bbox=tuple(b.tolist()), score=float(s), cls_id=int(c))
            for b, c, s in zip(boxes, labels, scores)
        ]
        tracked = tracker.update(detections, frame_shape=(height, width))

        if tracked:
            tracked_results = {
                "track_ids": np.array([t[0] for t in tracked], dtype=int),
                "labels": np.array([t[1] for t in tracked], dtype=int),
                "boxes": np.array([t[2] for t in tracked], dtype=np.float64),
                "scores": np.array([t[3] for t in tracked], dtype=np.float64),
            }
        else:
            tracked_results = {
                "track_ids": np.zeros(0, dtype=int),
                "labels": np.zeros(0, dtype=int),
                "boxes": np.zeros((0, 4), dtype=np.float64),
                "scores": np.zeros(0, dtype=np.float64),
            }

        out_vid.write(visualizer.draw(frame, tracked_results))
        pbar.update(1)
        success, frame = vid.read()

    pbar.close()
    vid.release()
    out_vid.release()
    logger.info(f"Output video saved: {output_path}")


def run_videos_tracked(torch_model, folder_path, output_path, label_to_name, tracker_cfg):
    video_files = sorted(
        f
        for f in folder_path.iterdir()
        if f.suffix.lower() in VIDEO_EXTS and not f.name.startswith(".")
    )
    if not video_files:
        logger.error(f"No video files found in {folder_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    visualizer = Visualizer(n_classes=max(label_to_name.keys()) + 1, class_names=label_to_name)

    for video_path in video_files:
        # Fresh tracker per video so IDs don't bleed across unrelated clips.
        tracker = ByteTrack(
            track_thresh=tracker_cfg["track_thresh"],
            unmatched_thresh=tracker_cfg["unmatched_thresh"],
            detrack_thresh=tracker_cfg["detrack_thresh"],
            tracking_thresh=tracker_cfg["tracking_thresh"],
            track_buffer=tracker_cfg["track_buffer"],
            max_age=tracker_cfg["max_age"],
            min_hits=tracker_cfg["min_hits"],
            iou_weight=tracker_cfg["iou_weight"],
            drag=tracker_cfg["drag"],
            velocity_alpha=tracker_cfg["velocity_alpha"],
        )
        out_path = output_path / f"{video_path.stem}_tracked.mp4"
        logger.info(f"Processing: {video_path}")
        _run_video_tracked(torch_model, tracker, visualizer, video_path, out_path)


@hydra.main(version_base=None, config_path="../../", config_name="config_size_m_1600_fbm_ft_from_yap_ckpt")
def main(cfg: DictConfig):
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)

    to_crop = cfg.infer.to_crop
    paddings = cfg.infer.paddings
    to_track = cfg.infer.get("to_track", True)

    folder_path = Path(str(cfg.train.path_to_test_data))
    data_type = figure_input_type(folder_path)

    # Tracking only applies to videos (and is box-based, so never for sem_seg).
    use_tracking = to_track and data_type == "video" and cfg.task != "sem_seg"

    if use_tracking:
        # ByteTrack defaults — picked to exercise the two-stage association.
        tracker_cfg = {
            "track_thresh": float(cfg.train.conf_thresh),  # high/low pool split
            "unmatched_thresh": 0.7,  # min score to start a new track
            "detrack_thresh": 0.4,  # hard floor inside the tracker
            "tracking_thresh": 0.8,  # max match cost (iou_weight*(1-IoU)+...)
            "track_buffer": 30,
            "max_age": 0,
            "min_hits": 2,
            "iou_weight": 0.75,
            "drag": 0.85,
            "velocity_alpha": 0.6,
        }
        if "track" in cfg:
            for k, v in dict(cfg.track).items():
                tracker_cfg[k] = v
    else:
        tracker_cfg = None

    torch_model = Torch_model(
        model_name=cfg.model_name,
        model_path=Path(cfg.train.path_to_save) / "model.pt",
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=cfg.train.conf_thresh,
        rect=cfg.export.dynamic_input,
        channels=cfg.train.in_channels,
        task=cfg.task,
        return_probs=cfg.infer.slice_inference
    )

    if data_type == "video" and cfg.train.in_channels != 3:
        raise ValueError(
            f"Video inference only supports 3-channel input, got in_channels={cfg.train.in_channels}"
        )

    output_path = Path(cfg.train.infer_path)
    if output_path.exists():
        rmtree(output_path)

    if data_type == "image":
        if cfg.task == "sem_seg":
            run_images_sem_seg(
                torch_model, folder_path, output_path, label_to_name=cfg.train.label_to_name, use_instance_segmentation_dataset=cfg.train.instance_segmentation_dataset, slice_inference=cfg.infer.get("slice_inference"), tile_size=cfg.infer.get("tile_size"), tile_overlap=cfg.infer.get("tile_overlap")
            )
        else:
            run_images(
                torch_model,
                folder_path,
                output_path,
                label_to_name=cfg.train.label_to_name,
                to_crop=to_crop,
                paddings=paddings,
                conf_thresh=cfg.train.conf_thresh,
            )
    elif data_type == "video":
        if cfg.task == "sem_seg":
            run_videos_sem_seg(
                torch_model, folder_path, output_path, label_to_name=cfg.train.label_to_name
            )
        elif use_tracking:
            run_videos_tracked(
                torch_model,
                folder_path,
                output_path,
                label_to_name=cfg.train.label_to_name,
                tracker_cfg=tracker_cfg,
            )
        else:
            run_videos(
                torch_model,
                folder_path,
                output_path,
                label_to_name=cfg.train.label_to_name,
                to_crop=to_crop,
                paddings=paddings,
                conf_thresh=cfg.train.conf_thresh,
            )


if __name__ == "__main__":
    main()
