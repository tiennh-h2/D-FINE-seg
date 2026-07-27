from pathlib import Path
from shutil import rmtree

import cv2
import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.dl.train import Trainer
from src.dl.dataset import Loader, read_image_hwc
from src.dl.utils import (
    Visualizer,
    abs_xyxy_to_norm_xywh,
    get_latest_experiment_name,
    overlay_sem_seg,
    sem_seg_palette,
)
from src.infer.torch_model import Torch_model

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def load_best_weights(trainer: Trainer, model_path: Path) -> None:
    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    # evaluate_sem_seg() prefers the EMA model when EMA is enabled.
    eval_model = (
        trainer.ema_model.model
        if trainer.ema_model is not None
        else trainer.model
    )

    if isinstance(eval_model, DDP):
        eval_model = eval_model.module

    incompatible = eval_model.load_state_dict(
        state_dict,
        strict=False,
    )

    logger.info(f"Loaded evaluation weights from: {model_path}")
    logger.info(f"Missing keys: {incompatible.missing_keys}")
    logger.info(f"Unexpected keys: {incompatible.unexpected_keys}")


def evaluate_trainer(cfg: DictConfig, base_loader, trainer):
    """Build non-distributed evaluation loaders and evaluate the validation set."""
    _, _, test_loader_eval = base_loader.build_dataloaders(
        distributed=False
    )
    trainer.test_loader = test_loader_eval
    trainer.distributed = False  # turn off DDP inside evaluate

    test_metrics = trainer.evaluate(
        val_loader=trainer.test_loader,
        conf_thresh=trainer.conf_thresh,
        iou_thresh=trainer.iou_thresh,
        path_to_save=Path(cfg.train.path_to_save),
        extended=True,
        mode="test",
    )
    logger.info(f"Validation metrics: {test_metrics}")
    return test_metrics


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


def run_images_sem_seg(torch_model, folder_path, output_path, label_to_name, use_instance_segmentation_dataset:bool=False):
    """Overlay + raw label-map PNG per image; crops/YOLO txt are box-based -> skipped."""
    palette = sem_seg_palette(len(label_to_name))
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)
    labels = set()
    img_paths = [img.name for img in folder_path.iterdir() if not img.name.startswith(".")]
    for img_path in tqdm(img_paths):
        img = read_image_hwc(folder_path / img_path)
        if img is None:
            logger.warning(f"Skipping unreadable image: {img_path}")
            continue
        is_npy = Path(img_path).suffix.lower() == ".npy"
        label_map = torch_model(img, bgr=not is_npy)[0]["sem_seg"].cpu().numpy()

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
        # GT-style output: grayscale PNG, pixel value = class id
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

    with open(output_path / "labels.txt", "w") as f:
        for class_id in sorted(labels):
            f.write(f"{label_to_name[int(class_id)]}\n")


def run(cfg: DictConfig, base_loader=None, trainer=None):
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
    )

    if data_type == "video" and cfg.train.in_channels != 3:
        raise ValueError(
            f"Video inference only supports 3-channel input, got in_channels={cfg.train.in_channels}"
        )

    val_metrics = None
    if base_loader is not None or trainer is not None:
        if base_loader is None or trainer is None:
            raise ValueError(
                "base_loader and trainer must either both be provided or both be omitted"
            )
        val_metrics = evaluate_trainer(cfg, base_loader, trainer)

    output_path = Path(cfg.train.infer_path)
    if output_path.exists():
        rmtree(output_path)

    if data_type == "image":
        if cfg.task == "sem_seg":
            run_images_sem_seg(
                torch_model, folder_path, output_path, label_to_name=cfg.train.label_to_name, use_instance_segmentation_dataset=cfg.train.instance_segmentation_dataset
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

    return val_metrics

@hydra.main(
    version_base=None,
    config_path="../../",
    config_name="config_size_l_1600",
)
def main(cfg: DictConfig):
    # Resolve the experiment before constructing Trainer.
    cfg.exp = get_latest_experiment_name(
        cfg.exp,
        cfg.train.path_to_save,
    )
    OmegaConf.set_struct(cfg.train, False)
    cfg.train.disable_log = True

    trainer = Trainer(cfg)

    load_best_weights(
        trainer,
        Path(cfg.train.path_to_save) / "model.pt",
    )

    metrics = trainer.evaluate(
        val_loader=trainer.test_loader,
        conf_thresh=trainer.conf_thresh,
        iou_thresh=trainer.iou_thresh,
        path_to_save=Path(cfg.train.path_to_save),
        extended=True,
        mode="test",
    )

    logger.info(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()
