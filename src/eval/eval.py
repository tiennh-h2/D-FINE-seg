import argparse
import os

import cv2
import numpy as np
import py_sod_metrics

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def safe_imread_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_to_gt_if_needed(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape != gt.shape:
        pred = cv2.resize(
            pred,
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return pred


def build_metric_pack():
    """
    Create a fresh metric pack.

    Important:
    Do NOT reuse the same global py_sod_metrics objects for multiple classes,
    otherwise the class results will be mixed together.
    """
    sample_gray = dict(with_adaptive=True, with_dynamic=True)

    metrics = {
        "FM": py_sod_metrics.Fmeasure(),
        "WFM": py_sod_metrics.WeightedFmeasure(),
        "SM": py_sod_metrics.Smeasure(),
        "EM": py_sod_metrics.Emeasure(),
        "MAE": py_sod_metrics.MAE(),
        "FMv2": py_sod_metrics.FmeasureV2(
            metric_handlers={
                "fm": py_sod_metrics.FmeasureHandler(
                    **sample_gray,
                    beta=0.3,
                ),
                "f1": py_sod_metrics.FmeasureHandler(
                    **sample_gray,
                    beta=1,
                ),
                "iou": py_sod_metrics.IOUHandler(**sample_gray),
                "dice": py_sod_metrics.DICEHandler(**sample_gray),
            }
        ),
    }

    return metrics


def step_metrics(metrics, pred: np.ndarray, gt: np.ndarray):
    """
    Update all metrics for one prediction / GT pair.

    pred: grayscale prediction, usually 0-255 probability map
    gt: grayscale binary GT, usually 0 or 255
    """
    pred = resize_to_gt_if_needed(pred, gt)

    # metrics["FM"].step(pred=pred, gt=gt)
    # metrics["WFM"].step(pred=pred, gt=gt)
    # metrics["SM"].step(pred=pred, gt=gt)
    # metrics["EM"].step(pred=pred, gt=gt)
    # metrics["MAE"].step(pred=pred, gt=gt)
    metrics["FMv2"].step(pred=pred, gt=gt)


def collect_results(metrics):
    fm = metrics["FM"].get_results()["fm"]
    wfm = metrics["WFM"].get_results()["wfm"]
    sm = metrics["SM"].get_results()["sm"]
    em = metrics["EM"].get_results()["em"]
    mae = metrics["MAE"].get_results()["mae"]
    fmv2 = metrics["FMv2"].get_results()

    results = {
        "meandice": float(fmv2["dice"]["dynamic"].mean()),
        "meaniou": float(fmv2["iou"]["dynamic"].mean()),
        "Smeasure": float(sm),
        "wFmeasure": float(wfm),
        "adpFm": float(fm["adp"]),
        "meanFm": float(fmv2["fm"]["dynamic"].mean()),
        "maxFm": float(fmv2["fm"]["dynamic"].max()),
        "meanEm": float(em["curve"].mean()),
        "MAE": float(mae),
    }

    return results


def print_results(title: str, results: dict, count: int):
    print()
    print("=" * 60)
    print(title)
    print(f"Images evaluated: {count}")
    print("-" * 60)
    print("mDice:          ", format(results["meandice"], ".3f"))
    print("mIoU:           ", format(results["meaniou"], ".3f"))
    # print("S_{alpha}:      ", format(results["Smeasure"], ".3f"))
    # print("F^{w}_{beta}:   ", format(results["wFmeasure"], ".3f"))
    # print("F_{beta}:       ", format(results["adpFm"], ".3f"))
    # print("F^{mean}_{beta}:", format(results["meanFm"], ".3f"))
    # print("F^{max}_{beta}: ", format(results["maxFm"], ".3f"))
    # print("E_{phi}:        ", format(results["meanEm"], ".3f"))
    # print("MAE:            ", format(results["MAE"], ".3f"))
    print("=" * 60)


def evaluate_file_pairs(file_pairs):
    metrics = build_metric_pack()

    for i, pair in enumerate(file_pairs):
        pred_path, gt_path = pair

        print(
            f"[{i}] Processing pred={os.path.basename(pred_path)} gt={os.path.basename(gt_path)}"
        )

        pred = safe_imread_gray(pred_path)
        gt = safe_imread_gray(gt_path)

        step_metrics(metrics, pred=pred, gt=gt)

    return collect_results(metrics), len(file_pairs)


def make_binary_pairs(pred_root: str, gt_root: str):
    pairs = []

    gt_names = sorted([f for f in os.listdir(gt_root) if is_image_file(f)])

    for gt_name in gt_names:
        stem, _ = os.path.splitext(gt_name)

        gt_path = os.path.join(gt_root, gt_name)
        pred_path = os.path.join(pred_root, stem + ".png")

        if not os.path.exists(pred_path):
            print(f"Warning: missing prediction, skipped: {pred_path}")
            continue

        pairs.append((pred_path, gt_path))

    return pairs


def make_multilabel_pairs_for_class(
    pred_root: str,
    gt_root: str,
    class_name: str,
):
    """
    Expected naming:

        pred_root/image001_house_boundary.png
        gt_root/image001_house_boundary.png

    where class_name = house_boundary.
    """
    pairs = []

    gt_names = sorted(
        [f for f in os.listdir(os.path.join(gt_root, class_name)) if is_image_file(f)]
    )

    for gt_name in gt_names:
        gt_stem, _ = os.path.splitext(gt_name)
        gt_path = os.path.join(gt_root, class_name, gt_name)
        pred_path = os.path.join(pred_root, f"{gt_stem}_{class_name}.png")

        if not os.path.exists(pred_path):
            pred_path = os.path.join(pred_root, class_name, f"{gt_stem}.png")
            if not os.path.exists(pred_path):
                print(
                    f"Warning: missing prediction for class '{class_name}', skipped: {pred_path}"
                )
                continue

        pairs.append((pred_path, gt_path))

    return pairs


def mean_results(result_list):
    if len(result_list) == 0:
        return None

    keys = result_list[0].keys()
    avg = {}

    for key in keys:
        avg[key] = float(np.mean([r[key] for r in result_list]))

    return avg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name shown in the result log",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Path to prediction masks",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        required=True,
        help="Path to ground truth masks",
    )
    parser.add_argument(
        "--class_names",
        nargs="+",
        type=str,
        default=None,
        help="List of class names for multilabel evaluation, e.g. --class_names house_boundary roof",
    )

    args = parser.parse_args()

    pred_root = args.pred_path
    gt_root = args.gt_path

    if args.class_names is None:
        print(f"Evaluating binary dataset: {args.dataset_name}")

        pairs = make_binary_pairs(pred_root, gt_root)

        if len(pairs) == 0:
            raise RuntimeError("No valid binary prediction / GT pairs found.")

        results, count = evaluate_file_pairs(pairs)
        print_results(args.dataset_name, results, count)

    else:
        print(f"Evaluating multilabel dataset: {args.dataset_name}")
        print(f"Classes: {args.class_names}")

        all_class_results = []
        all_class_counts = {}

        for class_name in args.class_names:
            print()
            print(f"Evaluating class: {class_name}")

            pairs = make_multilabel_pairs_for_class(
                pred_root=pred_root,
                gt_root=gt_root,
                class_name=class_name,
            )

            if len(pairs) == 0:
                print(f"Warning: no valid pairs found for class '{class_name}', skipped.")
                continue

            class_results, count = evaluate_file_pairs(pairs)

            all_class_results.append(class_results)
            all_class_counts[class_name] = count

            print_results(
                title=f"{args.dataset_name} / class: {class_name}",
                results=class_results,
                count=count,
            )

        if len(all_class_results) == 0:
            raise RuntimeError("No valid multilabel prediction / GT pairs found.")

        macro_results = mean_results(all_class_results)

        print_results(
            title=f"{args.dataset_name} / macro average over classes",
            results=macro_results,
            count=sum(all_class_counts.values()),
        )


if __name__ == "__main__":
    main()
