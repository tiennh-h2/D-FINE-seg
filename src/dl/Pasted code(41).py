def _axis_slice_starts(
    length: int,
    slice_size: int,
    overlap_ratio: float,
) -> list[int]:
    """Return gap-free SAHI-style slice starts for one image axis."""
    if length <= 0:
        raise ValueError(f"Image axis length must be positive, got {length}")
    if slice_size <= 0:
        raise ValueError(f"SAHI slice size must be positive, got {slice_size}")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(
            f"SAHI overlap ratio must be in [0, 1), got {overlap_ratio}"
        )

    max_start = max(length - slice_size, 0)
    step = max(int(slice_size * (1.0 - overlap_ratio)), 1)
    starts = list(range(0, max_start + 1, step))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def _build_sahi_crop_boxes(
    image_h: int,
    image_w: int,
    slice_h: int,
    slice_w: int,
    overlap_h: float,
    overlap_w: float,
) -> list[tuple[int, int, int, int]]:
    """Build `(x1, y1, x2, y2)` crops covering an image without duplicates."""
    y_starts = _axis_slice_starts(image_h, slice_h, overlap_h)
    x_starts = _axis_slice_starts(image_w, slice_w, overlap_w)
    return [
        (x1, y1, min(x1 + slice_w, image_w), min(y1 + slice_h, image_h))
        for y1 in y_starts
        for x1 in x_starts
    ]


class SemSegDataset(Dataset):
    """Dense per-pixel labels for task=sem_seg.

    Layout: images/<stem>.<ext> + labels/<stem>.png (single channel uint8,
    pixel value = class id; ignore_index excluded from loss and metrics).
    Masks always use NEAREST interpolation (LINEAR corrupts integer class ids)
    and every pad-introducing aug fills the mask with ignore_index.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],  # h, w
        root_path: Path,
        split: pd.DataFrame,
        debug_img_processing: bool,
        mode: str,
        cfg: DictConfig,
    ) -> None:
        self.root_path = root_path
        self.instance_segmentation_dataset = cfg.train.instance_segmentation_dataset
        self.split = split
        self.target_h, self.target_w = img_size
        self.in_channels = int(cfg.train.in_channels)
        self.norm = ([0.0] * self.in_channels, [1.0] * self.in_channels)
        self.debug_img_processing = debug_img_processing
        self.debug_img_path = Path(cfg.train.debug_img_path)
        self.mode = mode
        self.label_to_name = cfg.train.label_to_name
        self.ignore_index = int(cfg.train.sem_seg.ignore_index)
        self.cases_to_debug = 20
        # dense-mask mosaic shares the box path's mosaic_augs knobs (affine warps img+mask)
        self.mosaic_scale = cfg.train.mosaic_augs.mosaic_scale
        self.degrees = cfg.train.mosaic_augs.degrees
        self.translate = cfg.train.mosaic_augs.translate
        self.shear = cfg.train.mosaic_augs.shear
        # shared-memory so close_mosaic() reaches persistent workers (mirrors CustomDataset)
        self._shared_flags = torch.zeros(1).share_memory_()
        self.mosaic_prob = resolve_mosaic_prob(cfg) if mode == "train" else 0.0
        self.ignore_background = False
        self.keep_ratio = cfg.train.keep_ratio
        self._init_sahi(cfg)
        self._init_augs(cfg)

    @property
    def mosaic_prob(self) -> float:
        return float(self._shared_flags[0])

    @mosaic_prob.setter
    def mosaic_prob(self, value: float) -> None:
        self._shared_flags[0] = float(value)

    @staticmethod
    def _validate_image_mask_dimensions(image, mask, source_idx: int) -> None:
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                "Image and mask dimensions must match for source index "
                f"{source_idx}, got image={image.shape[:2]} and mask={mask.shape[:2]}"
            )

    def _init_sahi(self, cfg) -> None:
        """Build a virtual index containing one entry per SAHI crop."""
        sahi_cfg = getattr(cfg.train, "sahi", None)
        self.sahi_enabled = (
            self.mode == "train"
            and sahi_cfg is not None
            and bool(getattr(sahi_cfg, "enabled", False))
        )
        self._sample_index = []

        if not self.sahi_enabled:
            self._sample_index.extend((source_idx, None) for source_idx in range(len(self.split)))
            return

        self.sahi_slice_height = int(sahi_cfg.slice_height)
        self.sahi_slice_width = int(sahi_cfg.slice_width)
        self.sahi_overlap_height_ratio = float(sahi_cfg.overlap_height_ratio)
        self.sahi_overlap_width_ratio = float(sahi_cfg.overlap_width_ratio)
        if self.sahi_slice_height <= 0 or self.sahi_slice_width <= 0:
            raise ValueError(
                "SAHI slice size must be positive, got "
                f"{(self.sahi_slice_height, self.sahi_slice_width)}"
            )
        if not (
            0.0 <= self.sahi_overlap_height_ratio < 1.0
            and 0.0 <= self.sahi_overlap_width_ratio < 1.0
        ):
            raise ValueError(
                "SAHI overlap ratios must be in [0, 1), got "
                f"{(self.sahi_overlap_height_ratio, self.sahi_overlap_width_ratio)}"
            )

        for source_idx in range(len(self.split)):
            loaded = self._load_image_mask(source_idx)
            if loaded is None:
                raise ValueError(
                    f"Cannot build SAHI crops: source index {source_idx} is unreadable"
                )
            image, mask = loaded
            self._validate_image_mask_dimensions(image, mask, source_idx)
            crop_boxes = _build_sahi_crop_boxes(
                image_h=image.shape[0],
                image_w=image.shape[1],
                slice_h=self.sahi_slice_height,
                slice_w=self.sahi_slice_width,
                overlap_h=self.sahi_overlap_height_ratio,
                overlap_w=self.sahi_overlap_width_ratio,
            )
            self._sample_index.extend((source_idx, crop_box) for crop_box in crop_boxes)

    def _load_sample(self, idx: int):
        """Load one full source pair or one aligned SAHI image/mask crop."""
        source_idx, crop_box = self._sample_index[idx]
        loaded = self._load_image_mask(source_idx)
        if loaded is None:
            return None
        image, mask = loaded
        self._validate_image_mask_dimensions(image, mask, source_idx)
        if crop_box is None:
            return image, mask

        x1, y1, x2, y2 = crop_box
        image_h, image_w = image.shape[:2]
        if x2 > image_w or y2 > image_h:
            raise ValueError(
                f"SAHI crop {crop_box} exceeds image dimensions {(image_h, image_w)} "
                f"for source index {source_idx}"
            )
        return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def _init_augs(self, cfg) -> None:
        pad_color = tuple([114] * self.in_channels)
        if self.keep_ratio:
            # letterbox img (LINEAR/114) + dense mask (NEAREST/ignore_index); scaleup=True to match
            # the /infer letterbox() default so train-eval mIoU == bench mIoU
            resize = [
                LetterboxRect(
                    height=self.target_h,
                    width=self.target_w,
                    color=pad_color,
                    scaleup=True,
                    dense_mask=True,
                    mask_fill=self.ignore_index,
                    always_apply=True,
                )
            ]
        else:
            resize = [A.Resize(self.target_h, self.target_w, interpolation=cv2.INTER_LINEAR)]
        norm = [A.Normalize(mean=self.norm[0], std=self.norm[1]), ToTensorV2()]

        if self.mode == "train":
            augs = [
                A.CoarseDropout(
                    num_holes_range=(1, 2),
                    hole_height_range=(0.05, 0.15),
                    hole_width_range=(0.05, 0.15),
                    fill_mask=self.ignore_index,  # don't supervise classes under occluders
                    p=cfg.train.augs.coarse_dropout,
                ),
                A.RandomBrightnessContrast(p=cfg.train.augs.brightness),
                A.RandomGamma(p=cfg.train.augs.gamma),
                A.Blur(p=cfg.train.augs.blur),
                A.GaussNoise(p=cfg.train.augs.noise, std_range=(0.1, 0.2)),
                A.Affine(rotate=[90, 90], p=cfg.train.augs.rotate_90, fit_output=True),
                A.HorizontalFlip(p=cfg.train.augs.left_right_flip),
                A.VerticalFlip(p=cfg.train.augs.up_down_flip),
                A.Rotate(
                    limit=cfg.train.augs.rotation_degree,
                    p=cfg.train.augs.rotation_p,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=pad_color,
                    fill_mask=self.ignore_index,
                ),
            ]
            if self.in_channels == 3:
                augs.insert(5, A.ToGray(p=cfg.train.augs.to_gray))

            # scale-jitter + crop after the deployment-identical resize (mosaic alternative)
            jitter = cfg.train.augs.scale_jitter
            post_resize = []
            if jitter:
                lo, hi = float(jitter[0]), float(jitter[1])
                post_resize = [
                    A.RandomScale(scale_limit=(lo - 1.0, hi - 1.0), p=1.0),
                    A.PadIfNeeded(
                        min_height=self.target_h,
                        min_width=self.target_w,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=pad_color,
                        fill_mask=self.ignore_index,
                        position="random",
                    ),
                    A.RandomCrop(self.target_h, self.target_w),
                ]
            self.transform = A.Compose(
                augs + resize + post_resize + norm, mask_interpolation=cv2.INTER_NEAREST
            )
            # mosaic already emits a target-size affine crop -> augs + resize (snaps any aug size
            # drift back to target, e.g. Affine fit_output) + norm; skips scale_jitter (affine's job)
            self.mosaic_transform = A.Compose(
                augs + resize + norm, mask_interpolation=cv2.INTER_NEAREST
            )
        elif self.mode in ["val", "test", "bench"]:
            self.transform = A.Compose(resize + norm, mask_interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError(
                f"Unknown mode: {self.mode}, choose from ['train', 'val', 'test', 'bench']"
            )

    def _debug_image(self, idx, image: torch.Tensor, sem_mask: torch.Tensor, img_path: Path):
        mean = np.array(self.norm[0]).reshape(-1, 1, 1)
        std = np.array(self.norm[1]).reshape(-1, 1, 1)
        image_np = image.cpu().numpy() * std + mean
        image_np = np.transpose(image_np, (1, 2, 0))[:, :, :3]
        image_np = np.ascontiguousarray(np.clip(image_np * 255.0, 0, 255).astype(np.uint8))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        palette = sem_seg_palette(len(self.label_to_name))
        image_np = overlay_sem_seg(
            image_np,
            sem_mask.cpu().numpy().astype(np.uint8),
            palette,
            ignore_index=self.ignore_index,
        )
        save_dir = self.debug_img_path / self.mode
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"{idx}_idx_{img_path.stem}_debug.jpg"), image_np)

    def _load_image_mask(self, idx: int):
        """Load one (image HWC, mask HW) pair at native resolution, or None if unreadable."""
        if self.instance_segmentation_dataset:
            full_path = Path(self.split[idx][0])
        else:
            image_path = Path(self.split.iloc[idx].values[0])
            full_path = self.root_path / "images" / f"{image_path}"
        try:
            image = read_image_rgb(full_path, self.in_channels)
        except ValueError as e:
            logger.warning(f"Skipping {full_path}: {e}")
            return None
        if image is None:
            return None
        if self.instance_segmentation_dataset:
            mask_path = Path(self.split[idx][1])
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) // 255
            # mask = 255 - mask
        else:
            mask_path = self.root_path / "labels" / f"{image_path.stem}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"Skipping {full_path}: can't read mask")
            return None
        invalid = (mask >= len(self.label_to_name)) & (mask != self.ignore_index)
        if invalid.any():
            raise ValueError(
                f"{mask_path}: class id {int(mask[invalid][0])} >= num_classes="
                f"{len(self.label_to_name)} (ignore_index={self.ignore_index}); "
                "masks must use contiguous label_to_name ids"
            )
        return image, mask

    def _load_mosaic(self, idx: int):
        """4-image mosaic for dense masks: mirrors the box-path mosaic but tiles the label map
        alongside the image (NEAREST, ignore_index fill) — no polygons needed since the mask is
        just a second image plane. Tiles into a 2H x 2W canvas at a jittered junction, then affine-
        warps a target-size window out of it (scale/translate/shear from mosaic_augs; image
        LINEAR/114, mask NEAREST/ignore_index) — the scale jitter + crop the box path gets."""
        H, W = self.target_h, self.target_w
        yc = int(random.uniform(H * 0.6, H * 1.4))
        xc = int(random.uniform(W * 0.6, W * 1.4))
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        img4 = np.full((H * 2, W * 2, self.in_channels), 114, dtype=np.uint8)
        mask4 = np.full((H * 2, W * 2), self.ignore_index, dtype=np.uint8)
        for i, m_idx in enumerate(indices):
            r, retries = self._load_sample(m_idx), 0
            while r is None and retries < 3:
                r = self._load_sample(random.randint(0, len(self) - 1))
                retries += 1
            if r is None:
                return None
            img, mask = r
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            (lx1, ly1, lx2, ly2), (sx1, sy1, sx2, sy2) = get_mosaic_coordinate(
                img4, i, xc, yc, W, H, H, W
            )
            img4[ly1:ly2, lx1:lx2] = img[sy1:sy2, sx1:sx2]
            mask4[ly1:ly2, lx1:lx2] = mask[sy1:sy2, sx1:sx2]
        M, _ = get_transform_matrix(
            img4.shape[:2], (W, H), self.degrees, self.mosaic_scale, self.shear, self.translate
        )
        img4 = cv2.warpAffine(
            img4,
            M[:2],
            dsize=(W, H),
            flags=cv2.INTER_LINEAR,
            borderValue=tuple([255] * self.in_channels),
        )
        mask4 = cv2.warpAffine(
            mask4, M[:2], dsize=(W, H), flags=cv2.INTER_NEAREST, borderValue=0 if self.instance_segmentation_dataset else self.ignore_index
        )
        return img4, mask4

    def _concat_pad(self, tiles, axis, rotate_p=0.3):
        """Concatenate (img, mask) tiles along axis (0=vertical stack, 1=horizontal stack),
        padding the non-concat dimension with 114/ignore_index so nothing gets cropped.
        Each tile is independently rotated 90 CW/CCW with probability rotate_p before
        concatenation (rotation applied before padding, so it affects alignment too)."""
        rotated = []
        for img, mask in tiles:
            if random.random() < rotate_p:
                k = random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE])
                img = cv2.rotate(img, k)
                mask = cv2.rotate(mask, k)
            rotated.append((img, mask))
        tiles = rotated

        other_axis = 1 - axis
        max_dim = max(t[0].shape[other_axis] for t in tiles)

        padded_imgs, padded_masks = [], []
        for img, mask in tiles:
            pad = max_dim - img.shape[other_axis]
            if pad > 0:
                if axis == 0:  # stacking rows -> pad width (right side)
                    border = (0, 0, 0, pad)
                else:  # stacking cols -> pad height (bottom side)
                    border = (0, pad, 0, 0)
                img = cv2.copyMakeBorder(
                    img, *border, cv2.BORDER_CONSTANT, value=[255] * self.in_channels
                )
                mask = cv2.copyMakeBorder(
                    mask, *border, cv2.BORDER_CONSTANT, value=0 if self.instance_segmentation_dataset else self.ignore_index
                )
            padded_imgs.append(img)
            padded_masks.append(mask)

        img_cat = np.concatenate(padded_imgs, axis=axis)
        mask_cat = np.concatenate(padded_masks, axis=axis)
        return img_cat, mask_cat


    def _concat_images(self, idx: int):
        """Crop-free mosaic: picks 1-3 plans and lays them out fully intact — vertical stack,
        horizontal stack, or a mixed grid (one big tile beside two stacked smaller ones) for 3
        plans. Mismatched edges are padded with 114/ignore_index, never cropped. The combined
        canvas is resized once (LINEAR for image, NEAREST for mask) to the target size — the
        only place any content gets shrunk, and nothing is ever cut off."""
        n = random.choices([2, 3, 4])[0]
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(n - 1)]

        tiles = []
        for m_idx in indices:
            r, retries = self._load_sample(m_idx), 0
            while r is None and retries < 3:
                r = self._load_sample(random.randint(0, len(self) - 1))
                retries += 1
            if r is None:
                return None
            tiles.append(r)

        if n == 1:
            img, mask = tiles[0]

        elif n == 2:
            axis = random.choice([0, 1])  # 0 = vertical stack, 1 = horizontal stack
            if random.random() < 0.5:
                tiles = tiles[::-1]
            img, mask = self._concat_pad(tiles, axis)

        else:  # n == 3
            layout = random.choice(["vertical", "horizontal", "mixed"])
            random.shuffle(tiles)

            if layout == "vertical":
                img, mask = self._concat_pad(tiles, axis=0)
            elif layout == "horizontal":
                img, mask = self._concat_pad(tiles, axis=1)
            else:  # mixed: one big tile beside two stacked smaller ones
                big, *rest = tiles
                inner_axis = random.choice([0, 1])       # how the two small tiles stack
                outer_axis = 1 - inner_axis               # how big joins the stacked pair
                sub_img, sub_mask = self._concat_pad(rest, axis=inner_axis)
                pieces = [big, (sub_img, sub_mask)]
                if random.random() < 0.5:
                    pieces = pieces[::-1]
                img, mask = self._concat_pad(pieces, axis=outer_axis)

        img = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)
        return img, mask


    def close_mosaic(self):
        self.mosaic_prob = 0.0
        logger.info("Closing mosaic")

    def __getitem__(self, idx: int):
        """returns (image CHW float, sem_mask (H,W) long, image_path, orig_size (H,W))"""
        source_idx, _ = self._sample_index[idx]
        image_path = (
            Path(self.split[source_idx][0])
            if self.instance_segmentation_dataset
            else Path(self.split.iloc[source_idx].values[0])
        )
        if self.mosaic_prob and random.random() < self.mosaic_prob:
            if random.random() < 0.5:
                mosaic = self._load_mosaic(idx)
            else:
                mosaic = self._concat_images(idx)
            if mosaic is None:
                return None
            image, sem_mask = mosaic
            orig_size = torch.tensor([self.target_h, self.target_w])  # train-only; orig res unused
            transformed = self.mosaic_transform(image=image, mask=sem_mask)  # already target-size
        else:
            r = self._load_sample(idx)
            if r is None:
                return None
            image, sem_mask = r
            orig_size = torch.tensor(image.shape[:2])
            transformed = self.transform(image=image, mask=sem_mask)

        image_t = transformed["image"]
        sem_mask_t = transformed["mask"].long()

        if self.debug_img_processing and idx <= self.cases_to_debug:
            self._debug_image(idx, image_t, sem_mask_t, image_path)
        return image_t, sem_mask_t, image_path, orig_size

    def __len__(self):
        return len(self._sample_index)
