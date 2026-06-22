from pathlib import Path
import csv

import numpy as np
import torch
import torch.utils.data as data

from utils import process_feat

torch.set_default_tensor_type("torch.FloatTensor")


class Dataset_combine(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.is_normal = is_normal
        self.test_mode = test_mode
        self.transform = transform
        self.num_segments = getattr(args, "num_segments", 32)

        if test_mode:
            self.rgb_list_file = "list/ucf-rgb-test.list"
            self.dvs_list_file = "list/ucf-dvs-test.list"


        else:
            self.rgb_list_file = "list/ucf-rgb.list"
            self.dvs_list_file = "list/ucf-dvs.list"

        default_select_dir = "event_saliency/event_select_22_54%"


        self.select_csv_dir = Path(
            getattr(args, "select_csv_dir", default_select_dir)
        ).expanduser().resolve()

        self._parse_list()
        self.num_frame = self.num_segments
        self.select_index = self._build_select_index()

    def _normalize_name(self, name: str) -> str:
        name = Path(name).stem

        if name.endswith("_i3d"):
            name = name[:-4]

        if name.endswith("_x264"):
            name = name[:-5]

        if "_x264_" in name:
            name = name.split("_x264_")[0]

        return name.lower()

    def _normalize_select_stem(self, stem: str) -> str:
        s = stem.lower()

        for suf in ["_labeled_select", "_select", "_labeled"]:
            if s.endswith(suf):
                s = s[:-len(suf)]
                break

        return s

    def _build_select_index(self):
        index = {}

        if not self.select_csv_dir.exists():
            print(f"[WARN] select_csv_dir 不存在: {self.select_csv_dir}")
            return index

        for csv_path in self.select_csv_dir.glob("*.csv"):
            index[self._normalize_select_stem(csv_path.stem)] = csv_path

        # print(f"[INFO] 构建 select_flag CSV 索引: {len(index)} 条目")
        return index

    def _parse_list(self):
        with open(self.rgb_list_file, "r") as f:
            self.list_rgb = f.readlines()

        with open(self.dvs_list_file, "r") as f:
            self.list_dvs = f.readlines()

        if not self.test_mode:
            if self.is_normal:
                self.list_rgb = self.list_rgb[810:]
                self.list_dvs = self.list_dvs[810:]
                print("[INFO] normal list for ucf (train)")
            else:
                self.list_rgb = self.list_rgb[:810]
                self.list_dvs = self.list_dvs[:810]
                print("[INFO] abnormal list for ucf (train)")

    def _load_select_flags(self, video_name: str, T: int):
        key = video_name.lower()

        if key not in self.select_index:
            raise FileNotFoundError(
                f"未在 select_csv_dir 中找到对应 CSV: video={video_name}, dir={self.select_csv_dir}"
            )

        csv_path = self.select_index[key]
        flags = np.zeros(T, dtype=np.int64)

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            if header is None:
                raise ValueError(f"{csv_path} 为空或没有表头")

            try:
                idx_frame = header.index("frame_idx")
                idx_flag = header.index("select_flag")
            except ValueError as e:
                raise ValueError(f"{csv_path} 缺少必要列 frame_idx/select_flag: {e}")

            for row in reader:
                if len(row) <= max(idx_frame, idx_flag):
                    continue

                try:
                    fi = int(row[idx_frame])
                    sf = int(row[idx_flag])
                except Exception:
                    continue

                if 0 <= fi < T:
                    flags[fi] = sf

        return flags

    def _linear_interpolate_sparse(self, features_sparse: np.ndarray, selected_indices):
        C, L, D = features_sparse.shape

        selected_indices = np.array(
            sorted(set(map(int, selected_indices))),
            dtype=np.int64
        )

        selected_indices = selected_indices[
            (selected_indices >= 0) & (selected_indices < L)
        ]

        if len(selected_indices) == 0:
            return features_sparse.astype(np.float32)

        if len(selected_indices) == 1:
            idx = selected_indices[0]
            return np.repeat(
                features_sparse[:, idx:idx + 1, :],
                L,
                axis=1
            ).astype(np.float32)

        full_idx = np.arange(L, dtype=np.float32)
        selected_idx_float = selected_indices.astype(np.float32)
        filled = np.zeros_like(features_sparse, dtype=np.float32)

        for c in range(C):
            for d in range(D):
                y = features_sparse[c, selected_indices, d]
                filled[c, :, d] = np.interp(full_idx, selected_idx_float, y)

        return filled.astype(np.float32)

    def _linear_fill_by_indices(self, original_features: np.ndarray, selected_indices):
        _, L, _ = original_features.shape

        selected_indices = np.array(
            sorted(set(map(int, selected_indices))),
            dtype=np.int64
        )

        selected_indices = selected_indices[
            (selected_indices >= 0) & (selected_indices < L)
        ]

        if len(selected_indices) == 0:
            return original_features.astype(np.float32)

        sparse = np.zeros_like(original_features, dtype=np.float32)
        sparse[:, selected_indices, :] = original_features[:, selected_indices, :]

        return self._linear_interpolate_sparse(sparse, selected_indices)

    def _apply_select_flags_to_rgb(self, features_rgb: np.ndarray, select_flags: np.ndarray):
        _, T, _ = features_rgb.shape

        if select_flags is None or len(select_flags) == 0:
            return features_rgb.astype(np.float32)

        valid_len = min(T, len(select_flags))
        selected_indices = np.where(
            select_flags[:valid_len].astype(np.int64) == 1
        )[0]

        if len(selected_indices) == 0:
            return features_rgb.astype(np.float32)

        return self._linear_fill_by_indices(features_rgb, selected_indices)

    def __getitem__(self, index):
        label = self.get_label()

        rgb_path = self.list_rgb[index].strip()
        dvs_path = self.list_dvs[index].strip()

        features_rgb = np.load(rgb_path, allow_pickle=True).astype(np.float32)
        features_dvs = np.load(dvs_path, allow_pickle=True).astype(np.float32)

        video_name = self._normalize_name(rgb_path)

        features_rgb = features_rgb.transpose(1, 0, 2)
        features_dvs = features_dvs.transpose(1, 0, 2)

        _, T, _ = features_rgb.shape

        try:
            select_flags = self._load_select_flags(video_name, T)
            features_rgb = self._apply_select_flags_to_rgb(features_rgb, select_flags)
        except Exception as e:
            stage = "测试阶段" if self.test_mode else "训练阶段"
            print(
                f"[WARN] {stage}: {video_name} 使用原始 RGB 特征"
                f"（select_flag 加载失败: {e})"
            )

        if self.test_mode:
            features = np.concatenate((features_rgb, features_dvs), axis=-1)

            if self.transform is not None:
                features = self.transform(features)

            return features

        divided_rgb = []
        divided_dvs = []

        for i in range(features_rgb.shape[0]):
            divided_rgb.append(process_feat(features_rgb[i], self.num_segments))
            divided_dvs.append(process_feat(features_dvs[i], self.num_segments))

        divided_rgb = np.array(divided_rgb, dtype=np.float32)
        divided_dvs = np.array(divided_dvs, dtype=np.float32)

        features = np.concatenate((divided_rgb, divided_dvs), axis=-1)

        if self.transform is not None:
            features = self.transform(features)

        return features, label

    def get_label(self):
        return torch.tensor(0.0 if self.is_normal else 1.0)

    def __len__(self):
        return len(self.list_rgb)

    def get_num_frames(self):
        return self.num_frame