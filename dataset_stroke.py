import os
import re
import numpy as np
import torch
from torch_geometric.data import Data

# keystroke feature와 iki를 저장한 npz 파일에서 그래프 데이터셋을 준비하는 코드
# 여기서 변환된 데이터는 train_node.py에서 DataLoader로 불러와 모델 학습에 사용됨

def _parse_label_from_filename(filename: str) -> str:
    """
    Extract label string from filename like: key_hello world.wav -> "hello world".
    Uses everything after 'key_' and before extension.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^key_(.*)$", name)
    if not m:
        raise ValueError(f"Filename does not start with 'key_': {filename}")
    return m.group(1)


def _build_label_map(label_strings):
    # Collect all unique characters across dataset
    chars = sorted({ch for s in label_strings for ch in s})
    label_map = {ch: i for i, ch in enumerate(chars)}
    return label_map


def _build_chain_edges(n: int, undirected: bool = True):
    """Create chain edges i->i+1 (and optionally i+1->i)."""
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)
    src = torch.arange(0, n - 1, dtype=torch.long)
    dst = torch.arange(1, n, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    if undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def prepare_keystroke_sequence_dataset(base_path: str, undirected: bool = True):
    """
    base_path: folder containing *.npz files per wav sequence.
    Each npz must contain:
      - x: [N, 768]
      - iki: [N-1]
    Labels are extracted from filename: key_<label>.npz
    Returns list[Data], label_map
    """
    files = sorted([f for f in os.listdir(base_path) if f.endswith('.npz')])
    if not files:
        raise FileNotFoundError(f"No .npz files found in {base_path}")

    label_strings = [_parse_label_from_filename(f) for f in files]
    label_map = _build_label_map(label_strings)

    all_data = []
    for f, label_str in zip(files, label_strings):
        path = os.path.join(base_path, f)
        data = np.load(path)
        x = data['x']  # [N, 768]
        iki = data['iki']  # [N-1]

        if x.ndim != 2:
            raise ValueError(f"x must be 2D [N, F], got {x.shape} in {f}")
        n = x.shape[0]

        # Map label string to per-node labels
        y_chars = list(label_str)
        if len(y_chars) != n:
            raise ValueError(
                f"Label length ({len(y_chars)}) != num nodes ({n}) in {f}"
            )
        y = torch.tensor([label_map[ch] for ch in y_chars], dtype=torch.long)

        edge_index = _build_chain_edges(n, undirected=undirected)

        # edge_attr: [E, 1]
        if n < 2:
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            iki_t = torch.tensor(iki, dtype=torch.float32).view(-1, 1)
            if undirected:
                edge_attr = torch.cat([iki_t, iki_t], dim=0)
            else:
                edge_attr = iki_t

        graph = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )
        all_data.append(graph)

    return all_data, label_map


if __name__ == "__main__":
    BASE_PATH = "./npz_sequences"
    SAVE_PATH = "./processed_dataset_stroke.pt"

    dataset, label_map = prepare_keystroke_sequence_dataset(BASE_PATH)
    torch.save({'dataset': dataset, 'label_map': label_map}, SAVE_PATH)
    print(f"Saved: {SAVE_PATH} (graphs={len(dataset)}, classes={len(label_map)})")
