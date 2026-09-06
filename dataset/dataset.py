import numpy as np
import pandas as pd
import torch
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def load_wm811k_pickles(labeled_path, unlabeled_path):
    labeled_df = pd.read_pickle(labeled_path)
    unlabeled_df = pd.read_pickle(unlabeled_path)
    return labeled_df, unlabeled_df


class WaferMapToOneHotTensor:

    def __init__(self, num_classes=3):
        self.num_classes = num_classes

    def __call__(self, img):
        arr = np.asarray(img, dtype=np.int64)
        arr = np.clip(arr, 0, self.num_classes - 1)
        tensor = torch.from_numpy(arr)
        one_hot = torch.nn.functional.one_hot(tensor, num_classes=self.num_classes)
        return one_hot.permute(2, 0, 1).to(dtype=torch.float32)


class Cutout:
    def __init__(self, num_holes=4, cut_ratio=0.2, fill=0):
        self.num_holes = num_holes
        self.cut_ratio = cut_ratio
        self.fill = fill

    def __call__(self, img):
        arr = np.asarray(img, dtype=np.uint8).copy()
        h, w = arr.shape[:2]
        if h == 0 or w == 0:
            return img

        cut_size = max(1, int(min(h, w) * self.cut_ratio))
        half = cut_size // 2

        for _ in range(max(0, self.num_holes)):
            cy = np.random.randint(0, h)
            cx = np.random.randint(0, w)

            y1 = max(0, cy - half)
            y2 = min(h, cy + half)
            x1 = max(0, cx - half)
            x2 = min(w, cx + half)
            arr[y1:y2, x1:x2] = self.fill

        return Image.fromarray(arr)


class MaskedBernoulliNoiseFast:
    def __init__(self, noise_prob=0.05, min_=0, max_=1):
        self.noise_prob = noise_prob
        self.min_ = min_
        self.max_ = max_

    def __call__(self, img):
        arr = np.asarray(img, dtype=np.uint8)
        h, w = arr.shape[:2]
        if h == 0 or w == 0:
            return img

        # Apply noise only where wafer exists (x > 0).
        wafer_mask = arr > 0
        bernoulli_mask = np.random.rand(h, w) < self.noise_prob
        mask = wafer_mask & bernoulli_mask

        out = arr.copy()
        noise_value = 1 + np.random.randint(self.min_, self.max_ + 1, size=(h, w))
        out[mask] = noise_value[mask].astype(np.uint8)
        return Image.fromarray(out)


def stratified_split(labels, train_ratio=0.75, val_ratio=0.15, test_ratio=0.1, label_ratio=1.0, data_seed=42):
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), \
        "train_ratio + val_ratio + test_ratio must be 1.0"
    assert 0 < label_ratio <= 1.0, "label_ratio must be in (0, 1]"

    rng = np.random.RandomState(data_seed)
    train_idx, val_idx, test_idx = [], [], []

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))

        train_part = idx[:n_train]
        val_part = idx[n_train:n_train + n_val]
        test_part = idx[n_train + n_val:]

        # n(train data) * label_ratio
        n_labeled = int(np.floor(len(train_part) * label_ratio))
        labeled_train_part = train_part[:n_labeled]

        train_idx.extend(labeled_train_part)
        val_idx.extend(val_part)
        test_idx.extend(test_part)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


class TransformFixMatch:
    def __init__(self, image_size=128, num_holes=4, cut_ratio=0.2, noise_prob=0.05):
        self.weak = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(180, interpolation=InterpolationMode.NEAREST),
        ])

        self.strong = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            Cutout(num_holes=num_holes, cut_ratio=cut_ratio, fill=0),
            MaskedBernoulliNoiseFast(noise_prob=noise_prob, min_=0, max_=1),
        ])

        self.normalize = transforms.Compose([
            WaferMapToOneHotTensor(),
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class WM811KSSL(Dataset):
    def __init__(self, df, idx=None, transform=None):
        if idx is not None:
            self.df = df.iloc[idx].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        self.images = self.df["waferMap"].tolist()
        labels = self.df["label"]
        if pd.api.types.is_numeric_dtype(labels):
            self.labels = labels.fillna(-1).to_numpy(dtype=np.int64)
        else:
            self.labels = pd.to_numeric(labels, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_wm811k(labeled_path, unlabeled_path, 
               train_ratio=0.75,val_ratio=0.15, test_ratio=0.1, label_ratio=0.5,image_size=128, 
               cutout_num_holes=4, cutout_ratio=0.2, noise_prob=0.05,
               data_seed=42):

    transform_labeled = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
        transforms.RandomRotation(180, interpolation=InterpolationMode.NEAREST),
        WaferMapToOneHotTensor(),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
        WaferMapToOneHotTensor(),
    ])

    labeled_df, unlabeled_df = load_wm811k_pickles(labeled_path, unlabeled_path)
    labels = labeled_df["label"].to_numpy(dtype=np.int64)

    train_labeled_idxs, val_labeled_idxs, test_labeled_idxs = stratified_split(
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        label_ratio=label_ratio,
        data_seed=data_seed,
    )

    train_labeled_dataset = WM811KSSL(
        df=labeled_df,
        idx=train_labeled_idxs,
        transform=transform_labeled,
    )

    train_unlabeled_dataset = WM811KSSL(
        df=unlabeled_df,
        idx=None,
        transform=TransformFixMatch(
            image_size=image_size,
            num_holes=cutout_num_holes,
            cut_ratio=cutout_ratio,
            noise_prob=noise_prob,
        ),
    )

    test_dataset = WM811KSSL(
        df=labeled_df,
        idx=test_labeled_idxs,
        transform=transform_eval,
    )

    val_dataset = WM811KSSL(
        df=labeled_df,
        idx=val_labeled_idxs,
        transform=transform_eval,
    )

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


def get_wm811k_loaders(train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, batch_size, mu):
    labeled_trainloader = DataLoader(train_labeled_dataset, 
                                     sampler=RandomSampler(train_labeled_dataset),
                                     batch_size=batch_size,
                                     num_workers=2,
                                     pin_memory=True,
                                     persistent_workers=True,
                                     drop_last=False)
    unlabeled_trainloader = DataLoader(train_unlabeled_dataset, 
                                       sampler=RandomSampler(train_unlabeled_dataset),
                                       batch_size=batch_size * mu,
                                       num_workers=2,
                                       pin_memory=True,
                                       persistent_workers=True,
                                       drop_last=False)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size, drop_last=False)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size, drop_last=False)
    return labeled_trainloader, unlabeled_trainloader, val_loader, test_loader


def print_dataset_distribution(name, dataset):
    if "label" not in dataset.df.columns:
        print(f"\n[{name}]")
        print("Label column not found.")
        print(f"Total: {len(dataset.df)}")
        return

    counts = dataset.df["label"].value_counts().sort_index()
    total = len(dataset.df)

    idx_to_class = {
        0: "Center",
        1: "Donut",
        2: "Edge-Loc",
        3: "Edge-Ring",
        4: "Loc",
        5: "Random",
        6: "Scratch",
        7: "Near-Full",
        8: "none",
    }

    print(f"\n[{name}]")
    print(f"Total: {total}")

    for i in range(9):
        count = counts.get(i, 0)
        ratio = (count / total * 100) if total > 0 else 0.0
        print(f"{i} ({idx_to_class[i]}): {count} ({ratio:.2f}%)")


if __name__ == "__main__":
    train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset = get_wm811k(
        labeled_path="./data/wm811k/preprocessing/labeled.pkl",
        unlabeled_path="./data/wm811k/preprocessing/unlabeled.pkl",
        train_ratio=0.75,
        val_ratio=0.15,
        test_ratio=0.1,
        label_ratio=1.0,
        image_size=96,
        data_seed=0,
        cutout_num_holes=4,
        cutout_ratio=0.2,
        noise_prob=0.05,
    )

    print("Train labeled size:", len(train_labeled_dataset))
    print("Train unlabeled size:", len(train_unlabeled_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    print_dataset_distribution("Train labeled", train_labeled_dataset)
    print_dataset_distribution("Train unlabeled", train_unlabeled_dataset)
    print_dataset_distribution("Val", val_dataset)
    print_dataset_distribution("Test", test_dataset)
