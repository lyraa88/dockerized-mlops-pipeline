import os, io, json, time, cv2, numpy as np
from typing import Tuple, List
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Model ----------------
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.net(x)

# --------------- Transforms ---------------
def get_transforms():
    # (참고) 일반적인 torchvision 파이프라인
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

# --------------- Dirs & IO ---------------
def ensure_dirs(share_dir: str):
    for sub in ["backupdata", "data", "model", "metrics", "logs"]:
        os.makedirs(os.path.join(share_dir, sub), exist_ok=True)
    for d in range(10):
        os.makedirs(os.path.join(share_dir, "backupdata", str(d)), exist_ok=True)
        os.makedirs(os.path.join(share_dir, "data", str(d)), exist_ok=True)

def save_model(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(path: str) -> CNN:
    m = CNN().to(DEVICE)
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m

# --------------- Train / Eval ---------------
@torch.no_grad()
def evaluate(model: nn.Module, loader) -> float:
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    return correct / max(total, 1)

def evaluate_with_loss(model: nn.Module, loader) -> Tuple[float, float]:
    crit = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = crit(logits, y)
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc

def train(model: nn.Module, train_loader, val_loader, epochs=2, lr=1e-3) -> Tuple[nn.Module, float]:
    model.train().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_acc, best_sd = 0.0, None
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        acc = evaluate(model.eval(), val_loader)
        if acc > best_acc:
            best_acc = acc
            best_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        model.train()
    if best_sd: model.load_state_dict(best_sd, strict=True)
    model.eval()
    return model, best_acc

# --------------- Metrics (clean+aug + loss) ---------------
def write_latest_metrics(share_dir: str, version: int,
                             acc_clean: float, acc_aug: float,
                             loss_clean: float, loss_aug: float):
    mdir = os.path.join(share_dir, "metrics")
    os.makedirs(mdir, exist_ok=True)
    latest = {
        "version": int(version),
        "accuracy_clean": round(float(acc_clean), 4),
        "accuracy_aug": round(float(acc_aug), 4),
        "loss_clean": round(float(loss_clean), 6),
        "loss_aug": round(float(loss_aug), 6),
        "ts": int(time.time())
    }
    with open(os.path.join(mdir, "latest.json"), "w") as f: json.dump(latest, f)
    hist = os.path.join(mdir, "history.csv")
    need = not os.path.exists(hist)
    with open(hist, "a") as f:
        if need:
            f.write("ts,version,accuracy_clean,accuracy_aug,loss_clean,loss_aug\n")
        f.write(f"{latest['ts']},{version},{latest['accuracy_clean']},{latest['accuracy_aug']},{latest['loss_clean']},{latest['loss_aug']}\n")

# --------------- Preprocess (photo → 28x28) ---------------
def preprocess_to_28x28(img_bytes: bytes) -> Image.Image:
    # 1) 로드 & 그레이스케일
    arr = np.frombuffer(img_bytes, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if im is None:
        im = np.array(Image.open(io.BytesIO(img_bytes)).convert("L"))
    # 2) 노이즈 약화 + 리사이즈
    im = cv2.GaussianBlur(im, (3, 3), 0)
    im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_AREA)
    # 3) 흰 배경일 때만 반전 (MNIST 극성으로 맞춤)
    if im.mean() > 128:
        im = 255 - im
    # 4) 대비 향상: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    im = clahe.apply(im)
    # 5) 다이내믹 레인지 스트레치
    im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(im.astype(np.uint8))

# --------------- Data utils ---------------
def count_data_samples(share_dir: str) -> int:
    total, base = 0, os.path.join(share_dir, "data")
    for d in range(10):
        p = os.path.join(base, str(d))
        if os.path.exists(p):
            total += sum(1 for x in os.listdir(p) if x.lower().endswith((".png", ".jpg", ".jpeg")))
    return total

def clear_data_folder(share_dir: str):
    base = os.path.join(share_dir, "data")
    for d in range(10):
        p = os.path.join(base, str(d))
        if os.path.exists(p):
            for f in os.listdir(p):
                fp = os.path.join(p, f)
                if os.path.isfile(fp): 
                    try: os.remove(fp)
                    except: pass

def persist_feedback_to_backup(share_dir: str):
    """재학습 트리거 시점에만 data/*를 backupdata/*로 누적"""
    import shutil
    for c in range(10):
        src = os.path.join(share_dir, "data", str(c))
        dst = os.path.join(share_dir, "backupdata", str(c))
        if not os.path.isdir(src): 
            continue
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp")):
                s = os.path.join(src, f)
                d = os.path.join(dst, f)
                if not os.path.exists(d):
                    try: shutil.copy2(s, d)
                    except: pass

# --------------- Version ---------------
def load_version(share_dir: str) -> int:
    p = os.path.join(share_dir, "model", "version.json")
    if not os.path.exists(p): return 0
    with open(p) as f: return int(json.load(f).get("version", 0))

def save_version(share_dir: str, v: int):
    p = os.path.join(share_dir, "model", "version.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f: json.dump({"version": int(v)}, f)

# --------------- Datasets ---------------
class FolderDatasetBalanced(Dataset):
    """훈련용: root/<label>/* — 클래스 상한 & 균형 샘플링"""
    def __init__(self, root, max_per_class=None):
        self.items = []
        self.normalize = transforms.Normalize((0.1307,), (0.3081,))
        g = torch.Generator().manual_seed(42)
        for c in range(10):
            ddir = os.path.join(root, str(c))
            pool = []
            if os.path.exists(ddir):
                for f in os.listdir(ddir):
                    if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp")):
                        pool.append(os.path.join(ddir, f))
            if not pool:
                continue
            if max_per_class and len(pool) > max_per_class:
                idx = torch.randperm(len(pool), generator=g)[:max_per_class].tolist()
                pool = [pool[i] for i in idx]
            self.items += [(p, c) for p in pool]

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        with open(path, "rb") as fp:
            img_bytes = fp.read()
        pil = preprocess_to_28x28(img_bytes)     # ★ 추론과 동일 파이프라인
        x = transforms.ToTensor()(pil)
        x = self.normalize(x)
        return x, y

def make_feedback_sampler(ds: FolderDatasetBalanced) -> WeightedRandomSampler:
    if len(ds) == 0:
        # dummy sampler 방지
        return None
    labels = torch.tensor([y for _, y in ds.items], dtype=torch.long)
    counts = torch.bincount(labels, minlength=10).float()
    w_per_class = 1.0 / (counts + 1.0)   # 라벨 적을수록 더 큰 가중치
    weights = w_per_class[labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

def get_mnist_train_subset(n: int = 5000, seed: int = 42) -> Subset:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full = datasets.MNIST(root="/app/.torch", train=True, download=True, transform= tfm)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(full), generator=g)[:min(n, len(full))].tolist()
    return Subset(full, idx)

def get_mnist_test_balanced_subset(per_class: int = 100, seed: int = 42) -> Subset:
    tfm = transforms.Compose([transforms.ToTensor()])  # Normalize는 증강 후 마지막
    full = datasets.MNIST(root="/app/.torch", train=False, download=True, transform=tfm)
    targets = full.targets if hasattr(full, "targets") else full.test_labels
    g = torch.Generator().manual_seed(seed)
    indices = []
    for c in range(10):
        pool = (targets == c).nonzero(as_tuple=False).squeeze(1)
        select = pool[torch.randperm(len(pool), generator=g)[:per_class]]
        indices.extend(select.tolist())
    return Subset(full, indices)

class AugmentedMnistTest(Dataset):
    """MNIST test 균등 subset → (±deg 회전 + 가우시안 노이즈) → 마지막에 Normalize"""
    def __init__(self, mnist_subset: Subset, rot_deg: int = 15, noise_std: float = 0.15):
        self.base = mnist_subset
        self.rot_deg = rot_deg
        self.noise_std = noise_std
        self.normalize = transforms.Normalize((0.1307,), (0.3081,))
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]              # x: [0,1]
        angle = self.rot_deg if (i % 2 == 0) else -self.rot_deg
        x = TF.rotate(x, angle)
        x = (x + torch.randn_like(x) * self.noise_std).clamp(0, 1)
        x = self.normalize(x)            # ★ 마지막에 Normalize
        return x, y

def _list_files(root, cls):
    p = os.path.join(root, str(cls))
    if not os.path.exists(p): return []
    return [os.path.join(p, f) for f in os.listdir(p)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]

def get_backup_balanced_paths(share_dir: str, per_class: int, exclude_train: bool = True, seed: int = 42):
    rng = torch.Generator().manual_seed(seed)
    selected = {c: [] for c in range(10)}
    train_paths = set()
    if exclude_train:
        for c in range(10):
            for fp in _list_files(os.path.join(share_dir, "data"), c):
                train_paths.add(os.path.abspath(fp))
    for c in range(10):
        pool = [os.path.abspath(fp) for fp in _list_files(os.path.join(share_dir, "backupdata"), c)]
        if exclude_train:
            pool = [fp for fp in pool if fp not in train_paths]
        if len(pool) == 0: continue
        idx = torch.randperm(len(pool), generator=rng)[:min(per_class, len(pool))]
        selected[c] = [pool[i] for i in idx.tolist()]
    return selected

class BackupAugTest(Dataset):
    """backupdata 균등 샘플(누수 제외)을 증강해서 테스트. Normalize는 마지막."""
    def __init__(self, items, rot_deg=15, noise_std=0.15):
        self.items = items
        self.rot_deg, self.noise_std = rot_deg, noise_std
        self.to_tensor = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize((0.1307,), (0.3081,))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("L")
        x = self.to_tensor(img)          # [0,1]
        angle = float(self.rot_deg) if (i % 2 == 0) else -float(self.rot_deg)
        x = TF.rotate(x, angle)
        x = (x + torch.randn_like(x) * self.noise_std).clamp(0, 1)
        x = self.normalize(x)            # ★ 마지막에 Normalize
        return x, y

# ======== NEW: 재학습용 로더 생성 (feedback + backup + mnist) ========
def make_retrain_loaders(
    share_dir: str,
    mnist_subset_n: int = 5000,
    batch_train: int = 64,
    batch_eval: int = 256,
    test_per_class: int = 100,
    test_rot_deg: int = 15,
    test_noise_std: float = 0.15,
    test_source: str = "hybrid_aug",   # "hybrid_aug" | "mnist_aug" | "backup_aug"
    max_per_class_feedback: int = 500
):
    feed_root   = os.path.join(share_dir, "data")
    backup_root = os.path.join(share_dir, "backupdata")

    ds_feedback = FolderDatasetBalanced(feed_root,   max_per_class=max_per_class_feedback)
    ds_backup   = FolderDatasetBalanced(backup_root, max_per_class=None)
    ds_mnist_sub = get_mnist_train_subset(n=mnist_subset_n)

    # 로더들
    sampler_fb = make_feedback_sampler(ds_feedback)
    sampler_bk = make_feedback_sampler(ds_backup)
    Lfb = DataLoader(ds_feedback, batch_size=batch_train, sampler=sampler_fb) if sampler_fb else None
    Lbk = DataLoader(ds_backup,   batch_size=batch_train, sampler=sampler_bk) if sampler_bk else None
    Lmn = DataLoader(ds_mnist_sub, batch_size=batch_train, shuffle=True)

    # Val용 합본
    to_concat = [ds_mnist_sub]
    if len(ds_feedback) > 0: to_concat.append(ds_feedback)
    if len(ds_backup)   > 0: to_concat.append(ds_backup)
    ds_combo = ConcatDataset(to_concat) if len(to_concat) > 1 else to_concat[0]

    n = len(ds_combo); val_n = max(1, int(n * 0.2))
    ds_tr, ds_va = random_split(ds_combo, [n - val_n, val_n],
                                generator=torch.Generator().manual_seed(42))
    Lva = DataLoader(ds_va, batch_size=batch_eval)

    # Test 세트
    test_subset = get_mnist_test_balanced_subset(per_class=test_per_class)
    Lte_clean = DataLoader(test_subset, batch_size=batch_eval)

    if test_source == "mnist_aug":
        Lte_aug = DataLoader(
            AugmentedMnistTest(test_subset, rot_deg=test_rot_deg, noise_std=test_noise_std),
            batch_size=batch_eval
        )
    elif test_source == "backup_aug":
        sel = get_backup_balanced_paths(share_dir, per_class=test_per_class, exclude_train=True)
        items = [(p, c) for c, ps in sel.items() for p in ps]
        Lte_aug = DataLoader(BackupAugTest(items, test_rot_deg, test_noise_std), batch_size=batch_eval)
    else:  # hybrid_aug
        Lte_aug = DataLoader(
            AugmentedMnistTest(get_mnist_test_balanced_subset(per_class=test_per_class),
                               rot_deg=test_rot_deg, noise_std=test_noise_std),
            batch_size=batch_eval
        )

    return (Lfb, Lbk, Lmn), Lva, Lte_clean, Lte_aug

# ======== NEW: 여러 로더 번갈아 학습 ========
def train_with_multi_loaders(model: nn.Module, loaders: List[DataLoader], Lva: DataLoader,
                             epochs=2, lr=1e-3) -> Tuple[nn.Module, float]:
    model.train().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_acc, best_sd = 0.0, None

    active = [L for L in loaders if L is not None]
    if not active:
        return model.eval(), 0.0

    for _ in range(epochs):
        iters = [iter(L) for L in active]
        steps = max(len(L) for L in active)
        for _s in range(steps):
            for it in iters:
                try:
                    x, y = next(it)
                except StopIteration:
                    continue
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = crit(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
        acc = evaluate(model.eval(), Lva)
        if acc > best_acc:
            best_acc = acc
            best_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        model.train()

    if best_sd: model.load_state_dict(best_sd, strict=True)
    return model.eval(), best_acc

# ---------------- Baseline (변경 없음) ----------------
def train_baseline(share_dir: str, epochs=2):
    ensure_dirs(share_dir)
    model_path = os.path.join(share_dir, "model", "model.pth")
    per_class = int(os.environ.get("MNIST_TEST_PER_CLASS", "100"))
    rot_deg = int(os.environ.get("TEST_ROT_DEG", "15"))
    noise_std = float(os.environ.get("TEST_NOISE_STD", "0.15"))

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    ds_tr = datasets.MNIST(root="/app/.torch", train=True, download=True, transform=tfm)
    ds_te = datasets.MNIST(root="/app/.torch", train=False, download=True, transform=tfm)

    train_size = len(ds_tr) - 5000
    ds_tr, ds_va = random_split(ds_tr, [train_size, 5000], generator=torch.Generator().manual_seed(42))

    Ltr = DataLoader(ds_tr, batch_size=64, shuffle=True)
    Lva = DataLoader(ds_va, batch_size=256)
    Lte_full = DataLoader(ds_te, batch_size=256)

    model = CNN()
    model, _ = train(model, Ltr, Lva, epochs=epochs, lr=1e-3)
    _ = evaluate(model, Lte_full)

    test_subset = get_mnist_test_balanced_subset(per_class=per_class)
    Lte_clean = DataLoader(test_subset, batch_size=256)
    Lte_aug = DataLoader(AugmentedMnistTest(test_subset, rot_deg=rot_deg, noise_std=noise_std), batch_size=256)

    loss_clean, acc_clean = evaluate_with_loss(model, Lte_clean)
    loss_aug,   acc_aug   = evaluate_with_loss(model, Lte_aug)

    save_model(model, model_path)
    v = load_version(share_dir) + 1
    save_version(share_dir, v)
    write_latest_metrics(share_dir, version=v,
                             acc_clean=float(acc_clean), acc_aug=float(acc_aug),
                             loss_clean=float(loss_clean), loss_aug=float(loss_aug))
    print(f"[baseline] clean acc={acc_clean:.4f}, loss={loss_clean:.4f} | aug acc={acc_aug:.4f}, loss={loss_aug:.4f} | saved:{model_path}")

# ======== NEW: 재학습 1회 수행 ========
def retrain_once(share_dir: str, epochs=2, lr=1e-3):
    ensure_dirs(share_dir)
    model_path = os.path.join(share_dir, "model", "model.pth")

    (Lfb, Lbk, Lmn), Lva, Lte_clean, Lte_aug = make_retrain_loaders(
        share_dir=share_dir,
        mnist_subset_n=int(os.environ.get("MNIST_SUBSET_N", "5000")),
        batch_train=64,
        batch_eval=256,
        test_per_class=int(os.environ.get("MNIST_TEST_PER_CLASS", "100")),
        test_rot_deg=int(os.environ.get("TEST_ROT_DEG", "15")),
        test_noise_std=float(os.environ.get("TEST_NOISE_STD", "0.15")),
        test_source=os.environ.get("TEST_SOURCE", "hybrid_aug"),
        max_per_class_feedback=500
    )

    model = load_model(model_path) if os.path.exists(model_path) else CNN().to(DEVICE)

    loaders = [Lfb, Lbk, Lmn]
    if all(L is None for L in loaders):
        print("[retrain] No data from feedback/backup. Skipping retrain.")
        return False

    model, _ = train_with_multi_loaders(model, loaders, Lva, epochs=epochs, lr=lr)

    # 평가 & 저장 & 버전업
    loss_clean, acc_clean = evaluate_with_loss(model, Lte_clean)
    loss_aug,   acc_aug   = evaluate_with_loss(model, Lte_aug)
    save_model(model, model_path)
    v = load_version(share_dir) + 1
    save_version(share_dir, v)
    write_latest_metrics(share_dir, version=v,
                         acc_clean=float(acc_clean), acc_aug=float(acc_aug),
                         loss_clean=float(loss_clean), loss_aug=float(loss_aug))
    print(f"[retrain] clean acc={acc_clean:.4f}, loss={loss_clean:.4f} | aug acc={acc_aug:.4f}, loss={loss_aug:.4f} | saved:{model_path}")

    # ★ 트리거 시점에만 누적 + data 비움
    persist_feedback_to_backup(share_dir)
    clear_data_folder(share_dir)
    print("[retrain] persisted data/* → backupdata/* and cleared data/*.")
    return True

if __name__ == "__main__":
    SHARE = os.environ.get("SHARE_DIR", "/app/share_storage")
    torch.set_float32_matmul_precision("high")
    # 초기 베이스라인 1회 학습
    train_baseline(SHARE, epochs=2)
    # 필요 시 재학습 테스트
    # retrain_once(SHARE, epochs=2, lr=1e-3)
