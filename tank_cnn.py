"""
tank_cnn.py
-----------
CompSci 471 Final - Basic CNN Tank vs. Not-Tank Image Classification
Ilan Brejt, Grant Stec

Runs entirely from the terminal:
  1. Loads the labeled image dataset           (data/tank -> 1, data/not_tank -> 0)
  2. Shows how many images are in each category
  3. Splits the data 70 / 15 / 15  ->  train / validation / test
  4. Builds a small CNN (Conv2D + padding + stride, ReLU, MaxPool, output layer)
  5. Trains it and prints training + validation loss every epoch
  6. Evaluates on the untouched test set: accuracy + confusion matrix
  7. Lets you type the path of a new image and get TANK / NOT TANK

Usage:
    python tank_cnn.py                         # full run (train, test, then predict loop)
    python tank_cnn.py --epochs 40 --lr 0.001  # change hyperparameters
    python tank_cnn.py --predict some_image.png  # use the saved model on one image
    python tank_cnn.py --no-interactive        # skip the prediction prompt at the end
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMG_SIZE = 64                      # every image is resized to 64 x 64 RGB
CLASS_NAMES = {0: "NOT TANK", 1: "TANK"}
CLASS_FOLDERS = {"tank": 1, "not_tank": 0}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
MODEL_PATH = "tank_cnn.pt"


# =====================================================================
# 1. DATA LOADING
# =====================================================================
def load_image(path):
    """Open any image, force RGB, resize to IMG_SIZE, scale pixels to [0, 1].
    Returns a tensor shaped (3, H, W) = (channels, height, width)."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0      # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1)        # (3, H, W)


def load_dataset(data_dir):
    """Returns a list of (image_tensor, label, filename)."""
    samples = []
    for folder, label in CLASS_FOLDERS.items():
        path = os.path.join(data_dir, folder)
        if not os.path.isdir(path):
            sys.exit(f"Missing folder: {path}\nRun:  python generate_dataset.py")
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(IMG_EXTS):
                samples.append((load_image(os.path.join(path, name)), label, name))
    return samples


# =====================================================================
# 2. 70 / 15 / 15 SPLIT  (stratified so every split has both classes)
# =====================================================================
def split_dataset(samples, train_frac=0.70, val_frac=0.15):
    train, val, test = [], [], []
    for label in (1, 0):
        group = [s for s in samples if s[1] == label]
        random.shuffle(group)
        n_train = round(len(group) * train_frac)
        n_val = round(len(group) * val_frac)
        train += group[:n_train]
        val += group[n_train:n_train + n_val]
        test += group[n_train + n_val:]           # test is never trained on
    random.shuffle(train)
    return train, val, test


def to_tensors(split):
    x = torch.stack([s[0] for s in split])                      # (N, 3, 64, 64)
    y = torch.tensor([s[1] for s in split], dtype=torch.float32)  # (N,)
    return x, y


# =====================================================================
# 3. THE CNN
# =====================================================================
class TankCNN(nn.Module):
    """
    Input  : 3 x 64 x 64 image
    Block 1: Conv2D(3->8,  5x5, stride=2, padding=2) -> ReLU -> MaxPool(2)  => 8 x 16 x 16
    Block 2: Conv2D(8->16, 3x3, stride=1, padding=1) -> ReLU -> MaxPool(2)  => 16 x 8 x 8
    Global : MaxPool over the whole 8 x 8 map                              => 16 x 1 x 1
             (each of the 16 filters reports its strongest response anywhere
              in the image, so the vehicle can be on the left or right)
    Output : Flatten (16) -> Linear(16 -> 1) -> one score (logit)
             score > 0  (probability > 0.5)  => TANK, otherwise NOT TANK

    Output size of a conv layer = floor((W - K + 2P) / S) + 1
      Block 1 conv: (64 - 5 + 2*2) / 2 + 1 = 32   -> pool -> 16
      Block 2 conv: (16 - 3 + 2*1) / 1 + 1 = 16   -> pool -> 8
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # padding=2 keeps edge pixels in play; stride=2 halves the work
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),                      # negatives -> 0, positives kept
            nn.MaxPool2d(kernel_size=2),    # keep the strongest activation in each 2x2

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # global max pooling: 8x8 -> 1x1. Keeps the model tiny so it can't just
            # memorize 28 training images (a 1024-input layer overfit badly).
            nn.AdaptiveMaxPool2d(1),
        )
        self.classifier = nn.Linear(16, 1)            # binary output

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)          # raw score (logit)


# =====================================================================
# 4. TRAINING
# =====================================================================
def train(model, train_xy, val_xy, epochs, lr, batch_size):
    x_tr, y_tr = train_xy
    x_va, y_va = val_xy
    loss_fn = nn.BCEWithLogitsLoss()      # sigmoid + binary cross-entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val, best_state, best_epoch = float("inf"), None, 0

    print("\n" + "=" * 52)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>7}")
    print("-" * 52)
    for epoch in range(1, epochs + 1):
        # ---- training pass ----
        model.train()
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(0, len(x_tr), batch_size):
            idx = perm[i:i + batch_size]
            xb = x_tr[idx].clone()
            # simple augmentation: randomly mirror images left/right so the model
            # learns "tank" regardless of which way the vehicle faces
            flip = torch.rand(len(idx)) < 0.5
            xb[flip] = torch.flip(xb[flip], dims=[3])
            optimizer.zero_grad()
            logits = model(xb)                     # 1. predict
            loss = loss_fn(logits, y_tr[idx])      # 2. compare to labels
            loss.backward()                        # 3. backpropagation
            optimizer.step()                       # 4. adjust weights
            total += loss.item() * len(idx)
        train_loss = total / len(x_tr)

        # ---- validation pass (no weight updates) ----
        model.eval()
        with torch.no_grad():
            v_logits = model(x_va)
            val_loss = loss_fn(v_logits, y_va).item()
            val_acc = ((v_logits > 0).float() == y_va).float().mean().item()

        marker = ""
        if val_loss < best_val:                    # model selection uses validation only
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        history.append((epoch, train_loss, val_loss))
        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | {val_acc:>6.0%}{marker}")
    print("=" * 52)

    model.load_state_dict(best_state)
    print(f"Keeping weights from epoch {best_epoch} (lowest validation loss = {best_val:.4f})")

    # Overfitting check: training loss keeps dropping while validation loss climbs
    final_tr, final_va = history[-1][1], history[-1][2]
    if final_va > best_val * 1.5 and final_tr < final_va:
        print("Note: validation loss rose after its minimum while training loss kept "
              "falling -> sign of overfitting (that is why the best epoch is kept).")
    return history


# =====================================================================
# 5. EVALUATION
# =====================================================================
def evaluate(model, test_split):
    x_te, y_te = to_tensors(test_split)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_te))
    preds = (probs > 0.5).long()
    y = y_te.long()

    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)

    print("\nTest set predictions (these images were never used for training):")
    for (_, label, name), p, pr in zip(test_split, preds.tolist(), probs.tolist()):
        ok = "correct" if p == label else "WRONG"
        print(f"  {name:<20} true: {CLASS_NAMES[label]:<9} pred: {CLASS_NAMES[p]:<9} "
              f"P(tank)={pr:.2f}  {ok}")

    print("\nCONFUSION MATRIX")
    print("                      Predicted")
    print("                  TANK      NOT TANK")
    print(f"  Actual TANK     TP = {tp:<5} FN = {fn}")
    print(f"  Actual NOT TANK FP = {fp:<5} TN = {tn}")
    print(f"\n  True Positives  (tanks correctly found):        {tp}")
    print(f"  True Negatives  (non-tanks correctly rejected): {tn}")
    print(f"  False Positives (non-tanks called tanks):       {fp}")
    print(f"  False Negatives (tanks that were missed):       {fn}")
    print(f"\nTEST ACCURACY: {acc:.1%}  ({tp + tn}/{len(y)} correct)")
    return acc


# =====================================================================
# 6. PREDICT A NEW IMAGE
# =====================================================================
def predict(model, path):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(load_image(path).unsqueeze(0))).item()
    label = 1 if p > 0.5 else 0
    conf = p if label == 1 else 1 - p
    print(f"  -> {CLASS_NAMES[label]}  (confidence {conf:.0%}, P(tank) = {p:.2f})")


def predict_loop(model):
    print("\nTry your own image! Enter a path to a vehicle image (or press Enter / 'q' to quit).")
    while True:
        try:
            path = input("Image path: ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if path.lower() in ("", "q", "quit", "exit"):
            break
        if not os.path.isfile(path):
            print("  File not found, try again.")
            continue
        try:
            predict(model, path)
        except Exception as e:  # unreadable / non-image file
            print(f"  Could not read that image: {e}")


def save_loss_plot(history, path="loss_curve.png"):
    """Optional: saves a loss graph if matplotlib is installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ep = [h[0] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(ep, [h[1] for h in history], label="Training loss")
    plt.plot(ep, [h[2] for h in history], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("Tank CNN - training vs. validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"Saved loss graph to {path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Basic CNN: TANK vs NOT TANK")
    ap.add_argument("--data", default="data", help="dataset folder with tank/ and not_tank/")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.005, help="learning rate")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=471)
    ap.add_argument("--predict", metavar="IMAGE", help="classify one image with the saved model and exit")
    ap.add_argument("--no-interactive", action="store_true", help="skip the prediction prompt")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.predict:
        if not os.path.isfile(MODEL_PATH):
            sys.exit(f"No saved model ({MODEL_PATH}). Run  python tank_cnn.py  first.")
        model = TankCNN()
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Image: {args.predict}")
        predict(model, args.predict)
        return

    print("=" * 52)
    print("  BASIC CNN - TANK vs. NOT TANK CLASSIFIER")
    print("=" * 52)

    # 1-2. load + count
    samples = load_dataset(args.data)
    n_tank = sum(1 for s in samples if s[1] == 1)
    print(f"\nLoaded {len(samples)} images from '{args.data}/' (resized to {IMG_SIZE}x{IMG_SIZE} RGB)")
    print(f"  TANK     (label 1): {n_tank}")
    print(f"  NOT TANK (label 0): {len(samples) - n_tank}")

    # 3. split
    train_s, val_s, test_s = split_dataset(samples)
    print("\nDataset split (70 / 15 / 15):")
    for name, s in (("Training", train_s), ("Validation", val_s), ("Testing", test_s)):
        t = sum(1 for x in s if x[1] == 1)
        print(f"  {name:<10}: {len(s):>2} images  ({t} tank, {len(s) - t} not tank)")

    # 4. build
    model = TankCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print("\nCNN architecture:")
    print(model)
    print(f"Trainable parameters: {n_params:,}")

    # 5. train
    history = train(model, to_tensors(train_s), to_tensors(val_s),
                    args.epochs, args.lr, args.batch_size)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
    save_loss_plot(history)

    # 6. test
    evaluate(model, test_s)

    # 7. new image
    if not args.no_interactive:
        predict_loop(model)
    print("\nDone.")


if __name__ == "__main__":
    main()
