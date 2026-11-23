# CompSci471Final
# Ilan Brejt, Grant Stec

## Basic CNN: Tank vs. Not-Tank Image Classification

A terminal-based Python project that trains a small Convolutional Neural Network (PyTorch) to classify
cartoon-style military vehicle images as **TANK** (label 1) or **NOT TANK** (label 0).
All imagery is self-generated and synthetic (non-operational).

### Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | Draws the 40-image synthetic dataset (20 tanks, 20 trucks / wheeled APCs / jeeps) |
| `data/tank/`, `data/not_tank/` | The labeled dataset (folder name = label) |
| `tank_cnn.py` | Loads data, splits 70/15/15, builds + trains the CNN, evaluates, and predicts new images |
| `tank_cnn.pt` | Saved trained weights (created when you run `tank_cnn.py`) |
| `loss_curve.png` | Training vs. validation loss graph (created if matplotlib is installed) |

### Setup

```bash
pip install -r requirements.txt
```

### Run

```bash
python generate_dataset.py     # only needed if data/ is missing (images are already in the repo)
python tank_cnn.py             # full demo: load -> split -> train -> test -> try your own image
```

Other options:

```bash
python tank_cnn.py --predict data/tank/tank_05.png   # classify one image with the saved model
python tank_cnn.py --epochs 100 --lr 0.001           # change hyperparameters
python tank_cnn.py --no-interactive                  # skip the "enter an image path" prompt
python generate_dataset.py --out new_images --seed 5 # make brand-new unseen images to test on
```

### What the program does

1. **Loads** every image in `data/tank` (label 1) and `data/not_tank` (label 0), resized to 64x64 RGB.
2. **Displays** the number of images in each category (20 / 20).
3. **Splits** 70 / 15 / 15, balanced by class: 28 training, 6 validation, 6 testing.
   Test images are never used for training or for choosing the model.
4. **Builds** the CNN:

   ```
   Input 3x64x64
   Conv2D 3->8,  5x5, stride 2, padding 2 -> ReLU -> MaxPool 2x2   => 8 x 16 x 16
   Conv2D 8->16, 3x3, stride 1, padding 1 -> ReLU -> MaxPool 2x2   => 16 x 8 x 8
   Global MaxPool (8x8 -> 1x1)                                     => 16 values
   Linear 16 -> 1  (classification output; > 0.5 probability = TANK)
   ```
   Output size per conv layer = `(W - K + 2P) / S + 1`. ~1,800 trainable parameters in total.
5. **Trains** with binary cross-entropy loss, backpropagation, and the Adam optimizer, printing
   training loss, validation loss, and validation accuracy every epoch. Images are randomly mirrored
   left/right during training so direction doesn't matter. The weights from the epoch with the lowest
   **validation** loss are kept.
6. **Tests** on the 6 held-out images and prints each prediction, the confusion matrix
   (TP / TN / FP / FN), and test accuracy.
7. **Prompts** for a path to any new image and prints TANK / NOT TANK with a confidence.

### Results (default seed 471, 60 epochs)

- Test accuracy: **100% (6/6)**. TP = 3, TN = 3, FP = 0, FN = 0
- On 50 brand-new images made with a different generator seed: **47/50 (94%)**

### Notes for discussion

- **Overfitting:** the first version flattened the 16x8x8 feature map into a 1,024-input layer.
  It memorized the 28 training images (training loss fell, validation loss rose) and scored 50% on test.
  Replacing that with global max pooling (16 inputs) fixed it. The loss graph shows this effect early on:
  validation loss rises for about 20 epochs before the convolution filters learn general features.
- **Small-data variance:** with only 6 test images, each one is worth 16.7%. Different `--seed` values
  (which change the split and starting weights) gave between 50% and 100% test accuracy, and a few
  seeds get stuck near 50%. More images would make the results more stable.
- **Using your own images:** drop `.png`/`.jpg` files into `data/tank` and `data/not_tank`
  (cartoon / video-game / synthetic only). Any size works; they are resized automatically.
