# Money Counter

Detects and recognises coins in a photo with a Keras model and OpenCV, then reports the total amount (FCFA).

## How it works

1. `HoughCircles` finds coin-shaped regions in the image.
2. Each region is cropped, resized to 224x224, and saved to `temp_images/`.
3. `model/best_coin_recognition_model.h5` classifies each crop. Labels are in `model/labels.txt`.
4. Coins classified as noise are discarded. The rest are summed and drawn on the image.

## Setup

```bash
pip install tensorflow keras opencv-python pillow numpy
jupyter notebook main.ipynb
```

## Usage

Open `main.ipynb` and change the image path in the second cell (currently `data/test_images/4.jpg`). Run all cells. Each `cv2.imshow` window waits for a key press.

## Notes

- Requires a display for `cv2.imshow`. On a headless machine, remove or replace those calls.
- `temp_images/` and `.ipynb_checkpoints/` are generated files and are ignored by git.
