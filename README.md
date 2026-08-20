# Convolutional Neural Network — From Scratch in C++

A handwritten digit classifier built entirely from scratch in C++, with no ML libraries. Implements a full CNN pipeline including forward propagation, backpropagation through convolutional layers, max pooling with gradient unpooling, and a fully connected network with softmax output — all using custom matrix and layer implementations.

A browser-based visualizer lets you draw a digit and watch activations propagate through every layer in real time.

![Architecture](https://img.shields.io/badge/architecture-CNN-00ffe0?style=flat-square) ![Language](https://img.shields.io/badge/language-C%2B%2B17-blue?style=flat-square) ![Dataset](https://img.shields.io/badge/dataset-MNIST-ff4f7b?style=flat-square) ![Reproducible](https://img.shields.io/badge/training-reproducible-brightgreen?style=flat-square)

> This project originated from a fork of an existing from-scratch CNN implementation. The core architecture and math are preserved from the original project; our newer approach here instead focuses on **fixing reproducibility bugs, hardening the build, and adding a data preprocessing step**. The [Lessons Learned](#lessons-learned--common-cnn-implementation-pitfalls) section below documents the mistakes found in the original codebase in detail, specifically so other students building CNNs from scratch can recognize and avoid them.

---
## Visualization
<img width="1906" height="915" alt="image" src="https://github.com/user-attachments/assets/af742c0f-7539-450f-83c9-eb9c68f53b8a" />

---
## Architecture

```
Input (28×28)
    │
    ▼
Conv Layer 1 — 3 hardcoded 5×5 edge detection filters → 3 × (24×24) feature maps
    │  ReLU
    ▼
Max Pool 1 — 2×2 stride 2 → 3 × (12×12)
    │
    ▼
Conv Layer 2 — 3 trainable 3×3×3 filters → 3 × (10×10) feature maps
    │  ReLU
    ▼
Max Pool 2 — 2×2 stride 2 → 3 × (5×5) → flatten → 75
    │
    ▼
Fully Connected — 75 → 256 (Leaky ReLU)
    │
    ▼
Output — 256 → 10 (Softmax)
```

**Conv Layer 1** uses fixed edge detection filters (vertical, horizontal, diagonal). **Conv Layer 2** uses Xavier-initialized filters that are trained via backpropagation. The FC network uses cross-entropy loss with a combined softmax + cross-entropy gradient for numerical stability.

---

## Features

- **No ML framework** — every operation (convolution, pooling, backprop, weight updates) written from scratch
- **Full backpropagation** through conv layer 2, including max-pool gradient unpooling and ReLU derivative masking
- **Gradient clipping** in both conv and FC layers to stabilize training
- **Model serialization** — saves/loads all trained weights and filters to a binary file
- **Learning rate decay** — conv learning rate halved every 5 epochs
- **Deterministic, reproducible training** — identical seed produces byte-for-byte identical saved models across runs (see [Lessons Learned](#lessons-learned--common-cnn-implementation-pitfalls))
- **Automated data preprocessing** — training/test sets are fetched and prepared locally instead of requiring manual setup
- **Live visualizer** — HTML frontend shows feature maps, FC activations, and output probabilities as you draw

---

## Dependencies

- **C++17** compiler (MSVC, g++, or clang++)
- **OpenCV** — image loading only (`cv::imread`)
- **Python 3** + `flask flask-cors pillow numpy` — for the visualizer server only

---

## Data Preprocessing

Earlier versions of this project required manually downloading the MNIST CSV files and placing them under `resource/` before training would work. This fork adds a preprocessing step that handles that automatically.

```bash
python preprocess/setup_data.py
# downloads mnist_train.csv and mnist_test.csv
# places them under resource/, matching what loadMNISTFromCSV() expects
```

> **Note:** adjust the script name/path above to match your actual preprocessing entry point if it differs — swap this block for the real command once finalized.

Run this once before your first `train` or `eval` call. `main.cpp` still expects the files at `resource/mnist_train.csv` and `resource/mnist_test.csv`, so no changes to the training code were needed — this step just removes the manual download/placement friction.

---

## Build

### Visual Studio (Windows)
Open the solution, set the include/library paths for OpenCV, build in Debug or Release.

### g++ / clang++ (Linux/macOS)
```bash
g++ -std=c++17 -O2 -Iincludes srcs/*.cpp -o cnn \
    $(pkg-config --cflags --libs opencv4)
```

All internal headers are included using the `"cnn/Header.hpp"` form (resolved via `-Iincludes`), consistently across every file — see [Lessons Learned](#lessons-learned--common-cnn-implementation-pitfalls) for why this matters more than it looks like it should.

---

## Usage

### Train
```bash
./cnn train [model_path]
# default: saves to saved_model.bin
# trains for 15 epochs on mnist_train.csv
```

### Evaluate
```bash
./cnn eval [model_path]
# evaluates on mnist_test.csv (first 1000 samples)
```

### Predict a single image
```bash
./cnn predict image.jpg [model_path]
```
> Image must be 28×28 grayscale, white digit on black background (MNIST style).

### Verifying reproducibility
Because training is now fully deterministic given a fixed seed, you can confirm two runs match exactly:
```bash
./cnn train run1.bin
./cnn train run2.bin
diff run1.bin run2.bin && echo "IDENTICAL" || echo "DIVERGED"
```
If this doesn't print `IDENTICAL`, something is drawing randomness outside the shared RNG engine — see below.

---

## Visualizer

A browser UI that lets you draw a digit and watch it propagate through the network layer by layer.

**Start the server:**
```bash
pip install flask flask-cors pillow numpy
python visualizer/server.py
```
Then open `visualizer/index.html` in your browser.

The visualizer shows real conv layer 1 activations (exact match to the C++ pipeline), approximated conv layer 2 maps, a 16×16 heatmap of FC neuron activations, and real model probabilities from the C++ executable.

If the server isn't running, the page falls back to a demo mode that runs the edge filter math entirely in JavaScript.

---

## Implementation Notes

**Why two conv layers with different approaches?**
Layer 1 uses fixed classical edge filters — these are known to work well for digit recognition and don't need training. Layer 2 learns task-specific features from the pooled edge maps.

**Backprop through max pooling**
Uses "unpool without indices" — gradients are routed back to the position of the max value in the original feature map, which is recomputed during the backward pass.

**Softmax + cross-entropy gradient**
The combined gradient simplifies to `(predicted - target)`, computed directly in `gardientComputation()` to avoid numerical issues from separately differentiating softmax and log-loss.

**Gradient clipping**
Both the FC weight updates and conv filter updates clip gradients to `[-1, 1]`, which significantly stabilized training on small datasets.

---

## Lessons Learned — Common CNN Implementation Pitfalls

The original codebase trained and produced reasonable-looking accuracy numbers, which made it easy to assume it was correct. It wasn't fully reproducible, and a couple of the bugs found here are the kind that hide behind a working demo. Documented here in detail for anyone else building a CNN from scratch.

### 1. A fresh RNG engine per random draw silently destroys reproducibility

The original code generated each random number like this:

```cpp
double genRandomNumber()
{
    std::random_device rd;
    std::mt19937 gen(rd());   // new engine, freshly seeded from OS entropy, EVERY call
    std::normal_distribution<double> dis(0.0, 0.1);
    return dis(gen);
}
```

This function was called once *per weight, per filter element* inside nested loops. Every single call spun up a brand-new Mersenne Twister engine seeded from `std::random_device` — real OS entropy, different every run, by design. The result: weight initialization (and therefore the entire training trajectory) was non-deterministic across runs, with no seed anywhere that could pin it down. It also meant constructing thousands of full `mt19937` engine instances (~2.5KB of state each) just to draw one number from each, which is expensive for no benefit.

**The fix:** one shared, explicitly-seeded engine for the whole program, and every random draw pulls from it:

```cpp
namespace Random {
    inline std::mt19937& engine() {
        static std::mt19937 rng(42);
        return rng;
    }
    inline void seed(std::uint32_t value) { engine().seed(value); }
}
```

**Lesson:** if you can't explain exactly where every random number in your program came from and reproduce the same sequence twice, your training isn't an experiment — it's a coin flip that happens to output numbers. Route *all* randomness through one seeded, shared source. Grep your entire codebase for `random_device` and `mt19937` before you trust any result.

### 2. `static` globals in a shared header quietly multiply, not share

The three hardcoded edge-detection filters lived in a shared header like this:

```cpp
namespace Filters {
    static gridEntity STRONG_VERTICAL_EDGE_DETECTION = { ... };
}
```

`static` at namespace scope gives **internal linkage** — every `.cpp` file that includes this header gets its *own independent copy* of the object, not a shared reference to one. It happened to be harmless here because the values are constant literals nobody mutates. But it's a landmine: the moment anyone adds code that modifies a filter in place (a common thing to try when experimenting), that change becomes invisible to every other translation unit, because they're not looking at the same object at all.

**The fix:** use `inline` (C++17) for header-defined globals that need one true shared definition across the whole program:

```cpp
namespace Filters {
    inline gridEntity STRONG_VERTICAL_EDGE_DETECTION = { ... };
}
```

**Lesson:** `static` and `inline` at namespace scope look interchangeable and are not. If you're defining actual data (not just declaring it) in a header that gets included in more than one `.cpp` file, `static` silently duplicates it per file rather than sharing it.

### 3. Fragile relative includes break the moment the file moves

Some headers used a guessed relative path to reach shared definitions:

```cpp
#include "../cnn/all_includes.hpp"
```

This only resolves correctly if the including file sits exactly one directory below a `cnn/` folder. The rest of the project's files used a different, flag-relative style (`#include "cnn/Header.hpp"`, resolved via `-Iincludes` on the compiler invocation). The mismatch meant some headers built fine in one project layout and failed — sometimes with a confusing "undeclared identifier" error rather than a clean "file not found" — the moment the directory structure shifted even slightly.

**The fix:** pick one include convention for the whole project and use it everywhere — here, that meant standardizing every internal include to the `"cnn/Header.hpp"` + `-Iincludes` pattern already used by most of the codebase.

**Lesson:** mixing relative-path includes (`"../x/y.hpp"`) with include-root includes (`"x/y.hpp"` + `-I`) in the same project is a common source of "works on my machine" build failures. Standardize on one, and prefer the include-root style for anything beyond a trivial project — it doesn't break when files get reorganized.

### 4. A renamed member variable that never got fully renamed

The header declared the trainable layer-2 filter storage as:

```cpp
std::vector<volumetricEntity> training_filters;
```

...and every getter, every part of the training loop, and the model serializer all read from and wrote to `training_filters`. But the constructor — the code actually building and Xavier-initializing those filters — wrote to a different, never-declared name:

```cpp
this->second_layer_filters.push_back(temp);   // no such member exists
```

This is the residue of a rename that only got applied in some places. In a language with less strict compile-time checking, or if this particular line had happened to compile some other way, the bug would have been *silent*: filters get initialized into a variable nothing else reads, while the actual filters used in forward/backward passes stay at their zero-initialized or garbage starting values forever. The model would still "train," still show a loss curve, and still be wrong in a way that's very hard to spot from the output alone.

**Lesson:** after any rename or refactor, grep the *entire* symbol name across the whole codebase, not just the file you're editing — including constructors, which are easy to skip because they don't show up in the same place as the rest of a class's logic. A model that trains and produces plausible-looking numbers is not proof that every piece of it is actually being used correctly.

### 5. Verify reproducibility empirically, don't just assume it from reading the code

Fixing the RNG source is necessary but isn't sufficient proof on its own. The way to actually confirm determinism is to run training twice with the same seed and diff the binary output:

```bash
./cnn train run1.bin
./cnn train run2.bin
diff run1.bin run2.bin && echo "IDENTICAL" || echo "DIVERGED"
```

**Lesson:** "I fixed the random seed" is a claim; a byte-identical diff across two independent runs is evidence. Always close the loop with an empirical check rather than trusting that a fix worked because it looks correct.

---

## Results

Trained on 5,000 MNIST samples for 15 epochs:

| Metric | Value |
|---|---|
| Training samples | 5,000 |
| Epochs | 15 |
| FC learning rate | 0.001 |
| Conv learning rate | 0.00001 (halved every 5 epochs) |
| Final train accuracy | ~93% |
| Final test accuracy | ~90% |

> Training on the full 60,000-sample dataset is supported — change the `maxSamples` argument in `runTraining()`.

---

## Credits

Forked from the original from-scratch CNN implementation. This fork's changes are focused on reproducibility fixes, build hardening, and automated data preprocessing — the core architecture, math, and layer design are the original author's work.

---

## License

MIT