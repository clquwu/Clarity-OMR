# Clarity-OMR

Optical Music Recognition — convert PDF sheet music to MusicXML.

Models are downloaded automatically from [HuggingFace](https://huggingface.co/clquwu/Clarity-OMR) on first run. Stage-B uses `info/model.safetensors`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/clquwu/Clarity-OMR.git
cd Clarity-OMR
```

### 2. Install PyTorch

**GPU users (recommended):** Install PyTorch with your CUDA version **before** installing the other dependencies. This is important — if you skip this step, the CPU-only version of PyTorch will be installed instead and you won't get GPU acceleration.

Find your CUDA version:
```bash
nvidia-smi
```

Then install the matching PyTorch build:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for all available versions.

**CPU-only users:** Skip this step. The CPU version of PyTorch will be installed automatically in the next step.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage — outputs score-omr.musicxml next to the input PDF
python omr.py score.pdf

# Custom output path
python omr.py score.pdf -o output.musicxml

# Fast mode (beam-width 2 instead of 5, lower quality but ~2x faster)
python omr.py score.pdf --fast
```

### Options

| Flag | Description |
|---|---|
| `-o`, `--output` | Output MusicXML path (default: `<input>-omr.musicxml`) |
| `--fast` | Faster inference with beam-width 2 (default: 5) |
| `--beam-width N` | Override beam width directly |
| `--device cuda/cpu` | Force device (auto-detected by default) |
| `--pdf-dpi N` | PDF render resolution (default: 300) |
| `--work-dir PATH` | Directory for intermediate files |

### What happens when you run it

1. Models are resolved on first run: `yolo.pt` and `info/model.safetensors` are downloaded from HuggingFace
2. PDF is rendered to page images at 300 DPI
3. YOLO detects staff regions on each page
4. The OMR model recognizes music notation tokens from each staff crop
5. Tokens are assembled into a full score and exported as MusicXML

The output MusicXML can be opened in MuseScore, Finale, Sibelius, or any notation software.

## Comparing results

To compare your OMR output against a ground-truth MusicXML file:

```bash
pip install mir_eval

python src/eval/compare_musicxml.py ground_truth.musicxml omr_output.musicxml
```

This uses [mir_eval](https://github.com/mir-evaluation/mir_eval) transcription metrics (note-level precision, recall, F1) to measure how accurately the OMR output reproduces the original score.

## License

GPL-3.0 — see [LICENSE](LICENSE).
