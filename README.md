# VAV — ViT Attention Visualizer 
### (Currently in the testing phase, only Jupyter Notebook is provided for quick validation.)
<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Visualize attention mechanisms in Vision Transformers (ViT) with beautiful heatmaps and interactive analysis**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Usage](#-usage) • [API Documentation](#-api-documentation)

</div>

---

## 📖 Overview

**VAV (ViT Attention Visualizer)** is a powerful tool for visualizing and understanding attention patterns in Vision Transformer models. Built on top of Hugging Face Transformers and PyTorch, VAV provides multiple visualization modes including CLS token attention, attention rollout, and layer-wise analysis.

Whether you're researching transformer architectures, debugging model behavior, or exploring how ViTs process visual information, VAV makes it easy to generate insightful attention heatmaps for images and PDF documents.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔄 **Attention Rollout** | Smoothed attention flow across all layers using the Abnar & Zuidema (2020) method |
| 🎯 **CLS Attention** | Visualize the [CLS] token's attention over spatial patches for any layer and attention head |
| 📊 **All Layers Average** | Aggregate attention patterns across all encoder layers |
<!-- | 🖼️ **Image Analysis** | Process single images with base64 encoding support | -->
<!-- | 📄 **PDF Support** | Analyze multi-page PDF documents with per-page attention visualization | -->
<!-- | 🚀 **FastAPI Backend** | RESTful API for integration into web applications | -->

## 🛠️ Installation

### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- **Git** (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/VAV.git
cd VAV
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Using conda
conda create -n VAV python==3.12
conda activate VAV
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for PDF Support:**
- **PyMuPDF (recommended)**: Automatically installed via `pymupdf` package
- **pdf2image (alternative)**: Requires additional system dependencies:
  - **Windows**: Download [poppler for Windows](http://blog.alivate.com.au/poppler-windows/) and add to PATH
  - **macOS**: `brew install poppler`
  - **Linux**: `sudo apt-get install poppler-utils` (Ubuntu/Debian) or `sudo yum install poppler-utils` (CentOS/RHEL)

## 📥 Download Model Weights

VAV requires a pre-trained Vision Transformer model. You need to download the model weights from Hugging Face and place them in a local directory.

### Recommended Model

We recommend using **`google/vit-base-patch16-224`**, which is a well-tested ViT model with 12 layers and 12 attention heads.

### Method 1: Manual Download from Hugging Face Hub

1. Visit [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) on Hugging Face
2. Navigate to the "Files and versions" tab
3. Download the following files:
   - `config.json`
   - `preprocessor_config.json`
   - `model.safetensors` (or `pytorch_model.bin`)
4. Create a `vit/` directory in your project root
5. Place all downloaded files in the `vit/` directory

### Method 2: Using Hugging Face CLI (Recommended)

1. **Install Hugging Face CLI** (if not already installed):
   ```bash
   pip install huggingface-hub
   ```

2. **Download the model**:
   ```bash
   # Create the model directory
   mkdir vit
   
   # Download model, config, and processor
   huggingface-cli download google/vit-base-patch16-224 --local-dir vit
   ```

3. **Verify the download**: The `vit/` directory should contain:
   - `config.json` - Model configuration
   - `preprocessor_config.json` - Image processor configuration
   - `model.safetensors` or `pytorch_model.bin` - Model weights

### Alternative Models

You can use any Hugging Face ViT model that supports `ViTForImageClassification`. Popular alternatives:

- `google/vit-large-patch16-224` - Larger model with more parameters
- `google/vit-base-patch32-224` - Different patch size
- `WinKawaks/vit-small-patch16-224` - Smaller, faster model

To use a different model, change the `model_name_or_path` parameter in the code (e.g., `ViT(model_name_or_path="vit_large")` for a model in `vit_large/` directory).

## 🚀 Quick Start (with the testing image file "fish.pdf" provided)

### Option 1: Jupyter Notebook (Interactive)

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open `demo.ipynb`** 

3. **Run cells sequentially**:
   - Cell 0: Import dependencies
   - Cell 1: PDF processing setup
   - Cell 4-5: Initialize ViT model and analyzer
   - Cell 6: Visualize attention on an image
<!-- 
4. **Example visualization**:
   ```python
   from PIL import Image
   from demo import LocalViTAnalyzer  # After running notebook cells
   
   # Load image
   img = Image.open("your_image.jpg").convert("RGB")
   
   # Initialize analyzer
   analyzer = LocalViTAnalyzer(model_name_or_path="vit")
   
   # Generate attention heatmap
   heatmap_arr, top5 = analyzer.analyze_pil(
       img, 
       layer=-1,      # Last layer
       head=None,     # Average over all heads
       mode="rollout" # Attention rollout mode
   )
   ```

### Option 2: FastAPI Server

1. **Start the API server**:
   ```bash
   # If you have a separate main.py file:
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   
   # Or from Jupyter notebook, add this cell:
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)
   ```

2. **Test the API**:
   ```bash
   # Using curl
   curl http://localhost:8000/docs
   ```

3. **Access interactive API documentation**:
   - Open your browser and navigate to `http://localhost:8000/docs`
   - Use the Swagger UI to test endpoints interactively -->

## 📚 Usage

### Visualization Modes

VAV supports three visualization modes:

1. **`cls`** - CLS token attention for a specific layer
   ```python
   heatmap, top5 = analyzer.analyze_pil(img, layer=-1, head=None, mode="cls")
   ```

2. **`rollout`** - Attention rollout across all layers
   ```python
   heatmap, top5 = analyzer.analyze_pil(img, mode="rollout")
   ```

3. **`all_layers`** - Average attention over all layers
   ```python
   heatmap, top5 = analyzer.analyze_pil(img, mode="all_layers")
   ```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layer` | `int` | `-1` | Layer index (0-based). `-1` = last layer. For 12-layer ViT: 0-11 |
| `head` | `int \| None` | `None` | Attention head index. `None` = average over all heads |
| `mode` | `str` | `"cls"` | Visualization mode: `"cls"`, `"rollout"`, or `"all_layers"` |

<!-- ## 🔌 API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### POST `/api/analyze-image`

Analyze a single image and return attention heatmap with top-5 predictions.

**Request Body:**
```json
{
  "base64Data": "iVBORw0KGgoAAAANSUhEUgAA...",
  "layer": -1,
  "head": null,
  "mode": "rollout"
}
```

**Response:**
```json
{
  "success": true,
  "heatmaps": [
    {
      "page": 1,
      "heatmap": [[0.12, 0.45, ...], [0.23, 0.67, ...], ...],
      "top5": [
        [1, 0.9993, "goldfish, Carassius auratus"],
        [973, 0.0001, "coral reef"],
        ...
      ]
    }
  ],
  "pageCount": 1
}
```

#### POST `/api/analyze-pdf`

Analyze a PDF document and return attention heatmaps for each page.

**Request Body:**
```json
{
  "base64Data": "JVBERi0xLjQKJeLjz9MKMy...",
  "layer": -1,
  "head": null,
  "mode": "all_layers"
}
```

**Response:**
```json
{
  "success": true,
  "heatmaps": [
    {
      "page": 1,
      "heatmap": [[...], ...],
      "top5": [[...], ...]
    },
    {
      "page": 2,
      "heatmap": [[...], ...],
      "top5": [[...], ...]
    }
  ],
  "pageCount": 2,
  "mode": "all_layers"
}
```

### Example API Calls

#### Using cURL

**Analyze Image:**
```bash
BASE64=$(base64 -w0 path/to/image.jpg)
curl -X POST http://localhost:8000/api/analyze-image \
  -H "Content-Type: application/json" \
  -d "{\"base64Data\": \"$BASE64\", \"mode\": \"rollout\"}"
```

**Analyze PDF:**
```bash
BASE64=$(base64 -w0 path/to/document.pdf)
curl -X POST http://localhost:8000/api/analyze-pdf \
  -H "Content-Type: application/json" \
  -d "{\"base64Data\": \"$BASE64\", \"mode\": \"all_layers\"}"
```

#### Using Python

```python
import base64
import requests

# Analyze image
with open("image.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/api/analyze-image",
    json={
        "base64Data": b64,
        "layer": -1,
        "head": None,
        "mode": "rollout"
    }
)

data = response.json()
heatmap = data["heatmaps"][0]["heatmap"]
top5 = data["heatmaps"][0]["top5"]

print("Top 5 predictions:")
for idx, prob, name in top5:
    print(f"  {idx}: {prob:.4f} — {name}")
```

## ⚙️ Configuration

### Model Path

By default, VAV looks for the model in the `vit/` directory. To use a different path:

```python
analyzer = LocalViTAnalyzer(model_name_or_path="path/to/your/model")
```

### Device Selection

The model automatically uses GPU if available (CUDA), otherwise falls back to CPU. To force CPU:

```python
import torch
device = torch.device("cpu")
# Modify the ViT class to use this device
```

### Image Preprocessing

Images are automatically resized to the model's expected input size (typically 224×224 for ViT-base). The processor handles:
- Resizing
- Normalization
- Tensor conversion

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **`FileNotFoundError` or "Cannot load from vit"** | Ensure you've downloaded the model weights to the `vit/` directory. Verify it contains `config.json`, `preprocessor_config.json`, and model weights file. |
| **PDF returns 503 error** | Install a PDF backend: `pip install pymupdf` (recommended) or `pip install pdf2image` (requires poppler system package). |
| **Out of memory (OOM) errors** | Use a smaller ViT model, reduce image resolution before processing, or process images in smaller batches. |
| **CORS errors in browser** | The FastAPI app includes CORS middleware. Ensure you're making requests to the correct host and port. |
| **Wrong heatmap for a layer** | For `mode="cls"`, use `layer` values 0-11 (for 12-layer ViT). Use `head` values 0-11 to visualize specific attention heads. |
| **Model loading is slow** | First-time loading downloads tokenizer files. Subsequent loads are faster. Consider using `torch.compile()` for faster inference (PyTorch 2.0+). |
| **Import errors** | Ensure all dependencies are installed: `pip install -r requirements.txt` | -->

## 📖 References

- **Attention Rollout**: Abnar, S. & Zuidema, W. (2020). "Quantifying Attention Flow in Transformers." *Proceedings of ACL*.
- **Vision Transformer**: Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *NeurIPS*.
- **Hugging Face Transformers**: [Documentation](https://huggingface.co/docs/transformers)
- **ViT Model Card**: [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Hugging Face for the excellent Transformers library
- PyTorch team for the deep learning framework
- The vision transformer research community

---

<div align="center">

**Made with ❤️ for the ML research community**

[Report Bug](https://github.com/yourusername/VAV/issues) • [Request Feature](https://github.com/yourusername/VAV/issues) • [Documentation](https://github.com/yourusername/VAV/wiki)

</div>

