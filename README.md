# Photo Mosaic Generator

Create stunning photo mosaics from your screenshot collection! This tool intelligently arranges all your images to recreate a target photo using color-matching algorithms.

## ✨ Features

- 🎯 **Guaranteed Image Inclusion** - Every image is used at least once
- 🎨 **Intelligent Color Matching** - K-d tree algorithms match images to target colors
- 📅 **Date Filtering** - Filter images by date range for themed mosaics
- 🖼️ **Target Image Overlay** - Blend original image over mosaic for clarity
- 📏 **Automatic File Size Control** - Keeps output under 500MB
- 🔄 **Batch Processing** - Process multiple targets with date-based filenames
- 📱 **Apple Photo Support** - Reads EXIF data from iPhone/Mac photos
- ⚙️ **Automatic Optimization** - Canvas size and scaling calculated automatically

## 🚀 Quick Start

### Single Mosaic Mode

1. **Place your target image** in the project root (e.g., `target.jpg`)

2. **Add your screenshots** to the `screenshots` folder

3. **Configure** `mosaic_maker.py`:
   ```python
   TARGET_IMAGE = "target.jpg"
   IMAGE_FOLDER = "screenshots"
   START_DATE = None  # Or "2023-02-10"
   END_DATE = None    # Or "2023-11-07"
   OVERLAY_OPACITY = 0.3
   ```

4. **Run:**
   ```bash
   python mosaic_maker.py
   ```

### Batch Processing Mode (Recommended!)

1. **Create folder structure:**
   ```
   targets/         # Put your target images here
   screenshots/     # Your source photos
   outputs/         # Mosaics will be saved here (auto-created)
   ```

2. **Name target images** using date format:
   ```
   targets/2023-01-01 to 2023-12-31.jpg
   targets/2024-01-01 to 2024-06-30.jpg
   ```

3. **Run batch processor:**
   ```bash
   python batch_mosaic_maker.py
   ```

4. **Find mosaics** in `outputs/` folder:
   ```
   outputs/2023-01-01 to 2023-12-31 Mosaic.png
   outputs/2024-01-01 to 2024-06-30 Mosaic.png
   ```

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/collage-generator.git
cd collage-generator

# Install dependencies
pip install -r requirements.txt
```

## 📋 Requirements

- Python 3.7+
- PIL/Pillow (with EXIF support)
- NumPy
- SciPy
- tqdm
- colorama

See `requirements.txt` for exact versions.

## ⚙️ Configuration

### Basic Settings

| Option | Description | Example |
|--------|-------------|---------|
| `TARGET_IMAGE` | Image to recreate as mosaic | `"target.jpg"` |
| `IMAGE_FOLDER` | Source images folder | `"screenshots"` |
| `START_DATE` | Filter start date (optional) | `"2023-02-10"` or `None` |
| `END_DATE` | Filter end date (optional) | `"2023-11-07"` or `None` |
| `OVERLAY_OPACITY` | Target overlay opacity | `0.0` to `1.0` |
| `OUTPUT_FILE` | Output filename | `"mosaic_output.png"` |

### Overlay Opacity Guide

```python
OVERLAY_OPACITY = 0.0   # No overlay (pure mosaic)
OVERLAY_OPACITY = 0.3   # Subtle (recommended)
OVERLAY_OPACITY = 0.5   # Medium
OVERLAY_OPACITY = 1.0   # Full overlay
```

## 🔧 How It Works

### Three-Phase Placement System

#### Phase 1: Color-Matched Placement
- Analyzes target image into a color grid
- Places images in spiral order from center
- Each cell gets best color-matching image
- Creates core mosaic structure

#### Phase 2: Fill Remaining Gaps
- Places unused images in available spaces
- Size-based packing for efficiency
- Ensures every image appears

#### Phase 3: Canvas Expansion
- Expands canvas if images remain
- Balanced width/height expansion
- Optional image downscaling
- Guarantees 100% image inclusion

### Automatic File Size Control

Output files are automatically kept under 500MB:
- Estimates file size: `(width × height × 4 bytes) / 1MB`
- If over 500MB, downscales intelligently
- Maintains aspect ratio
- Shows actual file size after save

### Apple Photo Date Extraction

Enhanced EXIF support for Apple photos:
1. **DateTimeOriginal** - When photo was taken (best)
2. **DateTime** - General datetime
3. **DateTimeDigitized** - When digitized
4. **File modification time** - Fallback

Works with iPhone, iPad, and Mac photos!

## 📁 Project Structure

```
collage-generator/
├── mosaic_maker.py              # Single mosaic mode
├── batch_mosaic_maker.py        # Batch processing mode
├── requirements.txt
├── README.md
│
├── scripts/
│   ├── analyze_images_in_folder.py  # Image analysis & EXIF extraction
│   ├── assemble_mosaic.py           # Main mosaic assembly
│   ├── canvas_utils.py              # Canvas management
│   ├── color_matching.py            # K-d tree color matching
│   ├── image_filters.py             # Date filtering
│   ├── image_loader.py              # Image loading & scaling
│   ├── placement.py                 # Placement algorithms
│   ├── target_analyzer.py           # Target analysis
│   └── get_file_date.py            # EXIF date extraction
│
├── screenshots/                # Your source images
├── targets/                    # Target images for batch mode
├── outputs/                    # Generated mosaics (auto-created)
├── target.jpg                  # Single mode target
└── IMAGES_INFO.json           # Cached metadata
```

## 💡 Tips & Tricks

### Getting Best Results

1. **More images = better** - Use 500+ for detailed mosaics
2. **Color variety** - Diverse colors improve matching
3. **Quality targets** - Clear, high-contrast images work best
4. **Start with 0.3 overlay** - Adjust based on preference
5. **Delete IMAGES_INFO.json** - If you add/remove images

### Batch Processing Tips

1. **Consistent naming** - Use exact format: `YYYY-MM-DD to YYYY-MM-DD.jpg`
2. **Date order** - Script auto-corrects if dates are swapped
3. **Reuse analysis** - Screenshots analyzed once, reused for all targets
4. **Parallel processing** - All targets processed sequentially

### File Size Management

Output automatically kept under 500MB:
- Large collections (1000+ images) may produce huge canvases
- Auto-downscaling maintains visual quality
- Original resolution preserved when possible
- PNG format for best quality

## 🐛 Troubleshooting

**"No images to process. Exiting."**
- Check date range (may be swapped)
- Set dates to `None` to include all
- Verify images have EXIF data

**"Skipped X images without date information"**
- Some images lack EXIF metadata
- Will use file modification time as fallback
- Check IMAGES_INFO.json for dates

**"Filename doesn't match expected format"** (Batch mode)
- Use exact format: `2023-01-01 to 2023-12-31.jpg`
- Check for extra spaces
- File extension must be .jpg, .jpeg, or .png

**Images appear very small**
- Normal! Auto-scales to fit all images
- Limited to 500MB output size
- View at 100% zoom for detail

**File size still over 500MB**
- Rare edge case
- Try manually reducing `max_canvas_size` in code
- Consider splitting into multiple date ranges

## 📊 Performance

- **Analysis**: ~1-2 min for 1000 images
- **Assembly**: ~2-5 min for 1000 images
- **Memory**: ~2-4GB RAM
- **Output**: <500MB PNG (auto-scaled)

## 🎯 Examples

### Single Mosaic
```bash
# Edit mosaic_maker.py
TARGET_IMAGE = "my_face.jpg"
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

# Run
python mosaic_maker.py
```

### Batch Processing
```bash
# Create targets
targets/2023-01-01 to 2023-03-31.jpg  # Q1 2023
targets/2023-04-01 to 2023-06-30.jpg  # Q2 2023
targets/2023-07-01 to 2023-09-30.jpg  # Q3 2023
targets/2023-10-01 to 2023-12-31.jpg  # Q4 2023

# Run batch
python batch_mosaic_maker.py

# Results in outputs/
outputs/2023-01-01 to 2023-03-31 Mosaic.png
outputs/2023-04-01 to 2023-06-30 Mosaic.png
outputs/2023-07-01 to 2023-09-30 Mosaic.png
outputs/2023-10-01 to 2023-12-31 Mosaic.png
```

## 🔄 Workflow

### For Yearly Mosaics

1. Export all iPhone photos to `screenshots/`
2. Create target images for each year
3. Name them: `2020-01-01 to 2020-12-31.jpg`, etc.
4. Run `python batch_mosaic_maker.py`
5. Get yearly mosaics in `outputs/`!

### For Event Mosaics

1. Name target after event date range
2. Example: `2023-07-01 to 2023-07-07.jpg` (vacation week)
3. Mosaic uses only photos from that week
4. Perfect for trips, events, projects!

## 🤝 Contributing

Contributions welcome!
- Report bugs
- Suggest features
- Submit pull requests
- Share your mosaics!

## 📜 License

MIT License - Open source and free to use!

## 🙏 Credits

Created with ❤️ for preserving memories through creative photo collages.

---

**Happy Mosaic Making!** 🎨📸

*Now with batch processing and Apple photo support!*
