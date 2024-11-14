# Art Search Engine

A visual and text-based artwork search engine powered by LanceDB and CLIP embeddings. This application allows users to search through an art collection using either text descriptions or similar images.

## Features

- 🔍 Text-based artwork search
- 🖼️ Image-based similarity search
- 🎨 Combined gallery view of search results
- ✅ Automatic validation of image-metadata pairs
- 📝 Comprehensive logging system

## Prerequisites

- Python 3.12.5
- Virtual environment management tool (pyenv recommended)

## Installation

1. **Set up Python Environment**
```bash
# Using pyenv
pyenv install 3.12.5
pyenv local 3.12.5
pyenv virtualenv art_search
pyenv activate art_search
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## Data Setup

1. Create the following directory structure:
```
.
├── data/
│   ├── images/         # Place artwork images here
│   ├── metadata.xlsx   # Place metadata file here
│   └── logs/          # Validation logs (auto-generated)
```

2. **Metadata Requirements**
   Your `metadata.xlsx` file should include these columns:
   - Record ID
   - Accession No.
   - Artist/Maker
   - Title
   - Dating
   - Medium
   - Dimensions
   - Geo. Reference
   - Credit Line

## Running the Application

1. **Initialize the Database**
```bash
# Build the search index and validate data
python search.py
```

2. **Start the Web Server**
```bash
python app.py
```

3. **Access the Interface**
   - Open your web browser
   - Navigate to `http://localhost:5001`

## Usage

### Search Methods

1. **Text Search**
   - Enter descriptive text in the search box
   - Click "Search"

2. **Image Search**
   - Click "Choose File" in the image upload section
   - Select a reference image
   - Click "Search"

### Validation Logs

The system generates several log files in `data/logs/`:
- `artwork_validation.log`: General validation information
- `missing_images.txt`: Artworks lacking image files
- `missing_metadata.txt`: Images without matching metadata
- `multiple_images.txt`: Artworks with multiple images
- `valid_pairs.txt`: Successfully matched image-metadata pairs

## Development

### Key Components

- `app.py`: FastHTML web application
- `search.py`: Core search engine implementation
- `requirements.txt`: Python package dependencies

## Dependencies

Key libraries used:
- LanceDB: Vector database
- FastHTML: Web framework
- CLIP: Neural network for image embeddings
- Pillow: Image processing
- pandas: Data manipulation