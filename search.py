# image_search.py
import lancedb
import pandas as pd
from pathlib import Path
from PIL import Image
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry
import logging
from collections import defaultdict
import re
import openpyxl
import time
import os
from datetime import datetime
from humanize import naturalsize  # for human-readable file sizes
import torch  # Add to imports at top
from tqdm import tqdm  # Add to imports
import numpy as np

# Create logs directory
log_dir = Path("data/logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Simple log file paths without timestamps
log_file = log_dir / "artwork_validation.log"
missing_images_file = log_dir / "missing_images.txt"
missing_metadata_file = log_dir / "missing_metadata.txt"
multiple_images_file = log_dir / "multiple_images.txt"
valid_pairs_file = log_dir / "valid_pairs.txt"

# Update logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # 'w' mode overwrites the file
        logging.StreamHandler()
    ]
)

# 1. Setup CLIP embedding function
registry = EmbeddingFunctionRegistry.get_instance()
clip = registry.get("open-clip").create()

# 2. Define data model with proper embedding function
class Artwork(LanceModel):
    vector: Vector(clip.ndims()) = clip.VectorField()  # Let CLIP define dimensions
    image_uri: str = clip.SourceField()  # Automatically use CLIP for encoding
    record_id: str
    accession_no: str
    artist: str
    title: str
    date: str
    medium: str
    dimensions: str
    location: str
    credit: str
    
    @property
    def image(self):
        """Convenience property to open and return the image"""
        return Image.open(self.image_uri)

# 3. Data loading and processing
def validate_data(image_dir: str, xlsx_path: str, save_log: bool = True):
    """Validates image files against metadata and logs discrepancies"""
    try:
        df = pd.read_excel(xlsx_path)
        df = clean_metadata_df(df)
        accession_numbers = set(df['accession_no'].astype(str))
        logging.info(f"Found {len(accession_numbers)} unique accession numbers in metadata")
    except Exception as e:
        logging.error(f"Error loading Excel file: {e}")
        raise

    # Scan image directory
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.jpg'))
    logging.info(f"Found {len(image_files)} image files")

    # Create mapping of accession numbers to image files
    image_mapping = defaultdict(list)
    unmatched_images = []
    
    # Optimize the matching process
    logging.info("Matching images to metadata...")
    acc_patterns = {acc: re.compile(re.escape(str(acc))) for acc in accession_numbers}
    
    # Use tqdm for progress bar
    for img_path in tqdm(image_files, desc="Processing images"):
        filename = img_path.stem
        matched = False
        
        # Try to match accession number in filename
        for acc_num, pattern in acc_patterns.items():
            if pattern.search(filename):
                image_mapping[acc_num].append(img_path)
                matched = True
                break
                
        if not matched:
            unmatched_images.append(img_path)

    # Analyze results
    valid_pairs = []
    multiple_images = {}
    missing_images = set()
    
    # Process matches and multiples
    for acc_num, images in image_mapping.items():
        if len(images) > 1:
            multiple_images[acc_num] = images
        valid_pairs.append((acc_num, images[0]))  # Use first image for valid pairs

    # Check for missing images
    missing_images = accession_numbers - set(image_mapping.keys())

    if save_log:
        # Save missing images (in metadata but no image)
        with open(missing_images_file, 'w') as f:
            f.write("ARTWORKS WITH MISSING IMAGES:\n")
            f.write("-" * 50 + "\n")
            for acc_num in missing_images:
                artwork_info = df[df['accession_no'] == acc_num].iloc[0]
                f.write(f"Accession: {acc_num}\n")
                f.write(f"Title: {artwork_info['title']}\n")
                f.write(f"Artist: {artwork_info['artist']}\n")
                f.write("-" * 50 + "\n")

        # Save images with missing metadata
        with open(missing_metadata_file, 'w') as f:
            f.write("IMAGES WITH NO MATCHING METADATA:\n")
            f.write("-" * 50 + "\n")
            for img_path in unmatched_images:
                f.write(f"Image file: {img_path.name}\n")
                f.write("-" * 50 + "\n")

        # Save multiple images info
        with open(multiple_images_file, 'w') as f:
            f.write("ARTWORKS WITH MULTIPLE IMAGES:\n")
            f.write("-" * 50 + "\n")
            for acc_num, images in multiple_images.items():
                artwork_info = df[df['accession_no'] == acc_num].iloc[0]
                f.write(f"Accession: {acc_num}\n")
                f.write(f"Title: {artwork_info['title']}\n")
                f.write(f"Artist: {artwork_info['artist']}\n")
                f.write("Images:\n")
                for img in images:
                    f.write(f"- {img.name}\n")
                f.write("-" * 50 + "\n")

        # Save valid pairs
        with open(valid_pairs_file, 'w') as f:
            f.write("VALID ARTWORK-IMAGE PAIRS:\n")
            f.write("-" * 50 + "\n")
            for acc_num, img_path in valid_pairs:
                f.write(f"{acc_num}: {img_path.name}\n")

    # Log summary
    logging.info("\nValidation Summary:")
    logging.info(f"Valid pairs: {len(valid_pairs)}")
    logging.info(f"Multiple images: {len(multiple_images)}")
    logging.info(f"Missing images: {len(missing_images)}")
    logging.info(f"Images without metadata: {len(unmatched_images)}")

    return valid_pairs, multiple_images, missing_images, unmatched_images

def clean_metadata_df(df):
    """Clean and prepare the metadata DataFrame"""
    # Select and rename relevant columns
    df = df[[
        'Record ID',
        'Accession No.',
        'Artist/Maker',
        'Title',
        'Dating',
        'Medium',
        'Dimensions',
        'Geo. Reference',
        'Credit Line'
    ]].copy()
    
    # Rename columns to simpler names
    df = df.rename(columns={
        'Record ID': 'record_id',
        'Accession No.': 'accession_no',
        'Artist/Maker': 'artist',
        'Title': 'title',
        'Dating': 'date',
        'Medium': 'medium',
        'Dimensions': 'dimensions',
        'Geo. Reference': 'location',
        'Credit Line': 'credit'
    })
    
    # Clean accession numbers (remove any whitespace and convert to string)
    df['accession_no'] = df['accession_no'].astype(str).str.strip()
    
    # Remove any empty rows
    df = df.dropna(subset=['accession_no'])
    
    # Log sample data to verify cleaning
    logging.info("\nSample cleaned metadata:")
    logging.info(df.head().to_string())
    
    return df

def create_or_get_table(image_dir: str, xlsx_path: str, save_log: bool = True):
    try:
        start_time = time.time()
        db = lancedb.connect("~/.lancedb")
        
        if "artworks" in db:
            table = db["artworks"]
            logging.info(f"Loaded existing artworks table with {len(table)} records")
            return table
        
        logging.info("Creating new artworks table...")
        
        # Get valid pairs first
        if log_file.exists() and save_log:
            logging.info("Found existing validation log, skipping validation...")
            valid_pairs = []
            with open(valid_pairs_file, 'r') as f:
                for line in f:
                    if line.startswith("VALID ARTWORK-IMAGE PAIRS:") or line.startswith("-"):
                        continue
                    if line.strip():
                        acc_num, img_path = line.strip().split(": ", 1)
                        valid_pairs.append((acc_num, Path(image_dir) / img_path))
            logging.info(f"Loaded {len(valid_pairs)} valid pairs from validation log")
        else:
            validation_start = time.time()
            valid_pairs, _, _, _ = validate_data(image_dir, xlsx_path, save_log=save_log)
            logging.info(f"Data validation time: {time.time() - validation_start:.2f} seconds")
            logging.info(f"Found {len(valid_pairs)} valid pairs during validation")
        
        if not valid_pairs:
            raise ValueError("No valid artwork-image pairs found!")
            
        # Create a dictionary of accession numbers to image paths
        valid_pairs_dict = {acc_num: str(img_path) for acc_num, img_path in valid_pairs}
        logging.info(f"Created mapping dictionary with {len(valid_pairs_dict)} entries")
        
        # Read and clean metadata
        df = pd.read_excel(xlsx_path)
        df = clean_metadata_df(df)
        logging.info(f"Initial metadata DataFrame size: {len(df)}")
        
        # Filter to only valid pairs and add image_uri
        df = df[df['accession_no'].isin(valid_pairs_dict.keys())].copy()
        logging.info(f"Filtered metadata DataFrame size: {len(df)}")
        
        df['image_uri'] = df['accession_no'].map(valid_pairs_dict)
        logging.info(f"Number of image URIs mapped: {df['image_uri'].notna().sum()}")
        
        # Create table with our schema
        table = db.create_table("artworks", schema=Artwork)
        table.add(df)
        
        logging.info(f"Created table with {len(df)} artworks from {len(valid_pairs)} valid pairs")
        return table
        
    except Exception as e:
        logging.error(f"Error creating/getting table: {e}", exc_info=True)
        raise

# 4. Search function
def search_artworks(query, table, top_k=9):
    """Search artworks using either text or image query"""
    try:
        results = table.search(query).limit(top_k).to_pydantic(Artwork)
        return results
    except Exception as e:
        logging.error(f"Search error: {e}")
        raise

def format_results(results) -> tuple[list[str], str]:
    """Format search results for display"""
    images = []
    captions = []
    for r in results:
        if Path(r.image_uri).exists():
            images.append(str(r.image_uri))
            caption = (
                f"Artist: {r.artist or 'Unknown'}\n"
                f"Title: {r.title or 'Untitled'}\n"
                f"Date: {r.date or 'Unknown'}\n"
                f"Medium: {r.medium or 'Unknown'}\n"
                f"Accession: {r.accession_no or 'Unknown'}"
            )
            captions.append(caption)
    
    return images, "\n\n---\n\n".join(captions) if captions else "No results found"

def process_search(query=None, image_file=None):
    """Process either text or image search input"""
    try:
        if query:
            logging.info(f"Processing text search: {query}")
            results = search_artworks(query, table)
        elif image_file:
            logging.info(f"Processing image search from file: {image_file}")
            query_image = Image.open(image_file)
            results = search_artworks(query_image, table)
        else:
            return []
            
        # Format results for display
        gallery_items = []
        for r in results:
            if Path(r.image_uri).exists():
                gallery_items.append({
                    "image": r.image_uri,
                    "title": r.title,
                    "artist": r.artist,
                    "date": r.date
                })
        return gallery_items
            
    except Exception as e:
        logging.error(f"Search error: {e}", exc_info=True)
        return []

def test_searches(table):
    """Run test searches using both text and image queries"""
    print("\n=== Running Test Searches ===")
    
    # Test text search
    print("\n1. Testing text search for 'pineapple'...")
    text_results = table.search("pineapple").limit(9).to_pydantic(Artwork)
    print(f"Found {len(text_results)} results")
    for i, r in enumerate(text_results, 1):
        print(f"\nResult {i}:")
        print(f"Title: {r.title}")
        print(f"Artist: {r.artist}")
        print(f"Date: {r.date}")
        print(f"Medium: {r.medium}")
    
    # Test image search
    print("\n2. Testing image search with test.jpg...")
    try:
        test_image = Image.open("data/test.jpg")
        image_results = table.search(test_image).limit(9).to_pydantic(Artwork)
        print(f"Found {len(image_results)} results")
        for i, r in enumerate(image_results, 1):
            print(f"\nResult {i}:")
            print(f"Title: {r.title}")
            print(f"Artist: {r.artist}")
            print(f"Date: {r.date}")
            print(f"Medium: {r.medium}")
    except FileNotFoundError:
        print("Test image not found at data/test.jpg")
    except Exception as e:
        print(f"Error during image search: {e}")
    
    print("\n=== Test Searches Complete ===\n")

def main():
    global table
    
    IMAGE_DIR = "/home/erniesg/code/erniesg/lanceart/data/images"
    XLSX_PATH = "/home/erniesg/code/erniesg/lanceart/data/metadata.xlsx"
    
    print("Loading database...")
    table = create_or_get_table(IMAGE_DIR, XLSX_PATH)
    
    # Run test searches before launching interface
    test_searches(table)

if __name__ == "__main__":
    main()