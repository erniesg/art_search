from fasthtml.common import *
from search import initialize_search, search_artworks
from pathlib import Path
import tempfile

# Initialize the search system at startup
IMAGE_DIR = "data/images"
XLSX_PATH = "data/metadata.xlsx"
initialize_search(IMAGE_DIR, XLSX_PATH)

app = FastHTML()

# Add route to serve images
@app.get("/images/{filename:path}")
async def serve_image(filename: str):
    image_path = Path("data/images") / filename
    if image_path.exists():
        return FileResponse(image_path)
    return "Image not found", 404

@app.get("/")
def home():
    return Main(
        H1('Artwork Search'),
        Form(
            Div(
                Input(type="text", name="query", placeholder="Search by text..."),
                cls="search-input"
            ),
            Div(
                Input(type="file", name="image", accept="image/*"),
                cls="image-input"
            ),
            Button("Search", type="submit"),
            method="post",
            action="/search",
            enctype="multipart/form-data"
        ),
        Div(id="results")
    )

@app.post("/search")
async def search(query: str = None, image: UploadFile = None):
    if not query and not image:
        return P("Please provide either a text query or an image")
    
    # Only process image if it was actually uploaded (has a filename)
    if image and image.filename:
        # Save uploaded image to temp file and search
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            content = await image.read()
            tmp.write(content)
            results = search_artworks(image_path=tmp.name, limit=9)
            Path(tmp.name).unlink()  # Clean up temp file
    else:
        # Text search
        results = search_artworks(query=query, limit=8)
    
    if not results:
        return P("No results found")
    
    gallery_items = []
    for r in results:
        image_filename = Path(r.image_uri).name
        image_url = f"/images/{image_filename}"
        
        gallery_items.append(
            Article(
                Img(src=image_url, alt=r.title),
                H3(r.title),
                P(f"Artist: {r.artist}"),
                P(f"Date: {r.date}"),
                P(f"Medium: {r.medium}"),
                cls="gallery-item"
            )
        )
    
    return Div(
        *gallery_items,
        cls="gallery"
    )

# Add CSS for the gallery and form
css = Style("""
    .gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 2rem;
        padding: 2rem;
    }
    .gallery-item {
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 8px;
    }
    .gallery-item img {
        width: 100%;
        height: 300px;
        object-fit: cover;
        border-radius: 4px;
    }
    form {
        padding: 2rem;
        max-width: 600px;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .search-input, .image-input {
        width: 100%;
        padding: 0.5rem;
    }
    button {
        margin-top: 1rem;
    }
""")

app.hdrs += (css,)

serve()