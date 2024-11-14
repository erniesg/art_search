from fasthtml.common import *
from search import initialize_search, search_artworks
from pathlib import Path

# Initialize the search system at startup
IMAGE_DIR = "data/images"
XLSX_PATH = "data/metadata.xlsx"
initialize_search(IMAGE_DIR, XLSX_PATH)

app = FastHTML()

# Add route to serve images
@app.get("/images/{filename:path}")
async def serve_image(filename: str):
    image_path = Path.cwd() / "data/images" / filename
    if image_path.exists():
        return FileResponse(image_path)
    return "Image not found", 404

@app.get("/")
def home():
    return Main(
        H1('Artwork Search'),
        Form(
            Input(type="text", name="query", placeholder="Search artworks..."),
            Button("Search", type="submit"),
            method="post",
            action="/search"
        ),
        Div(id="results")
    )

@app.post("/search")
def search(query: str):
    results = search_artworks(query=query, limit=9)
    
    if not results:
        return P("No results found")
    
    gallery_items = []
    for r in results:
        # Convert full path to relative URL
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

# Add some CSS for the gallery
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
    }
    input[type="text"] {
        width: 100%;
        padding: 0.5rem;
        margin-right: 1rem;
    }
""")

app.hdrs += (css,)

serve()