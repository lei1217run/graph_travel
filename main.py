from fastapi import FastAPI
from typing import List, Optional
from fastapi import Query
from xiyou.compare_service import CompareChapterQuery, CompareBatchQuery, compare_chapter, compare_overview, compare_batch, CompareBatchBooksQuery, compare_overview_multi, compare_batch_books
from xiyou.config import load_settings, list_target_books

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/compare/chapter")
def api_compare_chapter(book: str, chapter_id: str, templates: List[str] = Query(default=["relations_plus","events_relations"]), model_name: Optional[str] = None, confidence_min: float = 0.0, include_events: bool = True):
    q = CompareChapterQuery(book=book, chapter_id=chapter_id, templates=templates, model_name=model_name, confidence_min=confidence_min, include_events=include_events)
    return compare_chapter(q)

@app.get("/compare/overview")
def api_compare_overview(book: str):
    return compare_overview(book)

@app.post("/compare/batch")
def api_compare_batch(payload: CompareBatchQuery):
    return compare_batch(payload)

@app.get("/compare/overview_multi")
def api_compare_overview_multi(books: List[str] = Query(default=[])):
    if not books:
        settings = load_settings()
        books = list_target_books(settings)
    return compare_overview_multi(books)

@app.post("/compare/batch_books")
def api_compare_batch_books(payload: CompareBatchBooksQuery):
    return compare_batch_books(payload)
