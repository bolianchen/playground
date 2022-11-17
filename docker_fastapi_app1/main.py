from typing import Union
from fastapi import FastAPI
app = FastAPI()

@app.get("/", summary="Get the root")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    """
    Get information according to the input item_id
    """
    return {"item_id": item_id, "q": q}

@app.get('/tokenizer/{text}')
def tokenized(text: str):
    """
    Get the tokenized result of the input text
    """
    return text.split()
