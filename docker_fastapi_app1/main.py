from typing import Union
from fastapi import FastAPI
app = FastAPI()

# greeting at the root
@app.get("/")
def read_root():
    return {"Hello": "World"}

# return item information by using path and query parameters
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# feed a text via a path parameter and return words by splitting the text
@app.get('/tokenizer/{text}')
def tokenized(text: str):
    return text.split()
