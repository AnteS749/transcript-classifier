from fastapi import FastAPI
from pydantic import BaseModel
from classifier import Classifier

app = FastAPI()
clf = Classifier(
    txt_path="data/browse_terms_output.txt",
    json_path="data/browse_terms_output.json"
)

class Input(BaseModel):
    locked_l0: str
    transcript: str

@app.post("/classify")
def classify_transcript(input: Input):
    result = clf.classify(input.transcript, input.locked_l0)
    return result