"""
This API takes a text input and analyzes it for toxic, obscene, threataning, insulting, 
and hate speach. Later versions will include a more accurate model and the ability to detect
suicidality. 
"""
import re
import string
from fastapi import BaseModel, FastAPI
import sklearn
from sklearn.externals import joblib


model = joblib.load('compressed_pipeline.pkl')
printable = set(string.printable)

app = FastAPI()


@app.post("/predict/{text}")
async def predict(text: str):
    """
    This is the enpoint function that reads in the text and uses unpickled model to 
    make predictions.
    """
    # clean text to remove characters and metadata that may interfere with accuracy of model
    text = clean_text(text)
    # model requires text be put into a list to function
    text = [text]
    # use unpickled model to make prediction
    prediction = model.predict(text).tolist()[0]
    # identify labels to process predictions
    labels = ["toxic", "severe toxic", "obscene", "threat", "insult", "identity hate"] 
    # process predictions to match labels
    results = [label for i, label in enumerate(labels) if prediction[i]]
    # get prediction probabalities for each possible label
    pred_probabilities = model.predict_proba(text).tolist()[0]
    # get prediction probabilities just for labels that apply
    probabilty = [pred_probability for i, pred_probability in enumerate(pred_probabilities) if prediction[i]]
    # for text that is not labeled provide a result to indicate
    if len(results) == 0:
        results = ["No toxic or offensive content detected"]
    # return results
    return {"text": text, "prediction": results, "probability": probabilty}


def clean_text(x):
    """
    Function for cleaning text to remove characters, user identification, and non-printable 
    characters that can interfer with the model's ability to make accurate predictions.
    """
    # remove newline characters
    x = re.sub('\\n',' ',x)
    # remove return characters
    x = re.sub('\\r',' ',x)
    # remove leading and trailing white space
    x = x.strip()
    # remove any text starting with User... 
    x = re.sub("\[\[User.*", ' ', x)
    # remove IP addresses or user IDs
    x = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", ' ', x)
    # remove URLs
    x = re.sub("(http://.*?\s)|(http://.*)", ' ', x)
    # remove non_printable characters eg unicode
    x = "".join(list(filter(lambda c: c in printable, x)))
    return x