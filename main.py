"""
Main
"""
import re
import string
from fastapi import FastAPI
import sklearn
from sklearn.externals import joblib
from pydantic import BaseModel


model = joblib.load('compressed_pipeline.pkl')
printable = set(string.printable)

app = FastAPI()

class Data(BaseModel):
    text: str


@app.post("/predict/")
async def predict(data: Data):
    """
    This API accepts string based requests and cleans the text.

    The following are removed:
    - Newline characters
    - Return characters
    - Leading and trailing white spaces
    - Usernames if indicated with the term 'User'
    - IP addresses and user IDs
    - Non-printable characters e.g. unicode

    Request URL:
    https://fast-api-test2.appspot.com/predict/

    Example Request Body:

    {"text": "string"}

    **Model**: Term frequency-inverse document frequency(Tfidf) and Logistic Regression


    The model is used to determine if the text contains toxic or offensive content.
    The possible labels are toxic, severe toxic, obscene, threat, insult, and identity
    hate. If no toxic or obscene content is detected the result will be, "No toxic or
    offensive content detected."

    The API returns the original text, any classifications that apply, and the predicted
    probability of the labels given.

    The current model is for testing purposes only, and will be replaced by a more robust natural language processing
    model that has a higher positive recall score for each class.Recall demonstrates how effectively a model identifies
    true positives, or in this case, toxicity. A low recall score would indicate a substantial amount of toxic content
    is classified as non-toxic. Next versions of the model will work to minimize false negatives to ensure potentially
    dangerous content is not missed.

    |**Class**    |**Recall Score**|
    |:-----------:|:--------------:|
    |Toxic        |.68             |
    |Severe Toxic |.29             |
    |Obscene      |.71             |
    |Threat       |.27             |
    |Insult       |.56             |
    |Identity Hate|.26             |
    """


    data_dict = data.dict()

    # clean text to remove characters and metadata that may interfere with accuracy of model
    text = clean_text(data.text)

    # model requires text be put into a list to function
    text = [text]

    # use unpickled model to make prediction
    prediction = model.predict(text).tolist()[0]

    # identify labels to process predictions
    labels = ["toxic", "severe toxic", "obscene", "threat", "insult", "identity hate"]

    # process predictions to match labels
    results = [label for i, label in enumerate(labels) if prediction[i]]

    # get prediction probabalities for each possible label
    pred_probs = model.predict_proba(text).tolist()[0]

    # get prediction probabilities just for labels that apply
    probabilty = [round(pred_prob, 2) for i, pred_prob in enumerate(pred_probs) if prediction[i]]

    # for text that is not labeled provide a result to indicate no class identification
    if len(results) == 0:
        results = ["No toxic or offensive content detected"]

    # return results
    return {"text": text, "prediction": results, "probability": probabilty}

@app.post("/classify")
async def classify(data: Data):
    """
    This is the same as the predict method, with some formatting differences.
    New output will be in the following format:

    {
    "text": "stupid you're the worst",
    "results": [
    {
      "prediction": "toxic",
      "probability": 1
    },
    {
      "prediction": "insult",
      "probability": 0.92}]}
     Recall is being used as an accuracy metric and the scores below reflect the percentage
     of positively identied instances of these classes.
    """
    data_dict = data.dict()

    # clean text to remove characters and metadata that may interfere with accuracy of model
    text = clean_text(data.text)

    # model requires text be put into a list to function
    text = [text]

    # FORMATTING CHANGE: removes string from array for output
    display_text = " ".join(str(i) for i in text)

    # use unpickled model to make prediction
    prediction = model.predict(text).tolist()[0]

    # identify labels to process predictions
    labels = ['toxic', 'severe toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # process predictions to match labels
    results = [label for i, label in enumerate(labels) if prediction[i]]

    # get prediction probabalities for each possible label
    pred_probs = model.predict_proba(text).tolist()[0]

    # get prediction probabilities just for labels that apply
    probability = [round(pred_prob, 2) for i, pred_prob in enumerate(pred_probs) if prediction[i]]

    # FORMATTING CHANGE: creates 1 array that contains prediction and probability for each class detected
    display_results = [{'prediction':label, 'probability':proba} for label,proba in zip(results, probability)]

    # for text that is not labeled provide a result to indicate no class identification
    if len(display_results) == 0:
        display_results = ["No toxic or offensive content detected"]

    # return results
    return {"text": display_text, "results": display_results}


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
