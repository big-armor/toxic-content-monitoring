"""
Main
"""
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from model import ToxicClassifierModel, vectors
from preprocessor import normalize_comment, clean_text



model = ToxicClassifierModel()
model.load_state_dict(torch.load("data/TCM_2.pt"))
model.eval()



app = FastAPI()

class Data(BaseModel):
    text: str


@app.post("/predict/")
async def predict(data: Data):
    """
    This API accepts string based requests and cleans the text before running it through the prediction model.

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
    
    The possible labels are:
    - toxic
    - severe 
    - toxic 
    - obscene 
    - threat 
    - insult 
    - identity hate

    The API returns the original text, any classifications that apply, and the predicted
    probability of the labels given.

    The current model is for testing purposes only, and will be replaced by a more robust natural language processing
    model that has a higher positive recall score for each class. Recall demonstrates how effectively a model identifies
    true positives, or in this case, toxicity. A low recall score would indicate a substantial amount of toxic content
    is classified as non-toxic. Next versions of the model will work to minimize false negatives to ensure potentially
    dangerous content is not missed.

    """


    data_dict = data.dict()

    # clean text to remove characters and metadata that may interfere with accuracy of model
    text = clean_text(data.text)

    # normalize comment
    normalized_words = normalize_comment(text)

    var_vectorized_sentence = Variable(torch.nn.utils.rnn.pad_sequence([vectors[normalized_words]]).permute(1,0,2)).float()

    output = model(var_vectorized_sentence)

    predProb = output.data.numpy().tolist()[0]

    threshold = 0.5
    preds = []
    
    for p in predProb:
        if p >= threshold:
            preds.append(True)
        else: 
            preds.append(False)
    

    # identify labels to process predictions
    labels = ['toxic', 'severe toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # FORMATTING CHANGE: creates 1 array that contains prediction and probability for each class detected
    display_results = {label:{'prediction':pred, 'probability':round(proba, 5)} for label,pred,proba in zip(labels, preds, predProb)}

    # return results
    return {"text": text, "results": display_results}


# Customize documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Harm Detection API",
        version="1.0.6",
        description="This API is used to detect toxicity of varying degrees in text. "\
                    "To read the documentation for post request click the **POST** button. "\
                    "Clicking the **Try it out** button will allow you to test requests."  ,
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi