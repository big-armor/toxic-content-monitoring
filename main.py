"""
Main  
"""
import re
import string
from fastapi import FastAPI
from pydantic import BaseModel
import sklearn
from sklearn.externals import joblib


model = joblib.load('compressed_pipeline.pkl')
printable = set(string.printable)

app = FastAPI()

class Data(BaseModel):
    text: str


@app.post("/predict/")
async def predict(data: Data):
    """
    This API accepts string based requests and cleans the text. 

    Example Request Body:

    {"text": "string"}
    
    The following are removed:
    - Newline characters 
    - Return characters
    - Leading and trailing white spaces
    - Usernames if indicated with the term 'User'
    - IP addresses and user IDs
    - Non-printable characters e.g. unicode

    **Model**: Term frequency-inverse document frequency(Tfidf) and Logistic Regression
    
    
    The model is used to determine if the text contains toxic or offensive content. 
    The possible labels are toxic, severe toxic, obscene, threat, insult, and identity 
    hate. If no toxic or obscene content is detected the result will be, "No toxic or 
    offensive content detected." 

    The API returns the original text, any classifications that apply, and the predicted 
    probability of the labels given.

    The current model is for testing purposes only, and will be replaced by a more robust natural language processing
    model that has a higher positive recall score for each class.

    
    |**Toxic**      |precision    |recall  |f1-score   |support|
    |:----------:|:-------:|:-------:|:-------:|:-------:|
    | negative   | 0.97    | 0.99    | 0.98    | 28927   |
    |  positive    |  0.85   |   0.68  |    0.75 | 2988    |
    |**accuracy**|         |         | 0.96    | 31915   |
    |**macro avg**|    0.91 |   0.83  |    0.86 |   31915 |
    |**weighted avg**|    0.96 |   0.96  |    0.96 |   31915 |

    --------


    |**Severe Toxic**|precision    |recall  |f1-score   |support|
    |:----------:|:-------:|:-------:|:-------:|:-------:|
    | negative      |  0.99   |  1.00    | 1.00   | 31612   |
    |    positive        |0.53     |0.29     |0.38     |  303  |
    |**accuracy**|         |         | 0.99    | 31915   |
    |**macro avg**|    0.76 |   0.65  |    0.69 |   31915 |
    |**weighted avg**|    0.99 |   0.99 |    0.99|   31915 |

    --------


    |**Obscene**   |precision    |recall  |f1-score   |support|
    |:----------:|:-------:|:-------:|:-------:|:-------:|
    | negative     |  0.98   |  0.99      | 0.99     | 30291  |
    |   positive      |0.88     |0.71      |0.79     |  1624 |
    |**accuracy**|         |         | 0.98    | 31915   |
    |**macro avg**|    0.93 |   0.85  |    0.89 |   31915 |
    |**weighted avg**|    0.98 |   0.98 |    0.98|   31915 |


    --------


    |**Threat**    |precision    |recall  |f1-score   |support|
    |:----------:|:-------:|:-------:|:-------:|:-------:|
    | negative     | 1.00    | 1.00   | 1.00    | 31825   |
    |    positive       |  0.63   |   0.27  |    0.38 | 90    |
    |**accuracy**|         |         | 0.96    | 31915   |
    |**macro avg**|    0.81 |   0.63  |    0.69 |   31915 |
    |**weighted avg**|    1.00 |   1.00 |    1.00 |   31915 |


    --------


    |**Insult**  |precision    |recall  |f1-score   |support|
    |:----------:|:-------:|:-------:|:-------:|:-------:|
    |negative       | 0.98    | 0.99    | 0.99    | 30412   |
    |    positive         |  0.76   |   0.56  |    0.65 | 1503    |
    |**accuracy**|         |         | 0.97    | 31915   |
    |**macro avg**|    0.87 |   0.78  |    0.82 |   31915 |
    |**weighted avg**|    0.97 |   0.97  |    0.97 |   31915 |


    --------


    |**Identity Hate**      |precision    |recall  |f1-score   |support|
    |:----------:|:-------:|:-------:|:-------:|:-------:|
    | negative     | 0.99    | 1.00    | 1.00    | 31638   |
    |  positive      |  0.60   |   0.26  |    0.36 | 277    |
    |**accuracy**|         |         | 0.99    | 31915   |
    |**macro avg**|    0.79 |   0.63  |    0.68 |   31915 |
    |**weighted avg**|    0.99 |   0.99  |    0.99 |   31915 |

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
    pred_probabilities = model.predict_proba(text).tolist()[0]
    
    # get prediction probabilities just for labels that apply
    probabilty = [pred_probability for i, pred_probability in enumerate(pred_probabilities) if prediction[i]]
   
    # for text that is not labeled provide a result to indicate no class identification
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