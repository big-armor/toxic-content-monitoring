The API uses FastAPI for [documentation](https://fastapi.tiangolo.com/tutorial/first-steps/)
* ``` uvicorn src.main:app --host 0.0.0.0``` will run the API locally. ```uvicorn``` can be installed via `pip`.
* going to the url for the api with /docs will allow one to see the documentation and test requests.

It is currently deployed through Google's Cloud Services: Google Cloud Storage and App-Engine: [documentation](https://cloud.google.com/appengine/docs)

**Preprocessing**

The API accepts string based requests and preprocesses the text before running it through the predictive model. During preprocessing, the following are removed:
  - Newline characters
  - Return characters
  - Leading and trailing white spaces
  - Usernames if indicated with the term 'User'
  - IP addresses and user IDs
  - Non-printable characters e.g. unicode

Words that are disguised using characters such as * or @ are replaced with letters and common astericks words are replaced with the appropriate word. The API also separates toxic words from surounding text in case they are hidden within non-toxic content.

**Making Requests**

  Example Request Body:

  {"text": "I am an example string"}

**Model**

  The model is used to determine if the text contains toxic or offensive content.

  The possible labels are:
  - toxic
  - severe toxic
  - obscene
  - threat
  - insult
  - identity hate

  The API returns the cleaned text, all labels, True of False for each label, and the predicted probability of each.

  The current model is a Bi-directional LSTM + GRU neural network made with PyTorch, assuming FastText vectorization. Considerable preprocessing is performed on the text before vectorization. The metrics used in evaluating this model are F1 and ROC-AUC scores.

  F1 score is defined as the harmonic mean between precision and recall on a scale of 0 to 1. Recall demonstrates how effectively this model identifies all relevant instances of toxicity. Precision demonstrates how effectively the model returns only these relevant instances.

  The AUC score represents the measure of separability, in this case, distinguishing between toxic and non-toxic content. Also on a scale of 0 to 1, a high AUC score indicates the model successfully classifies toxic vs non-toxic. The ROC represents the probability curve.

  The F1 score for this model is 0.753 and the ROC-AUC score is 0.987

**Deployment**

The following deployment instructions are for Google Cloud:

1) Create a VM instance keeping in mind the amount of Memory your model will need(Make sure to allow http/https traffic)

2) Connect to the instance you created by clicking the ssh button

3) run command:gcloud init to specifiy username and project

4) Clone the repository in the cloud shell

5) Change directory to the repository

6) Create a virtual environment(venv)

7) Connect to the virtual environment

8) Install Dependencies

9) Deploy using code:uvicorn main:app --host 0.0.0.0

10) Visit the api: http://YOUR-EXTERNAL-IP-ADDRESS:8000 (Your external IP address can be found in the console when navigating to the VM instance tab)
