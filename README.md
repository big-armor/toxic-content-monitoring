"# toxic-content-monitoring"

API uses FastText: [documentation](https://fastapi.tiangolo.com/tutorial/first-steps/)
Deployed through Google's Cloud Services: Google Cloud Storage and App-Engine: [documentation](https://cloud.google.com/appengine/docs)

This API accepts string based requests and preprocesses the text before
  running it through the predictive model.

  **Preprocessing**
  The following are removed:
  - Newline characters
  - Return characters
  - Leading and trailing white spaces
  - Usernames if indicated with the term 'User'
  - IP addresses and user IDs
  - Non-printable characters e.g. unicode

  Words that are disguised using characters such as * or @ are replaced with
  letters and common astericks words are replaced with coordinating word.

  The API also preprocesses text so that if toxic words are mixed in with
  non-toxic content, it will split the text before and after so the toxic words
  are identified.

  **Making Requests**
  Example Request Body:

  {"text": "string"}

  **Model**
  The model is used to determine if the text contains toxic or offensive content.

  The possible labels are:
  - toxic
  - severe toxic
  - obscene
  - threat
  - insult
  - identity hate

  The API returns the cleaned text, all labels, True of False for each label,
  and the predicted probability of each.

  The current model is a Bi-directional LSTM + GRU neural network made with
  PyTorch, assuming FastText vectorization. Considerable preprocessing is
  performed on the text before vectorization. The metrics used in evaluating
  this model are F1 and ROC-AUC scores.

  F1 score is defined as the harmonic mean between precision and recall on a scale of
  0 to 1. Recall demonstrates how effectively this model identifies all relevant
  instances of toxicity. Precision demonstrates how effectively the model returns
  only these relevant instances.

  The AUC score represents the measure of separability, in this case, distinguishing
  between toxic and non-toxic content. Also on a scale of 0 to 1, a high AUC score
  indicates the model successfully classifies toxic vs non-toxic. The ROC represents
  the probability curve.

  The F1 score for this model is 0.753 and the ROC-AUC score is 0.987
