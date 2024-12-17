# Text similarity task
Program that retrieves n-most (by default 3) similar docs in French only.

All details regarding preprocessing, training and inference can be found in the paper.

## Training data
The data used for training was Belgian French news from the publicly available [RTBF corpus](https://dial.uclouvain.be/pr/boreal/object/boreal:276580)

## Model
The model used was Word2Vec.

## DEPENDENCIES
```bash
pip install -r requirements.txt
```

## LAUNCHING
(optional)
```bash
python -m spacy download fr_core_news_md
```

Main program

```bash
python launch_session.py
```
The program starts an interactive CLI session for model inference.
<img src="https://github.com/user-attachments/assets/27e8340d-215b-4d0c-9b1e-61b0e5d0a669" width="500">

After entering a sentence, it prints the n-most similar docs along with other details on the console.
<img src="https://github.com/user-attachments/assets/9401eee0-798d-40b2-b381-f21d84116d6e" width="500">

## AUTHOR
bianca.ciobanica@student.uclouvain.be

