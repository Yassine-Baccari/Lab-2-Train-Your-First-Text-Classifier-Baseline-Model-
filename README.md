# Lab 2: Train Your First Text Classifier (Baseline Model)

**Level:** Beginner | **Time:** about 40 minutes | **Language:** Python 3.8+

---

## What you will do

In Lab 1, you loaded the dataset, explored it, split it into train and validation sets, and cleaned the text.

In this lab, you reuse those steps and **train your first model**. You will learn:

1. What a **baseline** is and why we always start with a simple one.
2. How **TF-IDF** turns text into numbers.
3. How to train a model and measure it with **precision, recall and F1**.
4. How to **save** your model and use it on new text.

---

## Setup

Install the libraries (the same as Lab 1, but you only need these three):

```bash
pip install pandas scikit-learn joblib
```

If `pip` is not found, use `python -m pip install pandas scikit-learn joblib`.

**How to work:** use a Jupyter notebook (one code block = one cell), or put all the code blocks, in order, into one file called `lab2.py` and run it with `python lab2.py`. Run the steps **in order**: each step needs the previous one.

---

## Step 1: Load, clean and split the data (recap of Lab 1)

We use the same dataset and the same stratified split as in Lab 1.

```python
import re
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
df = pd.read_csv(DATASET_LOC)

def clean_text(text):
    """Simple cleaning: lowercase, remove links, keep letters and numbers."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()

# Combine title + description into one text column, then clean it
df["text"] = (df.title.fillna("") + " " + df.description.fillna("")).apply(clean_text)
df = df[["text", "tag"]].dropna(subset=["tag"])

# Stratified split (same settings as Lab 1)
train_df, val_df = train_test_split(
    df, stratify=df.tag, test_size=0.2, random_state=1234
)

print("Train:", train_df.shape, "Validation:", val_df.shape)
print(train_df.tag.value_counts())
```

**What this does:** it builds one `text` column, cleans it, and splits the data 80% / 20% while keeping the same tag proportions.

**Expected result:**

```text
Train: (611, 2) Validation: (153, 2)
natural-language-processing    248
computer-vision                228
other                           85
mlops                           50
Name: count, dtype: int64
```

These are the same train counts as in Lab 1. (In older pandas versions the last line may say `Name: tag` instead.)

> **Note:** In Lab 1, you removed stop words with NLTK. In this lab, TF-IDF removes them for us (Step 3), so you don't need NLTK. Also, scikit-learn accepts text labels directly, so we don't need label encoding here.

---

## Step 2: Build a baseline (the "chance" model)

A **baseline** is a very simple model. Your real model must beat it, otherwise it is not useful.

We define a function that computes the metrics, then test two simple baselines.

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def evaluate(y_true, y_pred):
    """Return accuracy, precision, recall and F1 (weighted over the classes)."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f1, 3),
    }

y_train = train_df.tag
y_val = val_df.tag

# Baseline 1: guess a random tag (4 classes, so about 25% correct)
random_model = DummyClassifier(strategy="uniform", random_state=1234)
random_model.fit(train_df.text, y_train)
print("Random       :", evaluate(y_val, random_model.predict(val_df.text)))

# Baseline 2: always predict the most common tag
majority_model = DummyClassifier(strategy="most_frequent")
majority_model.fit(train_df.text, y_train)
print("Most frequent:", evaluate(y_val, majority_model.predict(val_df.text)))
```

**What this does:** `DummyClassifier` makes predictions without learning anything from the text. Because the classes are not equal in size, "always predict the biggest class" already beats random guessing on accuracy.

**Expected result** (the Random numbers are close to 0.25 but can vary a little):

```text
Random       : {'accuracy': 0.2..., 'precision': ..., 'recall': ..., 'f1': ...}
Most frequent: {'accuracy': 0.405, 'precision': 0.164, 'recall': 0.405, 'f1': 0.234}
```

**Remember these numbers.** Your real model must do much better than 0.405 accuracy.

---

## Step 3: Turn text into numbers with TF-IDF

A model cannot read words, only numbers. **TF-IDF** gives each word a score in each text:

- the score is **higher** if the word appears often in this text;
- the score is **lower** if the word appears in almost every text (like "the" or "model").

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_train = vectorizer.fit_transform(train_df.text)  # learn the vocabulary on TRAIN only
X_val = vectorizer.transform(val_df.text)          # reuse it on validation

print("X_train shape:", X_train.shape)
print("X_val shape:  ", X_val.shape)
```

**What this does:** `fit_transform` learns the vocabulary from the training texts and converts them to numbers. For validation we only use `transform`. This follows the Lab 1 rule: **learn from the training split only** to avoid data leakage.

**Expected result:** 611 rows for train and 153 for validation, with the **same number of columns** in both (at most 5000, one per word):

```text
X_train shape: (611, 4xxx)
X_val shape:   (153, 4xxx)
```

---

## Step 4: Train a model and evaluate it

We use **logistic regression**, a simple and fast classifier.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=1234)
model.fit(X_train, y_train)          # training = learning from the examples

y_pred = model.predict(X_val)        # predict the tags of the validation set
print("TF-IDF + Logistic Regression:", evaluate(y_val, y_pred))
```

**Expected result:** all scores are much higher than the baselines, usually between 0.75 and 0.90 (your exact numbers may differ slightly):

```text
TF-IDF + Logistic Regression: {'accuracy': 0.8..., 'precision': 0.8..., 'recall': 0.8..., 'f1': 0.8...}
```

**How to read the metrics:**

- **Precision:** when the model says "computer-vision", how often is it right?
- **Recall:** of all the real "computer-vision" projects, how many did the model find?
- **F1:** one number that combines precision and recall.
- **Weighted:** the average over classes, where bigger classes count more.

---

## Step 5: See the results for each class

An overall score can hide problems. Let's look at each tag.

```python
print(classification_report(y_val, y_pred, zero_division=0))
```

**Expected result:** a table with one row per tag. The `support` column is the number of validation examples (about 57, 13, 62 and 21). The small classes (`mlops`, `other`) usually have **lower recall** than the big ones. This is the class imbalance you saw in Lab 1.

```text
                             precision    recall  f1-score   support

            computer-vision       0.9x      0.9x      0.9x        57
                      mlops       ...       ...       ...         13
natural-language-processing       0.8x      0.9x      0.9x        62
                      other       ...       ...       ...         21

                   accuracy                           0.8x       153
                  macro avg       ...       ...       ...        153
               weighted avg       0.8x      0.8x      0.8x       153
```

---

## Step 6: Predict the tag of a new project

To predict on new text, apply **the same steps as in training**: clean, vectorize, then predict.

```python
def predict_tag(title, description):
    text = clean_text(title + " " + description)
    features = vectorizer.transform([text])
    return model.predict(features)[0]

print(predict_tag(
    "Transfer learning with transformers",
    "Using transformers for transfer learning on text classification tasks."
))
```

**Expected result:**

```text
natural-language-processing
```

---

## Step 7: Save and reload your model

We save the vectorizer **and** the model together, so the preprocessing is never forgotten.

```python
import joblib

joblib.dump({"vectorizer": vectorizer, "model": model}, "baseline.joblib")

loaded = joblib.load("baseline.joblib")
text = clean_text("Image segmentation with U-Net. Segment medical images using a convolutional network.")
print(loaded["model"].predict(loaded["vectorizer"].transform([text]))[0])
```

**Expected result:** a file `baseline.joblib` appears in your folder, and the last line prints a tag, most likely:

```text
computer-vision
```

---

## Practical exercise

Imbalanced classes can hurt small classes. Try this:

1. Train a **second model** with the option `class_weight="balanced"`. This makes mistakes on small classes count more.
2. Print the metrics of both models and compare the **weighted F1**.
3. Use `predict_tag` to predict the tag of this project:
   - Title: `Object detection with YOLO`
   - Description: `Detect objects in images and videos.`

Try it yourself before reading the correction.

---

## Correction

```python
# 1. Second model with balanced class weights
model_bal = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=1234)
model_bal.fit(X_train, y_train)
y_pred_bal = model_bal.predict(X_val)

# 2. Compare both models
print("Normal  :", evaluate(y_val, y_pred))
print("Balanced:", evaluate(y_val, y_pred_bal))

# 3. Predict a new project
print(predict_tag("Object detection with YOLO", "Detect objects in images and videos."))
```

**Expected result:**

```text
Normal  : {'accuracy': 0.8..., 'precision': 0.8..., 'recall': 0.8..., 'f1': 0.8...}
Balanced: {'accuracy': 0.8..., 'precision': 0.8..., 'recall': 0.8..., 'f1': 0.8...}
computer-vision
```

**Explanation:**

- `class_weight="balanced"` gives more weight to rare classes (`mlops`, `other`). It often improves recall on those classes. The overall score may go up a little, stay the same, or go down slightly. Both outcomes are fine: the goal is to **compare and decide**, not to always get a higher number.
- The predicted tag for the YOLO project should be `computer-vision`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` (the package name is `scikit-learn`, not `sklearn`). Use the same Python as the one running your code. |
| `ModuleNotFoundError: No module named 'pandas'` or `joblib` | Run `pip install pandas joblib`. |
| `URLError` or `HTTP Error` when loading the CSV | Check your internet connection. Or download the CSV from the link in `DATASET_LOC`, save it next to your code, and use `DATASET_LOC = "dataset.csv"`. |
| `NameError: name 'train_df' is not defined` (or `X_train`, `model`, ...) | You skipped a step or restarted the notebook. Run the steps again **in order**, from Step 1. |
| `ConvergenceWarning: lbfgs failed to converge` | Increase `max_iter` (for example `max_iter=3000`). It is only a warning. |
| `UndefinedMetricWarning` | You forgot `zero_division=0` in the metrics. Use the `evaluate` function from Step 2. |
| `ValueError: ... inconsistent numbers of samples` | You mixed train and validation data. Use `X_train` with `y_train`, and `X_val` with `y_val`. |
| `ValueError: np.nan is an invalid document` | A title or description is empty. Keep the `.fillna("")` calls from Step 1. |
| Your numbers differ a little from the expected ones | This is normal. Library versions can change small details. Check that the shapes in Step 1 and the baseline in Step 2 match. |
| `FileNotFoundError: baseline.joblib` | Run Step 7 first, and run your code from the same folder where the file was saved. |

---

## Summary

- You always start with a **baseline** to know what "good" means.
- **TF-IDF** turns text into numbers, learned from the **training split only**.
- A simple model (**logistic regression**) already beats the baselines by a lot.
- **Precision, recall and F1** (overall and per class) tell you how good the model really is.
- You **save the vectorizer and the model together** so they can be reused on new data.

**Next lab idea:** fine-tune a pretrained language model (like SciBERT) and compare it to this baseline.
