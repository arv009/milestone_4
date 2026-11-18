# Milestone 4: Sentiment Analysis from Classic NLP to LLMs

**Question:** When is a simple model good enough, and when does an LLM earn its cost?

This project classifies financial news sentences as **negative**, **neutral**, or **positive** with three tiers of
models, then compares them on accuracy, macro-F1, hard-case accuracy, speed, and cost.

| Tier | Approach | Models |
|---|---|---|
| 1. Rule-based | Word-score lexicon, no training | VADER, TextBlob |
| 2. Classic ML | Bag-of-words / TF-IDF features + Naive Bayes | NLTK + Naive Bayes, TF-IDF + Naive Bayes |
| 3. LLM | Prompted with instructions, no training (zero-shot) | Claude Haiku 5.5 (`claude-haiku-5-5`) |

## Dataset

**Financial PhraseBank** (`all-data.csv`): 4,846 sentences from financial news labeled by finance experts
(4,840 after removing duplicates). The file has no header row and is Latin-1 encoded.

| Class | Share |
|---|---|
| neutral | ~59% |
| positive | ~28% |
| negative | ~12% |

Because the classes are imbalanced, **macro-F1** is the main metric.

**Fixed splits** (seed `42`, stratified):
- 80% train / 20% test
- A **300-sentence test subset** taken from the test set. Every model is scored on these same 300 sentences.
- A hand-written set of **30 hard cases** (negation, mixed signals, comparisons, domain words like
  "loss narrowed", and neutral sentences with emotional words)

## How to run

### 1. Install the libraries
```bash
pip install -r requirements.txt
```

### 2. Add your Claude API key
Create a file named `.env` in this folder with one line:
```
ANTHROPIC_API_KEY=your-key-here
```
`.env` is listed in `.gitignore`, so Git will not commit it. **Never** put the key inside the notebook.
If you share the project as a zip or a drag-and-drop upload, remove `.env` first, because `.gitignore` only applies to Git.

### 3. Run the notebook
Open `sentiment_analysis.ipynb` in VS Code (or Jupyter), restart the kernel, and choose **Run All**.

- The Claude step makes about 330 API calls (300 test + 30 hard cases). With Haiku 5.5 this costs a few cents.
- Claude's answers are saved to `results/claude_predictions.csv`. On later runs the notebook loads this file
  instead of calling the API again. Delete it to get fresh predictions.
  

## Notebook steps

1. **Setup**: imports, load the API key with `load_dotenv()`
2. **Import data**: load `all-data.csv`, drop duplicates
3. **EDA**: class balance, average sentence length, 10 random examples per class, word cloud
4. **Train/test split**: fixed split, 300-sentence test subset, 30 hard cases
5. **Models**:
   - **VADER**: compound score with a ±0.2 threshold
   - **TextBlob**: polarity with a ±0.2 threshold
   - **NLTK + Naive Bayes**: `TweetTokenizer` + `mark_negation` + stopwords (negation words kept), top 3,000 words as bag-of-words features
   - **TF-IDF + Naive Bayes**: TF-IDF on unigrams and bigrams + `ComplementNB`
   - **Claude Haiku 5.5**: zero-shot system prompt, one-word answer, token counts logged for cost
6. **Comparison**: results table, macro-F1 chart, hard-cases chart, speed chart, confusion matrices
7. **Error analysis**: sentences every model got wrong, most common mistakes per model, hard cases side by side

## Metrics

| Measure | How |
|---|---|
| Accuracy | Share of correct labels |
| Macro-F1 | `f1_score(y, preds, average="macro")`, the main metric |
| Hard-cases acc. | Accuracy on the 30 hard cases |
| Sec / 1k preds | Prediction time on the 300 test sentences, scaled to 1,000 |
| Cost / 1k preds | Claude only: (input tokens × $0.10 + output tokens × $0.50) per 1M tokens, scaled to 1,000 predictions |

## Results

Scores on the 300-sentence test subset:

| Model | Accuracy | Macro-F1 | Hard-cases acc. | Sec / 1k preds | Cost / 1k preds |
|---|---|---|---|---|---|
| TF-IDF + Naive Bayes | 0.773 | 0.724 | 0.533 | ~0.1 | $0 |
| NLTK + Naive Bayes | 0.683 | 0.638 | 0.533 | ~0.1 | $0 |
| VADER | 0.610 | 0.571 | 0.333 | ~0.03 | $0 |
| TextBlob | 0.620 | 0.494 | 0.167 | ~0.08 | $0 |
| Claude Haiku 5.5 (zero-shot) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

> Fill in the Claude row from `results/metrics.csv` after running the notebook with your API key.
> Speed numbers vary slightly from machine to machine.

## Takeaways (so far)

- **VADER and TextBlob** need no training and are instant and free, but their dictionaries were built for social
  media and general text. They miss financial phrases like "profit rose" or "loss narrowed", and they label
  many sentences neutral.
- **Naive Bayes** learns financial vocabulary from the training data and is still almost free and very fast.
  TF-IDF with bigrams works best of the classic models. Negation and mixed-signal sentences are still hard for it.
- **Claude Haiku** needs no training data and should handle the hard cases best, but every prediction costs
  money and takes much longer. Compare its row to decide when that extra accuracy is worth it.
