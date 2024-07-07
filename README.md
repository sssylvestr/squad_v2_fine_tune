# Fine tuning Transformer models on SQuAD 2.0 dataset
The objective of this project is to develop a question-answering (QA) system using Transformers for the Stanford Question Answering Dataset (SQuAD) 2.0. The project involved researching existing papers on the subject, pre-processing the SQuAD 2.0 dataset, fine-tuning several candidate(BERT, ALBERT, ROBERTA base versions) models on reduced dataset, and evaluating its accuracy.
## Papers
* [Ensemble ALBERT on SQuAD 2.0](https://arxiv.org/abs/2110.09665) - comparative research on the performance of different ALBERT-based architectures on SQuAD 2.0 dataset from the creators of former 1 ranked model in the [leaderboard](https://paperswithcode.com/sota/question-answering-on-squad20);
* [What do Models Learn from Question Answering Datasets?](https://arxiv.org/abs/2004.03490) - evaluation of BERT-based models on several question answering datasets;
* [Exploring BERT Parameter Efficiency on the Stanford Question Answering Dataset v2.0](https://arxiv.org/pdf/2002.10670) - provides useful info about experiments on freezing certain portion of BERT layers and comparing their performances;
* [Comparative Analysis of State-of-the-Art Q\&AModels: BERT, RoBERTa, DistilBERT, and ALBERT onSQuAD v2 Dataset](https://www.researchsquare.com/article/rs-3956898/v1) - comparing the performance of different encoder-only architectures on SQuAD 2.0;
* [Evaluating QA: Metrics, Predictions, and the Null Response](https://chatgpt.com/c/1416cebb-3249-41a4-a625-ab6c78b7af3c) - nice overview of the metrics suitable for evaluation of extractive QA model performance;
* Some of the utility functions used in my work were taken from [Hugging Face Question Answering Tutorial](https://huggingface.co/learn/nlp-course/chapter7/7?fw=pt) and this [GitHub repo](https://github.com/e-tweedy/roberta-qa-squad2/blob/main/README.md).
## Dataset
[Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/) (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.
## How to use the model?
If you want to play around with the model you can access this [Streamlit App](https://sylvestr-squad.streamlit.app/).
![The Image](https://github.com/sssylvestr/squad_v2_fine_tune/blob/main/streamlit_app.png?raw=true)
