# Fine tuning Transformer models on SQuAD 2.0 dataset
The objective of this project is to develop an extarctive question-answering (QA) system using Transformers for the Stanford Question Answering Dataset (SQuAD) 2.0. The project involved researching existing papers on the topic, pre-processing the SQuAD 2.0 dataset, fine-tuning several candidate(BERT, ALBERT, ROBERTA base versions) models on reduced dataset, evaluating its accuracy, training the best model on full train set, creating a streamlit app for easy access to the model.
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
## Further Thoughts
If I was given more time for completing this task I would:
* invest more time to hyperparameter tuning(especially dropout rate and batch size);
* given having stronger compute I would try using larger versions of the models instead of base. As it is claimed in this [paper](https://arxiv.org/abs/2110.09665): using larger models with more parameters gives the most significant boost to ALBERT performance on SQuAD 2.0 dataset(they compared it to stacking more layers on top of (albert-base + extractive QA output layer) model). Presumably the same logic might apply to BERT and RoBERTa;
* do some error analysis to see where the model fails and where it is strong. We could potentially identify some mislabeled examples and augment dataset based on our findings;
* the top spots in [leaserboard](https://paperswithcode.com/sota/question-answering-on-squad20) belong to ensemble models, so this could also be a potential next step if we wanted to get a maximum score on this dataset;
* it is also common for extractive QA models to struggle with abbreviations(E.g. we have a context with a full title of 'National Basketball Association' and we ask a question 'What is NBA?'). So we might want to augment our data in a way to improve model's performance on this type of situations;
* among other potential pains of such models is its sensibilty to spelling errors. We could also augment the data by intruducing more examples with intentional misspelling: character swaps, character delition, phonetic replacements.
