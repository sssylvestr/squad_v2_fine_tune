import collections
import numpy as np
from tqdm import tqdm
import evaluate

def format_predictions(start_logits, end_logits, inputs, examples,
                      n_best=20, max_answer_length=30, convert_empty=False):
    """
    Postprocessing of logits into prediction data
    Parameters:
    -----------
    start_logits, end_logits : list, list
        sequences of logits corresponding to possible start
        and end token indices of the answer
    inputs : dataset
        The tokenized and and preprocessed dataset containing columns
        'example_id', 'offset_mapping' (other columns are ignored)
    examples : datasets.Dataset
        The dataset of examples.  Must have columns:
        'id', 'question', 'context'
    n_best : int
        The number of top start/end (by logit) indices to consider
    max_answer_length : int
        The maximum length (in characters) allowed for a candidate answer
    convert_empty : bool
        Whether to transform prediction text of "" to
        "I don't have an answer based on the context provided."

    Returns:
    --------
    predicted_answers : list(dict)
        for each entry, keys are 'id', 'prediction_text'
    """
    assert n_best <= len(inputs['offset_mapping'][0]), 'n_best cannot be larger than max_length'

    # Dictionary whose keys are example ids and values are corresponding indices of tokenized feature sequences
    example_to_inputs = collections.defaultdict(list)
    for idx, feature in enumerate(inputs):
        example_to_inputs[feature["example_id"]].append(idx)

    # Loop through examples
    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # For each example, loop through corresponding features
        for feature_index in example_to_inputs[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]

            # Retrieve modified offset_mapping;
            # Context tokens indices have actual offset_mapping pair,
            # all other indices have None
            offsets = inputs[feature_index]['offset_mapping']

            # Get indices of n_best most likely start, end token index values for the answer
            start_indices = np.argsort(start_logit)[-1:-n_best-1:-1].tolist()
            end_indices = np.argsort(end_logit)[-1:-n_best-1:-1].tolist()

            # Loop over all possible start,end pairs
            for start_index in start_indices:
                for end_index in end_indices:
                    # Skip pair which would require an answer to have negative length or length greater than max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    # Skip pairs having None for exactly one of offsets[start_index],
                    # offsets[end_index] - which would require the answer to only
                    # partially lie in this context sequence
                    if (offsets[start_index] is None) ^ (offsets[end_index] is None):
                        continue

                    # Pairs which have None for both correspond to an
                    # empty string as the answer prediction
                    # Adding logits is equivalent to multiplying probabilities
                    if (offsets[start_index] is None) & (offsets[end_index] is None):
                        answers.append(
                            {
                                "text": '',
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                        )
                    # If neither are None and the answer has positive length less than
                    # max_answer_length, then this corresponds to a non-empty answer candidate
                    # in the context and we want to include it in our list
                    else:
                        answers.append(
                            {
                                "text": context[offsets[start_index][0]: offsets[end_index][1]],
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                        )
        # Retrieve logits and probability scores for all viable start,end combinations

        # If there are candidate answers, choose the candidate with largest logit score
        # this might be ''
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x['logit_score'])
            predicted_answers.append({'id': example_id, 'prediction_text': best_answer['text']})
        else:
            predicted_answers.append({'id': example_id, 'prediction_text': ''})
        if convert_empty:
            for pred in predicted_answers:
                if pred['prediction_text'] == '':
                    pred['prediction_text'] = "I don't have an answer based on the context provided."
    return predicted_answers

def compute_metrics(start_logits, end_logits, inputs, examples,
                    n_best=20, max_answer_length=30):
    """
    Compute the results of the SQuAD v2 metric on predictions
    Parameters:
    -----------
    start_logits, end_logits : list, list
        sequences of logits corresponding to possible start
        and end token indices of the answer
    inputs : dataset
        The tokenized and and preprocessed dataset containing columns
        'example_id', 'offset_mapping' (other columns are ignored)
    examples : datasets.Dataset
        The dataset of examples.  Must have columns:
        'id', 'question', 'context'
    n_best : int
        The number of top start/end (by logit) indices to consider
    max_answer_length : int
        The maximum length (in characters) allowed for a candidate answer
    Returns:
    --------
    metrics : dict
        dictionary of metric values
    """
    metric = evaluate.load('squad_v2')
    predicted_answers = format_predictions(start_logits, end_logits, inputs, examples,
                                           n_best=n_best, max_answer_length=max_answer_length)
    for pred in predicted_answers:
        pred['no_answer_probability'] = 1.0 if pred['prediction_text'] == '' else 0.0
    theoretical_answers = [{'id': ex['id'], 'answers': ex['answers']} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def format_metrics(metrics):
    formatted_metrics = f"""
    Overall Metrics:
        - Exact Match: {metrics['exact']}
        - F1 Score: {metrics['f1']}
        - Total Questions: {metrics['total']}

    Has Answer Metrics:
        - Exact Match: {metrics['HasAns_exact']}
        - F1 Score: {metrics['HasAns_f1']}
        - Total Questions: {metrics['HasAns_total']}

    No Answer Metrics:
        - Exact Match: {metrics['NoAns_exact']}
        - F1 Score: {metrics['NoAns_f1']}
        - Total Questions: {metrics['NoAns_total']}

    Best Threshold Metrics:
        - Best Exact Match: {metrics['best_exact']} (Threshold: {metrics['best_exact_thresh']})
        - Best F1 Score: {metrics['best_f1']} (Threshold: {metrics['best_f1_thresh']})
    """
    print(formatted_metrics)
