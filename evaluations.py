from rouge import Rouge
import numpy as np
import string


rouge = Rouge()


def rouge_score(hyp_ids, ref_ids, tokenizer):
    hyps = [tokenizer.decode(hyp_ids, skip_special_tokens=True)]
    if len(hyps[0]) == 0:
        return 0.0
    refs = [tokenizer.decode(ref_ids, skip_special_tokens=True)]
    try:
        rouge_score = rouge.get_scores(hyps, refs)[0]['rouge-l']['f']
    except ValueError:
        return 0.0
    return rouge_score



def acc_score(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return np.sum(preds == labels) / float(len(labels))



def bleu_score(hyp_ids, ref_ids, tokenizer):
    hyps = tokenizer.decode(hyp_ids, skip_special_tokens=True).lower()
    if len(hyps) == 0:
        return 0.0
    refs = tokenizer.decode(ref_ids, skip_special_tokens=True).lower()
    translator = str.maketrans('', '', string.punctuation)
    no_punctuation = hyps.translate(translator)
    hyps_words = no_punctuation.split()
    
    translator = str.maketrans('', '', string.punctuation)
    no_punctuation = refs.translate(translator)
    refs_words = no_punctuation.split()