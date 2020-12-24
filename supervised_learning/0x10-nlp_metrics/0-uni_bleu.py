#!/usr/bin/env python3
"""
NLP traductor model
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence:

    References is a list of reference translations
        each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence

    Returns: the unigram BLEU score
    """
    uniq_lst = list(set(sentence))
    word_dct = {}

    for rfc in references:
        for word in rfc:
            if word in uniq_lst:
                if word not in word_dct.keys():
                    word_dct[word] = rfc.count(word)
                else:
                    crnt = rfc.count(word)
                    prev = word_dct[word]
                    word_dct[word] = max(crnt, prev)

    cnd_len = len(sentence)
    prob = sum(word_dct.values()) / cnd_len

    best_tpl = []
    for rfc in references:
        ref_len = len(rfc)
        diff = abs(ref_len - cnd_len)
        best_tpl.append((diff, ref_len))

    sort_tpl = sorted(best_tpl, key=lambda x: x[0])
    best_mch = sort_tpl[0][1]

    # Brevity penalty
    if cnd_len > best_mch:
        bp = 1
    else:
        bp = np.exp(1 - (best_mch / cnd_len))

    Bleu_score = bp * np.exp(np.log(prob))
    return Bleu_score
