from my_utils.misc.bleu import _modified_ngram_precision, corpus_bleu

from my_utils import Dictionary
def test_modified_ngram_precision():

    candidate1 = 'it is a guide to action which ensures that the military always obeys the commands of the party'
    candidate2 = 'it is to insure the troops forever hearing the activity guidebook that party direct'

    reference1 = 'it is a guide to action that ensures that the military will forever heed party commands'
    reference2 = 'it is the guiding principle which guarantees the military forces always being under the command of the party'
    reference3 = 'it is the practical guide for the army always to heed the directions of the party'
    references = [reference1, reference2, reference3]

    dictionary = Dictionary()
    for candidate in [candidate1, candidate2]:
        for w in candidate.split():
            dictionary.add_word(w)
    for reference in references:
        for w in reference.split():
            dictionary.add_word(w)

    candidate1 = [dictionary(w) for w in candidate1.split()]
    candidate2 = [dictionary(w) for w in candidate2.split()]
    references = [[dictionary(w) for w in reference.split()] for reference in references]
    assert 17/18 == _modified_ngram_precision([candidate1], [references], n=1)
    assert 8/14 == _modified_ngram_precision([candidate2], [references], n=1)
    assert 10/17 == _modified_ngram_precision([candidate1], [references], n=2)
    assert 1/13 == _modified_ngram_precision([candidate2], [references], n=2)
    assert 1 == _modified_ngram_precision([[dictionary('of'), dictionary('the')]], [references], n=2)


import math
from nltk import word_tokenize, bleu_score
def test_corpus_bleu():
    hy = "I have a pen"
    hy = word_tokenize(hy)
    re = "I have a apple"
    re = word_tokenize(re)
    res = [ re ]
    assert math.isclose(bleu_score.sentence_bleu(res, hy), corpus_bleu([hy], [res]))
