from collections import Counter
import numpy as np

def corpus_bleu(cand_list, refs_list):
    BP = _brevity_penalty(cand_list, refs_list)
    geo_mean = _geometric_mean_precision(cand_list, refs_list, N=4)
    return BP*geo_mean

def sentence_bleu(cand, ref):
    BP = _brevity_penalty([cand], [[ref]])
    geo_mean = _geometric_mean_precision([cand], [[ref]], N=4, smoothing=True)
    return BP*geo_mean

# BP
def _brevity_penalty(cand_list, refs_list):
    r, c = 0, 0
    for cand, refs in zip(cand_list, refs_list):
        best_len = _best_match_length(cand, refs)
        r += best_len
        c += len(cand)
    if c >= r:
        return 1
    else:
        return np.exp(1-r/c)

def _best_match_length(cand, refs):
    ref_lens = np.array([len(ref) for ref in refs])
    can_len = len(cand)
    distances = abs(can_len - ref_lens)
    idx = np.argmin(distances)
    return ref_lens[idx]

# precision
def _geometric_mean_precision(cand_list, refs_list, N, smoothing=False):
    precision = 0
    for n in range(1, N+1):
        p = _modified_ngram_precision(cand_list, refs_list, n, smoothing)
        if p == 0:
            continue
        precision += np.log(p)
    return np.exp(precision/N)

def _modified_ngram_precision(cand_list, refs_list, n, smoothing=False):
    clipped_freq, total_freq = 0, 0
    for cand, refs in zip(cand_list, refs_list):
        c, t = _clipped_ngram_count(cand, refs, n)
        clipped_freq += c
        total_freq += t
    if smoothing:
        clipped_freq += 1
        total_freq += 1
    return clipped_freq/total_freq

def _clipped_ngram_count(cand, refs, n):
    prd_counter = Counter(_get_n_gram(cand, n))
    ref_counter = _get_ref_counter(refs, n)
    clipped_freq = 0
    for n_gram, freq in prd_counter.items():
        tgt_freq = ref_counter[n_gram]
        clipped_freq += min(freq, tgt_freq)
    return clipped_freq, sum(prd_counter.values())

def _get_ref_counter(refs, n):
    counters = [Counter(_get_n_gram(ref, n)) for ref in refs]
    total_clipped_counter = Counter()
    # get all keys
    keys = set()
    for c in counters:
        keys = keys.union(c.keys())
    # combine all counters
    for key in keys:
        freqs = [c[key] for c in counters]
        total_clipped_counter[key] = max(freqs)
    return total_clipped_counter

def _get_n_gram(seq, n):
    n_gram = []
    for i in range(len(seq)-n+1):
        n_gram.append(tuple(seq[i:i+n]))
    return n_gram
