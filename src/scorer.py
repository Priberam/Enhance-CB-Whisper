from typing import Union, List
from string2string.alignment import NeedlemanWunsch
from priberam_tokenizer import PriberamTokenizer, Token


def entity_recall(
    preds: List[str],
    refs: List[str],
    mentions: List[List[dict]],
    ner_tags: Union[str, List[str]],
    char_split: bool = False
):

    assert not isinstance(ner_tags, str) or ner_tags == 'ALL', f'invalid NER tags'
    if ner_tags == 'ALL':
        ner_tags = ['ALL']

    # Create an instance of the PriberamTokenizer class
    tokenizer = PriberamTokenizer()
    
    # Create an instance of the NeedlemanWunsch class
    nw = NeedlemanWunsch(gap_char='[SKIP]')    

    # Counter
    counts = {ner_tag: {
        'TP': 0,
        'FN': 0,
        'N': 0
    } for ner_tag in set(ner_tags + ['ALL'])} 

    for pred, ref, mentions_ in zip(preds, refs, mentions):

        # case the prediction is empty
        if pred.strip() == '':
            # Add to FN and total N
            for m_ in mentions_:
                if ner_tags == ['ALL'] and not m_['ner_tag'] in counts.keys():
                    counts[m_['ner_tag']] = {'TP': 0, 'FN': 0, 'N': 0}
                if m_['ner_tag'] in counts.keys():
                    counts[m_['ner_tag']]['N'] += 1
                    counts['ALL']['N'] += 1
                    counts[m_['ner_tag']]['FN'] += 1
                    counts['ALL']['FN'] += 1
            continue

        # obtain tokens using the Priberam tokenizer
        # and filter the ones corresponding to spaces or newlines
        pred_tokens = [token for token in tokenizer.tokenize(pred)[0] if token.type != 'newline']
        ref_tokens = [token for token in tokenizer.tokenize(ref)[0] if token.type != 'newline']
        if char_split:
            pred_tokens = [Token(
                index = -1, 
                start = token.start + c_idx, 
                end = token.start + c_idx + 1, 
                text = c_, 
                type = 'text'
            ) for token in pred_tokens for c_idx, c_ in enumerate(token.text)]
            ref_tokens = [Token(
                index = -1, 
                start = token.start + c_idx, 
                end = token.start + c_idx + 1, 
                text = c_, 
                type = 'text'
            ) for token in ref_tokens for c_idx, c_ in enumerate(token.text)]

        # global align the predicted and reference sequences of tokens
        seq1, seq2 = nw.get_alignment([tk.text for tk in pred_tokens], [tk.text for tk in ref_tokens], return_score_matrix=False)

        # process the segmented prediction
        # the segmented output is delimeted by '|'
        # so we have to deal with possible occurences of '|' in the source when spliting
        offset = 0
        t_seq1 = seq1.split('|')
        t_seq1_ = []
        idx = 0
        while idx < len(t_seq1):
            if t_seq1[idx] != ' ':
                t_seq1_.append(t_seq1[idx].strip())
                idx += 1
            else:
                idx += 2
                t_seq1_.append('|')
        t_seq1 = t_seq1_

        # process the segmented reference
        # the segmented output is delimeted by '|'
        # so we have to deal with possible occurences of '|' in the source when spliting
        offset = 0
        t_seq2 = seq2.split('|')
        t_seq2_ = []
        idx = 0
        while idx < len(t_seq2):
            if t_seq2[idx] != ' ':
                t_seq2_.append(t_seq2[idx].strip())
                idx += 1
            else:
                idx += 2
                t_seq2_.append('|')
        t_seq2 = t_seq2_

        indices = [-1] * len(t_seq2)
        for idx, t_ in enumerate(indices):
            if t_seq2[idx].strip() == '[SKIP]':
                offset -= 1
            else:
                indices[idx] = idx + offset
        # align the mentions with the tokens
        m_indices = [-1] * len(ref_tokens)
        for idx in range(len(ref_tokens)):
            for m_idx, m_ in enumerate(mentions_):
                if (m_['end_offset'] - ref_tokens[idx].start) * (m_['total_offset'] - ref_tokens[idx].end) < 0:
                    m_indices[idx] = m_idx
        for idx in [idx for idx, i_ in enumerate(indices) if i_ == -1]:
            if idx > 0 and idx < len(m_indices) and m_indices[idx-1] == m_indices[idx]:
                m_indices.insert(idx, m_indices[idx-1])
            else:
                m_indices.insert(idx, -1)
        idx = 0
        m_indices_ = []
        while idx < len(m_indices):
            if m_indices[idx] != -1:
                m_index = m_indices[idx]
                m_indices_.append([m_index, []])
                while idx < len(m_indices) and m_indices[idx] != -1 and m_index == m_indices[idx]:
                    m_indices_[-1][1].append(idx)
                    idx += 1
            else:
                idx += 1
        m_indices = m_indices_
        
        # Count TP, FN and total N
        for m_index, i_seq in m_indices:
            m_ = mentions_[m_index]
            if ner_tags == ['ALL'] and not m_['ner_tag'] in counts.keys():
                counts[m_['ner_tag']] = {'TP': 0, 'FN': 0, 'N': 0}
            if m_['ner_tag'] in counts.keys():
                counts[m_['ner_tag']]['N'] += 1
                counts['ALL']['N'] += 1
                if all(t_seq1[i_] == t_seq2[i_] for i_ in i_seq):
                    counts[m_['ner_tag']]['TP'] += 1
                    counts['ALL']['TP'] += 1
                else:
                    counts[m_['ner_tag']]['FN'] += 1
                    counts['ALL']['FN'] += 1

    # compute recall
    recall = {key: float(counts[key]['TP']) / float(counts[key]['N']) if counts[key]['N'] != 0 else 0 for key in counts.keys()}

    return recall