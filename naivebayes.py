import numpy as np
import sys
import string
import scipy.stats


def main():
    if len(sys.argv) != 5:
        print("Arg ERROR")
        exit(1)

    text_path = sys.argv[1]
    label_path = sys.argv[2]
    heldout_text_path = sys.argv[3]
    heldout_label_path = sys.argv[4]
    with open(text_path) as text_file:
        texts = text_file.read().split('\n')
    with open(label_path) as label_file:
        labels = label_file.read().split('\n')

    ratio = 1.0
    feat_dim = 500
    train_texts = texts[:int(ratio*len(texts))]
    test_texts = texts[int(ratio*len(texts)):]
    train_labels = labels[:int(ratio*len(labels))]
    test_labels = labels[int(ratio*len(labels)):]

    # ----------------- tokenization -----------------

    vocab = {}
    for line in train_texts:
        line = line.translate(str.maketrans('', '', string.punctuation))
        line = line.lower()
        tokens = line.split()
        for token in tokens:
            contain_digit = any(map(str.isdigit, token))
            if contain_digit:
                continue
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1
    with open('stopword.list') as stopword_file:
        stopword_list = stopword_file.read().split('\n')
    for stopword in stopword_list:
        vocab.pop(stopword, None)
    vocab.pop('br')
    vocab = {k: v for k, v in sorted(vocab.items(), key=lambda vocab: vocab[1], reverse=True)}

    # ----------------- feature extraction -----------------

    vocab_list = list(vocab)[:feat_dim]
    vocab_top500 = {vocab_list[i]: i for i in range(0, len(vocab_list))}

    CTF_features = []
    for line in train_texts:
        line = line.translate(str.maketrans('', '', string.punctuation))
        line = line.lower()
        tokens = line.split()
        CTF_feature = np.zeros(feat_dim)
        for token in tokens:
            contain_digit = any(map(str.isdigit, token))
            if contain_digit:
                continue
            if token in vocab_list:
                CTF_feature[vocab_top500[token]] += 1
        CTF_features.append(CTF_feature)
    CTF_features = np.asarray(CTF_features)
    CTF_pos_features = []
    CTF_neg_features = []
    for i in range(len(train_labels)):
        if train_labels[i] == 'pos':
            CTF_pos_features.append(CTF_features[i])
        elif train_labels[i] == 'neg':
            CTF_neg_features.append(CTF_features[i])

    CTF_pos_mean = np.mean(CTF_pos_features, axis=0)
    CTF_pos_stdev = np.std(CTF_pos_features, axis=0)
    CTF_neg_mean = np.mean(CTF_neg_features, axis=0)
    CTF_neg_stdev = np.std(CTF_neg_features, axis=0)

    with open(heldout_text_path) as heldout_text_file:
        heldout_texts = heldout_text_file.read().split('\n')

    CTF_features = []
    for line in heldout_texts:
        line = line.translate(str.maketrans('', '', string.punctuation))
        line = line.lower()
        tokens = line.split()
        CTF_feature = np.zeros(feat_dim)
        for token in tokens:
            contain_digit = any(map(str.isdigit, token))
            if contain_digit:
                continue
            if token in vocab_list:
                CTF_feature[vocab_top500[token]] += 1
        CTF_features.append(CTF_feature)
    CTF_features = np.asarray(CTF_features)
    prob_pos = []
    for i in range(CTF_features.shape[0]):
        prob_pos.append(scipy.stats.norm(CTF_pos_mean, CTF_pos_stdev).pdf(CTF_features[i]))
    prob_pos = np.asarray(prob_pos)
    prob_pos = np.nan_to_num(prob_pos, nan=1.0)
    prob_pos = np.prod(prob_pos, axis=1)

    prob_neg = []
    for i in range(CTF_features.shape[0]):
        prob_neg.append(scipy.stats.norm(CTF_neg_mean, CTF_neg_stdev).pdf(CTF_features[i]))
    prob_neg = np.asarray(prob_neg)
    prob_neg = np.nan_to_num(prob_neg, nan=1.0)
    prob_neg = np.prod(prob_neg, axis=1)

    result = ['neg' for i in range(CTF_features.shape[0])]
    postag = np.where(prob_pos > prob_neg)
    for index in postag[0]:
        result[index] = 'pos'
    result = np.asarray(result)
    np.savetxt(heldout_label_path, result, fmt='%s')

    print("stop")

    return


if __name__ == '__main__':
    main()
