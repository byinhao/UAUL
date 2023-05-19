import codecs as cs
datasets = ['Laptop', 'Restaurant']
dtypes = ['train', 'dev', 'test']
sp2sentiment = ["negative", "neutral", "positive"]
for dataset in datasets:
    for dtype in dtypes:
        f = cs.open('raw_data/' + dataset + '-ACOS/' + dataset.lower() + '_quad_' + dtype + '.tsv', 'r',
                    encoding='utf-8').readlines()
        all_sentences = []
        all_labels = []
        for line in f:
            line = line.strip().split('\t')
            sentence = line[0].strip()
            labels = line[1:]
            split_sentence = sentence.split()
            new_labels = []
            for label in labels:
                label = label.strip()
                at, ac, sp, ot = label.split()
                at_start, at_end = at.split(',')
                ot_start, ot_end = ot.split(',')
                if int(at_start) == -1:
                    at = 'NULL'
                else:
                    at = split_sentence[int(at_start): int(at_end)]
                    at = ' '.join(at)
                if int(ot_start) == -1:
                    ot = 'NULL'
                else:
                    ot = split_sentence[int(ot_start): int(ot_end)]
                    ot = ' '.join(ot)
                sp = sp2sentiment[int(sp)]
                if '#' in ac:
                    ac = ' '.join(ac.split('#')).lower()
                elif '_' in ac:
                    ac = ' '.join(ac.split('_')).lower()
                new_labels.append([at, ac, sp, ot])
            all_sentences.append(sentence)
            all_labels.append(new_labels)

        f_w = cs.open(dataset + '/' + dtype + '.txt', 'w', encoding='utf-8')

        for i in range(len(all_sentences)):
            f_w.write(all_sentences[i])
            f_w.write('####')
            f_w.write(str(all_labels[i]))
            f_w.write('\n')
