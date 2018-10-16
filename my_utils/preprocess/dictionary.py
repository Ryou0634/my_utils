class Dictionary():
    '''
    Dictionary to convert a word string to a numerical index, and vice versa.

    Attribute
    ----------
    unk : str or None
        If specified, Dictionary will return an special index for unknown words.
    stoi : dict
        A dictionaty contains {str(word): int(index)}
    itos : List
        A list concains words. The order corresponds to stoi.
    '''
    def __init__(self, init_words=None, unknown=None, filepath=None):
        '''
        Parameters
        ----------
        init_words : List[str]
            If specified, Dictionary will be initialized with the words in the list.
        unknown : str
            If specified, dictionary will be able to handle unknown words.
            The string will be used for unknown words. We recommend using '<UNK>'.
        filepath : str
            If specified, dictionary will read this file.
            The file has "{word}" in each line.
        '''
        init_vocab = []

        if filepath:
            init_vocab += self._read_file(filepath)
        if init_words:
            init_vocab += init_words
        self.unk = unknown
        if unknown is not None:
            init_vocab += [self.unk]
        self.itos = init_vocab
        self.stoi = dict([(word, i) for i, word in enumerate(init_vocab)])

    def _read_file(self, path):
        with open(path, 'r') as f:
            init_vocab = f.read().split('\n')
        return init_vocab

    def add_word(self, word):
        if word not in self.itos:
            self.itos.append(word)
            self.stoi[word] = len(self.itos) - 1
        return self.stoi[word]

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def save(self, path):
        with open(path, 'w') as f:
            f.write('\n'.join(self.itos))

    def __len__(self):
        return len(self.itos)

    def __call__(self, word):
        if self.unk:
            try:
                return self.stoi[word]
            except KeyError:
                return self.stoi[self.unk]
        else:
            return self.stoi[word]

    def __getitem__(self, idx):
        return self.itos[idx]

    def __repr__(self):
        repr = 'Dictionary(\n' + \
                    '\tsize: {}\n'.format(len(self))
        n = 5
        head = '\t' + str(dict(list(self.stoi.items())[:n]))
        if len(self) > n:
            head = head[:-1] + '...'
        repr += head + '\n)'
        return repr
