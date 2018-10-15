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
    def __init__(self, words=None, unknown=None):
        '''
        Parameters
        ----------
        words : List[str]
            If specified, Dictionary will be initialized with the words in the list.
        unknown : str
            If specidied, dictionary will be able to handle unknown words.
            The string will be used for unknown words. We recommend using '<UNK>'.
        '''
        if words is None: words = [] # To aboid bug. Note that default parameter is instatiated when class module is read.

        self.unk = unknown
        if unknown is not None:
            words.append(self.unk)
        self.itos = words
        self.stoi = dict([(word, i) for i, word in enumerate(words)])

    def add_word(self, word):
        if word not in self.itos:
            self.itos.append(word)
            self.stoi[word] = len(self.itos) - 1
        return self.stoi[word]

    def add_words(self, words):
        for word in words:
            self.add_word(word)

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
