import pprint

class Example():
    '''
    An instance of dataset.
    Typically this consists of features and target.
    '''
    def __init__(self):
        pass

    def set_data(self, fields, data):
        '''
        Parameters
        ----------
        fields : str or List[str]
            Specify the name of attribute
        data : (List of) data
            Data to be set
        '''
        if isinstance(fields, str):
            setattr(self, fields, data)
        elif isinstance(fields, list):
            for f, d in zip(fields, data):
                setattr(self, f, d)
        return

    def get_sentences(self):
        return [data for field, data in vars(self).items() if field != 'label']

    def __repr__(self):
        return pprint.pformat(vars(self))

class Dataset():
    '''
    Dataset which contains Examples.
    '''
    def __init__(self, fields=None, filepath=None, tokenize=None, header=False):
        '''
        fields : str of List[str]
            This must be (sentence, or label) to work with Corpus class.
        '''
        self.data = []
        if (fields or filepath or tokenize) is not None:
            self._init_read_file(fields, filepath, tokenize, header)


    def _init_read_file(self, fields, filepath, tokenize, header):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if header: lines = lines[1:]
            for line in lines:
                ex = Example()
                if tokenize:
                    line = tokenize(line)
                ex.set_data(fields, line)
                self.data.append(ex)

    def read_file(self, fields, filepath, tokenize, header=False):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if header: lines = lines[1:]
            assert len(self.data) == len(lines)
            for ex, line in zip(self.data, lines):
                if tokenize:
                    line = tokenize(line)
                ex.set_data(fields, line)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        string = 'Dataset\n'
        string += 'Size: {}'.format(len(self))
        return string

class Dictionary():
    def __init__(self, words=None):
        if words is None: words = [] # To aboid bug. Note that default parameter is instatiated when class module is read.
        self.itos = words
        self.stoi = dict([(word, i) for i, word in enumerate(words)])

    def add_word(self, word):
        if word not in self.itos:
            self.itos.append(word)
            self.stoi[word] = len(self.itos) - 1
        return self.stoi[word]

    def __len__(self):
        return len(self.itos)

    def __call__(self, word_or_idx):
        if type(word_or_idx) is int:
            return self.itos[word_or_idx]
        elif type(word_or_idx) is str:
            return self.stoi[word_or_idx]
        else:
            raise Exception('Invalid Input. Input must be int or str')

    def __repr__(self):
        string = 'Dictionary\n'
        string += 'Size: {}'.format(len(self))
        return string

class Corpus():
    '''
    A set of train, test, dev dataset.

    Attributes
    ----------
    train, dev, test : Dataset

    +++ Available after running build_vocab() +++
    {field}_freq : dict
        The frequency list of {field}.
    {field} : Dictionary
        Dictionary of {field}.
    '''

    def __init__(self, train, dev=None, test=None):
        self.train = train
        self.dev = dev
        self.test = test

    def build_vocab(self, field, attrs, typ, min_freq=0):
        setattr(self, field+'_freq', {})
        freq = getattr(self, field+'_freq')
        if self.train:
            self._count_freq('train', freq, attrs, typ)
        if self.dev:
            self._count_freq('dev', freq, attrs, typ)
        if self.test:
            self._count_freq('test', freq, attrs, typ)
        words = [word for word, f in freq.items() if f > min_freq]
        setattr(self, field, Dictionary(words))
        if min_freq > 0:
            getattr(self, field).add_word('<OOV>')

    def get_numericalized(self, dataset, field, attr, typ):
        dictionary = getattr(self, field)
        dataset = getattr(self, dataset)
        numericalized = []
        for ex in dataset.data:
            data = getattr(ex, attr)
            if typ == 'seq':
                num = [dictionary(w) for w in data]
            elif typ == 'label':
                num = dictionary(data)
            numericalized.append(num)
        return numericalized

    def _count_freq(self, dataset, freq, attrs, typ):
        dataset = getattr(self, dataset)
        for ex in dataset.data:
            for attr in attrs:
                data = getattr(ex, attr)
                if typ == 'seq':
                    for w in data:
                        freq[w] = freq.get(w, 0) + 1
                elif typ == 'label':
                    freq[data] = freq.get(data, 0) + 1
        return

    def __repr__(self):
        string = 'Corpus\n'
        if self.train:
            string += 'Train: {}\n'.format(len(self.train))
        if self.dev:
            string += 'Dev: {}\n'.format(len(self.dev))
        if self.test:
            string += 'Test: {}\n'.format(len(self.test))
        for var in vars(self):
            dictionary = getattr(self, var)
            if isinstance(dictionary, Dictionary):
                string += '======' + var + '======\n'
                string += 'Vocabulary: {}\n'.format(len(dictionary))
                freq = getattr(self, var+'_freq')
                string += 'The number of tokens: {}\n'.format(sum(freq.values()))
        return string
