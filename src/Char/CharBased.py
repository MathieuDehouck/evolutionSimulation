"""
..
  A pure character based implementation of changes for fast iteration.
  Mathieu Dehouck
  06/2024
"""
from random import randint, choice, choices
from pickle import load
 
from Changes import Abstract_Change
from Language import Language

# load some probability derived from the index diachronica
#freq = load(open('tools/unconditioned_change_freq.pkl', 'rb'))
freq = load(open('tools/cond_change_freq_1.pkl', 'rb'))
weights = {}
contexts = {}
for src, cond_freqs in freq.items():
    weights[src] = [], []
    
    for tgt, freqs in cond_freqs.items():
        weights[src][0].append(tgt)
        weights[src][1].append(sum(freqs.values()))

        
        contexts[src, tgt] = tuple(zip(*freqs.items()))


class Change(Abstract_Change):
    """
    a basic character based change
    """

    def __init__(self, src, tgt, context):
        Abstract_Change.__init__(self)
        self.src = src
        self.tgt = tgt
        self.context = context


    @classmethod
    def generate(self, language, args):
        """
        a static method that generate a random change from the state of language
        """
        sources = set()

        while len(sources) == 0: # it's possible that a word be a phonetic dead end...  ɯ
            # pick a word at random
            w = choice(language.voc)
            w_ = ''.join([c for c in w if c not in language.syllable_sep])
            print(w, w_)

            # get all the possible sources for a change, pick one, then pick a target
            for src in weights:
                if src not in w_:
                    continue

                if 'ː' in src:
                    sources.add(src)
                    continue

                for i in range(len(w_)):
                    if w_[i:i+len(src)] == src:
                        #print(i, w, src)
                        if i+len(src) < len(w_) and w_[i+len(src)] == 'ː':
                            continue

                        sources.add(src)

        sources = sorted(sources)
        #for x in sources:
        #    print(x, weights[x])

        #ws = [sum([sum(weights[x][y][1]) for y in weights[x]]) for x in sources]
        #print([weights[x] for x in sources])
        ws = [sum(weights[x][1]) for x in sources]
        src = choices(sources, ws)[0]
        
        tgt = choices(weights[src][0], weights[src][1])[0]
        #print(src, tgt)
        
        return self(src, tgt, None)


    def __repr__(self):
        return self.src + '>' + self.tgt

    
    def affect(self, language):
        """
        apply a change to a language and return a new language
        """
        print(self.src, self.tgt)
        
        lang = Language(language.index + ' : ' + str(self), [])
        nvoc = set()
        atall = False
        for w in language.voc:
            w_ = w
            if self.context == None:
                w_ = w_.replace(self.src, self.tgt)
            else:
                print('Not Implemented Yet!')
                NotImplementedYet

            w_ = w_.simplify()
            if '∅' in w_.form:
                print('HERE')

            nvoc.add(w_)

            if w != w_:
                atall = True
                #print(w, w_, sep='\t->\t')

        lang.set_vocabulary(nvoc, language.syllable_sep)

        if atall == False:
            print('#######', self.src, self.tgt)
        print()
        
        return lang
    


change_map = {'phonet':Change}





def initialise(language, args):
    #print(args)
    voc = set()
    while len(voc) < args.getint('voc_size'):
        if args.getint('stress') == 0:
            w = "'"
        else:
            w = '⋅'
            print('Stress not implemented')

        for _ in range(randint(1, args.getint('max_length'))):
            w += choice(args.getlist('syllable')) + '⋅'
            

        if args.getboolean('regularize'):
            w = w.replace('c⋅v', '⋅cv')
            
        #print(args.getlist('c'), args.getlist('v'))
        w = ''.join([choice(args.getlist(c)) if c in args else c for c in w])

        if args.getboolean('geminate'):
            vowels = args.getlist('v')
            for v in vowels:
                w = w.replace(v+'⋅'+v, v+'ː')

        voc.add(w)
        #voc.add(Word(w, 0))

    voc = [Word(w, i) for i, w in enumerate(sorted(voc))]
    language.set_vocabulary(voc, ["'", '⋅'])
            



class Word():

    def __init__(self, form, index):
        self.form = form
        self.index = index


    def has(self, x):
        """
        check whether x is in self.form even concidering syllable boundary # add main and secondary stress
        """
        firsts = [i for i,c in enumerate(self.form) if c == x[0]]
        matched = []
        for i in firsts:
            k = 0
            j = 0
            while j < len(x) and i+k < len(self.form):
                if self.form[i+k] == '⋅':
                    k += 1

                elif self.form[i+k] == x[j]:
                    k += 1
                    j += 1

                else:
                    k = -1
                    break

            if i+k < len(self.form):
                if self.form[i+k] == 'ː':
                    continue
            
            if self.form[i:i+k].replace('⋅', '') == x:
                matched.append((i, i+k))

        if matched != []:
            return matched
        return None


    def replace(self, x, y):
        matched = self.has(x)
        if matched:
            f = ''
            p = 0
            for a, b in matched:
                if self.form[b] == y[-1] and self.form[b+1] == 'ː':
                    f += self.form[p:b]
                else:
                    f += self.form[p:a] + y
                p = b
            f += self.form[b:]

            print(self.form, matched, x, y, f, sep='\t')
            return Word(f, self.index)

        else:
            return self
        """
        if x[-1] == 'ː' or x+'ː' not in self.form:
            return Word(self.form.replace(x, y), self.index)
        else:
            f = self.form.replace(x+'ː', '༎').replace(x, y).replace('༎', x+'ː') # we protect geminated sounds
            return Word(self.form.replace(x, y), self.index)
        """
    
    def simplify(self):
        diff = False
        if '∅' in self.form:
            diff = True
            f = self.form.replace('∅', '')
        else:
            f = self.form
        w = f[0]
        for i in range(1, len(f)):
            if f[i] == w[-1]:
                w += 'ː'
                diff = True
            else:
                w += f[i]

        if diff:
            return Word(w, self.index)
        return self
                

    def __lt__(self, other):
        if self.form == other.form:
            return self.index < other.index
        return self.form < other.form

    def __getitem__(self, i):
        return self.form[i]


    def __repr__(self):
        return self.form


    def __hash__(self):
        return hash((self.form, self.index))

    
    def __len__(self):
        return len(self.form)
