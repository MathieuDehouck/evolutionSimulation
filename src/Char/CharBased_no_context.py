"""
..
  A pure character based implementation of changes for fast iteration.
  Mathieu Dehouck
  06/2024
"""
from random import randint, choice, choices
from pickle import load

from Abst_Changes import Abstract_Change
from Language import Language
from Char.Chr_Form import Form

"""
vowels = ['a', 'a̯', 'ã', 'å', 'ɑ', 'ɑ́', '', 'ɑ̃', 'ɒ', 'æ', 'ʌ',
          'e', 'e̯', 'è', 'é', 'ê', 'ẽ', 'e̯', 'ə', 'ə́', 'ə̃', 'ɛ', 'ɛ̃', 'ɜ',
          'i', 'i̯', 'ı̃', 'î', 'ɨ', 'ɨ̯', 'ɨ̆', 'ɪ', 'ɪ̃',
          'ɔ', 'ɔ̃', 'o', 'o̰', 'ò', 'ó', 'ô', 'õ', 'o̯', 'ø', 'œ',
          'u', 'u̯', 'ũ', 'ʊ', 'ʊ́', 'y', 'ỹ', 'ʏ', 'ɤ', 'ъ', 'ь', 'ı', 'ṵ', 'ʉ', 'ɐ', 'ŭ']
"""
#mods = ['́', '̀', '̆', '̊', '̃', '̄', '̚', '̲', '̰', '̯', '̩', '̺', '̻', '̪', '̝', '̥', '͜', '͡', 'ː', 'ˑ', 'ʰ', 'ʱ', 'ʲ', 'ᵐ', 'ⁿ', 'ᵑ', 'ʷ', 'ˀ', 'ʼ', 'ˤ']


# load some probability derived from the index diachronica
freq = load(open('../tools/unconditioned_change_freq.pkl', 'rb'))
#freq = load(open('tools/cond_change_freq_1.pkl', 'rb'))
weights = {}
for src, freqs in freq.items():
    #weights[src] = {}
    #for tgt, freqs in cond_freqs.items():
    #    weights[src][tgt] = tuple(zip(*freqs))

    freqs = [f for f in freqs if f[0] != '∅']
    if freqs != []:
        weights[src] = tuple(zip(*freqs))

#cons = ['m', 'ɱ', 'n', 'ɳ', 'ɲ', 'ŋ', 'ɴ', 'p', 't', 'ʈ', 'c', 'k', 'q', 'ʔ', 'ʍ', 'w', 'b', 'd', 'ɖ', 'ɟ', 'ɡ', 'g', 'ɢ', 'r', 'ʀ', 'ɾ', 'ɽ', 'ɕ', 'ɸ', 'f', 'θ', 's', 'ʃ', 'ʂ', 'ç', 'x', 'χ', 'ħ', 'h', 'ʑ', 'β', 'v', 'ð', 'z', 'ʒ', 'ʐ', 'ʝ', 'ɣ', 'ʁ', 'ʕ', 'ɦ', 'ɬ', 'ɮ', 'ʋ', 'ɹ', 'j', 'ɰ', 'l', 'ɭ', 'ʎ', 'ʟ', 'ǁ', 'ǂ', 'ɓ', 'ɗ', 'ʄ', 'ɠ']

sounds = set()
for w, (ys, _) in sorted(weights.items()):
    sounds.update(w)
    for y in ys:
        sounds.update(y)

    #x = '   '.join([x for x in w if x not in 'bcdefghijklmnopqrstuvwxyz0123456789' and x not in vowels and x not in 'βθχʑʒŋʔɡʕʝʟɫɬɭɮɰɲʃɳɴɸɽɾʀʁʂʈʄʋʍʎʐɹɟɠɢɣɦɕçðħɓɕɖɗı'])

"""
for x in sorted(sounds):
    if x not in vowels and x not in mods:
        ()#print(x)
"""
#exit()


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

                ids = w.has(src)
                if ids:
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

