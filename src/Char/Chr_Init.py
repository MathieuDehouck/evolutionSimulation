"""
Language initialiser for char based models
..
  Mathieu Dehouck
  08/2024
"""

from random import randint, choice, choices

from Language import Language
from Char.Chr_Form import Form


def initialise(language, args):
    #print(args)
    sanity = 0

    rules = {k:args['language.grammar.rules'].getrule(k) for k in args['language.grammar.rules']}
    syls = {k:args['language.grammar.syllables'].getlist(k) for k in args['language.grammar.syllables']}
    snds = {k:args['language.grammar.sounds'].getlist(k) for k in args['language.grammar.sounds']}
    #print(rules)

    syllables = {}, {}, {}, {}
    
    
    voc = set()
    raw = set()
    while len(voc) < args['language'].getint('voc_size'):
        # initialise the word
        w = choice(rules['word'])

        # unfold w into a sequence of syllables
        ln = randint(1, args['language'].getint('max_length'))
        while [x for x in w if x in rules] != []:
            i = 0
            while i < len(w) and w[i] not in rules:
                i += 1

            if len(w) == ln:
                pick_from = [x for x in rules[w[i]] if len(x) == 1]
            else:
                pick_from = [x for x in rules[w[i]] if len(x) == 2]

            w = w[:i] + choice(pick_from) + w[i+1:]

        # replace syllable types by archyphonemes sequences
        w = [choice(syls[x]) for x in w]
        #print(ln, w)
        
        if args['language'].getint('stress') != 0:
            print('Stress not implemented')
            
        w = "'" + '⋅'.join(w) + '⋅'
        
        #if args.getboolean('regularize'):
        #    w = w.replace('c⋅v', '⋅cv')
            
        # replace arhcyphonemes by actual sounds
        w = ''.join([choice(snds[c]) if c in snds else c for c in w])

        if w not in raw:
            raw.add(w)
        else:
            continue

        if args['language'].getboolean('long'):
            vowels = args['language.grammar.sounds'].getlist('V')
            for v in vowels:
                w = w.replace(v+'⋅'+v, v+'ː')


        # catch the syllables in each position before geminating consonants
        lables = w.replace("'", '⋅').split('⋅')
        lables = [s for s in lables if s != '']

        if lables[0] not in syllables[0]:
            syllables[0][lables[0]] = 1
        else:
            syllables[0][lables[0]] += 1

        if lables[-1] not in syllables[2]:
            syllables[2][lables[-1]] = 1
        else:
            syllables[2][lables[-1]] += 1

        for s in lables[1:-1]:
            if s not in syllables[1]:
                syllables[1][s] = 1
            else:
                syllables[1][s] += 1

        for s in lables:
            if s not in syllables[-1]:
                syllables[-1][s] = 1
            else:
                syllables[-1][s] += 1
            
                
        # takes care of gemination        
        if args['language'].getboolean('geminate'):
            cons = args['language.grammar.sounds'].getlist('C')
            for c in cons:
                w = w.replace(c+'⋅'+c, c+'ː').replace(c+c, c+'ː')

        voc.add(w)
        #voc.add(Word(w, 0))

    voc = [Form(w, i) for i, w in enumerate(sorted(voc, key=lambda x: x.replace('⋅','')))]
    language.set_vocabulary(voc, ["'", '⋅'])


    return syllables
