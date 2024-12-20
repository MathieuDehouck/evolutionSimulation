"""
..
  Simulating (language) evolution
  Mathieu Dehouck
  10/2023
"""

from random import seed

from Command_Line import get_args

from Language import Language


# parse command line and config file
gargs, fargs, config, mother = get_args()

# set seed
seed(gargs['seed'])
print(gargs['seed'])

    
# get the changes implemented for the desired models
if fargs['model'] == 'char':
    from Char.CharBased_no_context import change_map
    from Char.Chr_Init import initialise
    #from CharBased import change_map, initialise


# assuming a tree model
if fargs['geo_model'] == 'tree':
    from TreeEvolution import evolve
else:
    print('Not implemented yet!')

    

# Run the simulation

# 1. initialise the ancestral language (note we could hav several)
lang = Language('origin', [])
lables = initialise(lang, mother)


voc = lang.voc
M = max(len(x) for x in voc)
for i in range(0, len(voc), 5):
    print('\t'.join([w.form.ljust(M+1) for w in voc[i:i+5]]))

print()
for syl in lables:
    print(sorted(syl.items(), key=lambda x: (x[1], x)))
    print()

    

# 2. now run the evolution
tree = evolve(lang, change_map, **fargs)

cogs = {w.index:[w.form] for w in lang.voc}

wids = [max(len(w.form) for w in lang.voc)+2]
for lf in tree.leaves():
    print(lf.lang.index)
    M = 0
    for w in lf.lang.voc:
        M = max(M, len(w))
        cogs[w.index].append(w.form)
    wids.append(M+2)
        
    #voc = lf.lang.voc
    #for i in range(0, len(voc), 5):
    #    print('\t'.join(voc[i:i+5]))




for k, vs in sorted(cogs.items()):
    print(k, ' \t'.join(v.ljust(l) for v, l in zip(vs, wids)).replace('⋅', '.').replace('ː', ':'))

