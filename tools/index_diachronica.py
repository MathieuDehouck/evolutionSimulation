from sys import argv
from tqdm import tqdm
from pickle import dump
from re import compile

f = open(argv[1])

context1 = compile('\w*_\w*')

"""
we need to out put the lines that could not be parsed properly
"""

changes = {}
for l in f:
    if l == '  <h2>46 Vowel Shifts</h2>\n':
        #print(l)
        break
    
    if 'class="schg"' in l:
        l = l.strip()
        l = l.replace('<sub>', '⋅').replace('</sub>', '').replace('<sup>', '⋅⋅').replace('</sup>', '').replace('<b>', '').replace('ɡ', 'g').replace('— ', '')
        l = l[l.find('>')+1:]

        #print(l)

        if '<b' in l:
            print(l)
        
        if '/' in l:
            try:
                chng, const = l.split(' / ')
            except:
                #print('######', l)
                continue

        else:
            chng = l
            const = ""

        chng = chng.split(' → ')
        #if len(chng) > 2:
        #print(chng)
        before = chng[0]
        after = chng[-1]
        #print(before, after, const, sep='\t')

        if '→' in before or '→' in after:
            #print('######', before, after, before==after, sep='\t')
            continue
        
        elif '#' in before or '#' in after:
            continue
        
        elif ',' in before or ',' in after:
            continue

        elif '{' in before or '}' in before or '{' in after:
            continue

        elif '[' in before or ']' in before or '[' in after:
            continue

        elif '(' in before or ')' in before or '(' in after:
            continue

        elif '~' in before or '~' in after:
            continue

        else:
            #print(before, after, const, sep='\t')
            befores = before.split(' ')
            afters = after.split(' ')

            #print(befores, afters, sep='\t')

        if const != "":
            mtch = context1.match(const)
            if mtch:
                if const != mtch[0]:
                    #print(mtch[0], const)
                    continue
            else:
                continue
            
        z = const

        for x, y in zip(befores, afters):
            if '?' in x or '?' in y:
                continue

            if '…' in x or '…' in y:
                continue

            if '∅' in x:
                if z[0] == '_':
                    lz = ''
                    rz = z[1:]
                elif z[-1] == '_':
                    lz = z[:-1]
                    rz = ''
                else:
                    lz, rz = z.split('_')

                if y.islower():
                    if lz.islower() and rz.islower():
                        z = ""
                        x = lz + rz
                        y = lz + y + rz
                    elif lz.islower():
                        z = "_" + rz
                        x = lz
                        y = lz + y
                    elif rz.islower():
                        z = lz + "_"
                        x = rz
                        y = y + rz
                    else:
                        #print(x, y, z)
                        # HERE
                        continue

                else:
                    continue
                    
                if z == '_':
                    z = ""
                    
                    #print(x, y, z)


            elif (x.islower() or x in ['ʔ']) and (y.islower() or y in ['∅', 'ʔ']):
                ()
                """
                if x not in changes:
                    changes[x] = {}

                if y not in changes[x]:
                    changes[x][y] = {}

                if z not in changes[x][y]:
                    changes[x][y][z] = 0
    
                changes[x][y][z] += 1
                """

            else:
                ()
                #print(x, y, l, before, befores, sep='\t')


            if x not in changes:
                changes[x] = {}

            if y not in changes[x]:
                changes[x][y] = {}

            if z not in changes[x][y]:
                changes[x][y][z] = 0
    
            changes[x][y][z] += 1
                


for x, ys in sorted(changes.items()):
    print(x)
    for y, z in sorted(ys.items()):
        print(x, '>', y, '/', z, sep='\t')
    '''
    for y, zs in sorted(ys.items()):
        changes[x][y] = sorted(zs.items(), key=lambda k: (k[1], k))
        for z, freq in changes[x][y]:
            print(x, y, z, freq, sep='\t')
    '''

exit()
    
dump(changes, open('cond_change_freq_1.pkl', 'wb'))
