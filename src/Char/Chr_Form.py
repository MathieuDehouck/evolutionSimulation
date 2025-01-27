"""
..
  A pure character based implementation of word.
  Mathieu Dehouck
  06/2024
""" 

from Abst_Tokens import Abstract_Form
from Char.Chr_chars import mods, vowels


class Form(Abstract_Form):
    
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
                if self.form[i+k] in mods:
                    continue
            
            if self.form[i:i+k].replace('⋅', '') == x:
                matched.append((i, i+k))

        if matched != []:
            return matched
        return None


    def replace(self, x, y):
        """
        replace x by y in a character based word meaningfull way
        """
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
            return Form(f, self.index)

        else:
            return self
        """
        if x[-1] == 'ː' or x+'ː' not in self.form:
            return Form(self.form.replace(x, y), self.index)
        else:
            f = self.form.replace(x+'ː', '༎').replace(x, y).replace('༎', x+'ː') # we protect geminated sounds
            return Form(self.form.replace(x, y), self.index)
        """

        
    def simplify(self):
        diff = False
        if '∅' in self.form:
            diff = True
            f = self.form.replace('∅', '')
        else:
            f = self.form
        skip = False
        w = f[0]
        for i in range(1, len(f)):
            if skip:
                skip = False
                continue
            if f[i] == w[-1]:
                w += 'ː'
                diff = True
            elif w[-1] in vowels and f[i] == '⋅' and i+1 != len(f) and f[i+1] == w[-1]:
                w += 'ː'
                diff = True
                skip = True
            else:
                w += f[i]

        if diff:
            return Form(w, self.index)
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
