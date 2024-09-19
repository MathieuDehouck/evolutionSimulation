"""
..
  What the represent language states.
  Mathieu Dehouck
  10/2023
"""





class Language():

    def __init__(self, index, voc):
        self.index = index
        self.voc = voc
        self.sounds = []


    def set_vocabulary(self, voc, syllable_sep, index=False):
        if index:
            self.voc = sorted(voc)
            for i, w in enumerate(self.voc):
                w.index = i

        else:
            self.voc = sorted(voc, key=lambda w:w.index)
            
        self.sounds = set()

        for w in voc:
            for x in w:
                if x not in syllable_sep:
                    self.sounds.add(x)
        
        self.sounds = sorted(self.sounds)
        self.syllable_sep = syllable_sep
