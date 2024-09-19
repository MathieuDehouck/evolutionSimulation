from unicodedata import category


chs = ['\u0325', '\u032c', '\u030a', '\u02b0', '\u0339', '\u031f', '\u0308', '\u0329', '\u02de', '\u0330', '\u02b7', '\u031c', '\u0320', '\u033d', '\u032f', '\u0324', '\u033c', '\u02b2', '\u02e0', '\u0334', '\u031e', '\u0319', '\u033a', '\u0303', '\u02e1', '\u02e4', '\u031d', '\u0318', '\u032a', '\u033b', '\u207f', '\u031a', '\u02b3', '\u02b5', '\u02c0', '\u02b1', '\u02b4', '\u02b6', '\u0322']



for x in chs:
    c = category(x)
    if c == 'Mn':
        print(' '+x)
    else:
        print(x)
