"""
The necessary bits to parse the commandline arguments and configuration files.
..
  Simulating (language) evolution
  Mathieu Dehouck
  08/2024
"""

from argparse import ArgumentParser
from configparser import ConfigParser



def get_args():

    # setting arg parser and config file
    ap = ArgumentParser()
    ap.add_argument('config', help='A .ini file containing options.', default=None)

    ap.add_argument('-m', '--model', help='The type of model used to evolve forms.', default='char')
    ap.add_argument('-g', '--geo_model', help='The type of structure underlying the change history. Values: tree, (wave...)', default='tree')
    
    ap.add_argument('-d', '--depth', help='The maximum depth a branch.', type=int, default='20')
    ap.add_argument('-w', '--width', help='The eventual width of the tree.', type=int, default='10') # if we allow branches to die off, we could have a running width too

    ap.add_argument('--seed', help='The seed for the random number generator.', type=int)

    args = ap.parse_args()

    fargs = {k:v for k,v in vars(args).items() if k not in ['seed', 'config']}
    gargs = {k:v for k,v in vars(args).items() if k not in fargs}

    
    if args.config:
        config = ConfigParser(inline_comment_prefixes='#',
                              converters={'list':(lambda x: x.split()),
                                          'rule':(lambda x: [y.split() for y in x.split(' | ')])})
        config.optionxform = lambda opt: opt # we keep casing as it is
        config.read(args.config)

        mother = {k:config[k] for k in config.sections() if 'language' in k} # this is the initial language grammar


    if 'seed' not in gargs:
        if args.config and 'seed' in config['misc']:
            gargs['seed'] = config.getint('misc', 'seed')
    if not gargs['seed']:
        gargs['seed'] = 0


        
    return gargs, fargs, config, mother

