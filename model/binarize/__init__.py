from option import args
if args.binarize == 'approxsign':
    from .approxsign import *
elif args.binarize == 'ste':
    from .WapproxAste import *