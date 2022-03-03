from option import args
if args.binarize == 'approxsign':
    from .approxsign import *
elif args.binarize == 'WapproxAste':
    from .WapproxAste import *