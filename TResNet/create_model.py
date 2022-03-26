import logging

from tresnet import TResnetM, TResnetL, TResnetXL

logger = logging.getLogger(__name__)

def create_model(args):
    model_params = {
        'args' : args, 'num_classes' : args.num_classes
    }
    args = model_params['args']
    args.model_name = args.model_name.lower()
    
    if args.model_name=='tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name=='tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name=='tresnet_l_v2':
        model = TResnetL_V2(model_params)
    elif args.model_name=='tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    return model