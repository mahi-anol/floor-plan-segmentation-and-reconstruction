import collections

### This config is mentioned in the paper and was achieved through NAS.

baseline_model_config={
    'stage1': {
        'r':(224,224), # Resolution (Hi,wi)
        'c':32, # output channel (ci)
        'l':1, #(li) number of repeat for a specific operator.
    },
    'stage2': {
        'r':(112,112), # Resolution (Hi,wi)
        'c':16, # output channel (ci)
        'l':1, #(li) number of repeat for a specific operator.
    },
    'stage3': {
        'r':(112,112), # Resolution (Hi,wi)
        'c':24, # output channel (ci)
        'l':2, #(li) number of repeat for a specific operator.
    },
    'stage4': {
        'r':(56,56), # Resolution (Hi,wi)
        'c':40, # output channel (ci)
        'l':2, #(li) number of repeat for a specific operator.
    },
    'stage5': {
        'r':(28,28), # Resolution (Hi,wi)
        'c':80, # output channel (ci)
        'l':3, #(li) number of repeat for a specific operator.
    },
    'stage6': {
        'r':(14,14), # Resolution (Hi,wi)
        'c':112, # output channel (ci)
        'l':3, #(li) number of repeat for a specific operator.
    },
    'stage7': {
        'r':(14,14), # Resolution (Hi,wi)
        'c':192, # output channel (ci)
        'l':4, #(li) number of repeat for a specific operator.
    },
    'stage8': {
        'r':(7,7), # Resolution (Hi,wi)
        'c':320, # output channel (ci)
        'l':1, #(li) number of repeat for a specific operator.
    },
    'stage9': {
        'r':(7,7), # Resolution (Hi,wi)
        'c':1280, # output channel (ci)
        'l':1, #(li) number of repeat for a specific operator.
    },
}


kernel_configs=[3,3,3,5,3,5,5,3,1]

tpu_friendly_efficient_resolutions=[224,240,260,300,380,456,528,600] # used for initial input.

best_grid_searched_coefficient=collections.namedtuple(typename="Best_Coefficients",
                                                      field_names=['alpha','beta','gemma'])(beta=[1.0,1.0,1.1,1.2,1.4,1.6,1.8,2.0,2.2,4.3]
                                                                                            ,alpha=[1.0,1.1,1.2,1.4,1.8,2.2,2.6,3.1,3.6,5.3]
                                                                                            ,gemma='Not needed'
                                                                                            ) # Directly Taken from the original repo, as the mentioned value from the paper doesnt match.

