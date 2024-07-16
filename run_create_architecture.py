from resources.create_architecture import CreateFullAuto as cfa

params = {
    'kernels'        : 8,
    'dim'            : 50,
    'number_layers'  : 4,
    'mode_l1'        : 'random',
    'mode_l2'        : 'random',
    'param_l1'       : 0.001,
    'param_l2'       : 0.001,
    'mode_do'        : 'random',
    'param_do'       : 0.2
}

cfa_instance = cfa(params['kernels'], params['dim'], params['number_layers'], params['mode_l1'], params['mode_l2'],
                    params['param_l1'], params['param_l2'], params['mode_do'], params['param_do'])

model = cfa_instance.architecture()
print(model.summary())

#dropout = cfa.choice_do()
#print(dropout.summary())

#reg = cfa.choice_reg()
#print(reg)

#_, _, _, do =cfa.addparam()
#print()
