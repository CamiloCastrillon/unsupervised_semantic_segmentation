from resources.create_architecture import CreateFullAuto as cfa

verify_errors='y'
verify_warnings='y'
kernels=8
dim=50
number_layers=4
mode_l1=None
mode_l2=None
param_l1=None
param_l2=None
mode_do=None
param_do=None
lr=0.001
pth_save_model = 'C:/camilo/resources_uss/models/full_auto_encoder/'

model_fa = cfa().create_model(verify_errors, verify_warnings, kernels, dim, number_layers, mode_l1, mode_l2, param_l1, param_l2, mode_do, param_do, lr)
print(f'\n{model_fa.summary()}\n')
cfa().save_model('y', model_fa, pth_save_model)
