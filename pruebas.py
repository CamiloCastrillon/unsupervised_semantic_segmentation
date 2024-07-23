from resources.create_architecture import CreateFullAuto as cfa

pth_model='C:/camilo/resources_uss/modelos/full_auto_encoder/full_auto_encoder_23-07-2024_02-40_PM.keras'
model = cfa().load_full_auto('y', pth_model)
print(model.summary())