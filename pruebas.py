number_layers = 6

for i in range(1, number_layers+1):
    if i == 1:
        print(f'capa inicial {i}')
    if i == number_layers:
        print(f'capa final {i}')
    elif not i in [1, number_layers]:
        print(f'capa intermedia {i}')