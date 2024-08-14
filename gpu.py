import tensorflow as tf

def print_gpu_memory_info():
    # Obtén una lista de GPUs físicas
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("No se detectaron GPUs.")
        return

    # Imprime información sobre la memoria total
    for gpu in gpus:
        # Aquí solo imprimimos información básica
        print(f"Nombre de la GPU: {gpu.name}")
        # TensorFlow 2.x no tiene un método directo para obtener la memoria libre y usada.
        # Solo podemos verificar la memoria total al crear el dispositivo.
        try:
            # Verificar si el dispositivo está siendo utilizado
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configuración de crecimiento de memoria habilitada para: {gpu.name}")
        except Exception as e:
            print(f"Error al configurar la GPU: {e}")

print_gpu_memory_info()