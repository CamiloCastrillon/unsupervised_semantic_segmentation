import sys
import os
import time

def error_message(text):
    """
    Define un formato para los mensajes de error
    """
    print(f'\n********************************\nError:\n{text}\n********************************\n')
    sys.exit()

def warning_message(text):
    """
    Define un formato para los mensajes de advertencia con un tiempo de espera de 5 segundos
    """
    print(f'\n********************************\nWarning:\n{text}\n********************************\n')
    time.sleep(5)
    os.system('cls')

def method_menssage(method):
    print(f'\nEjecutando {method}.')
