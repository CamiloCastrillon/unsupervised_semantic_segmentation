import sys

def error_message(text):
    """
    Define un formato para los mensajes de error
    """
    print(f'\n********************************\nError:\n{text}\n********************************\n')
    sys.exit()

def warning_message(text):
    """
    Define un formato para los mensajes de advertencia
    """
    print(f'\n********************************\nWarning:\n{text}\n********************************\n')

def method_menssage(method):
    print(f'\nEjecutando {method}.')
