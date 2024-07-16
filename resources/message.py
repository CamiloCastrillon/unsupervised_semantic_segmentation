import sys
import os
import time

def error_message(err_des:str) -> None:
    """
    Define un formato para los mensajes de error.

    Args:
        err_des (str): Texto que describe el error.
    
    Returns:
        None: No se espera argumento de salida.
    """
    print(f'\n********************************\nError:\n{err_des}\n********************************\n')
    sys.exit()

def warning_message(war_des:str) -> None:
    """
    Define un formato para los mensajes de advertencia con un tiempo de espera de 10 segundos.

    Args:
        war_des (str): Texto que describe la advertencia.
    
    Returns:
        None: No se espera argumento de salida.
    """
    for remaining in range (10, 0, -1):
        print(f'\n********************************\nWarning:\n{war_des}\nTiempo de espera: {remaining}.\n********************************\n')
        time.sleep(1)
        os.system('cls' if os.name == 'nt' else 'clear')

def method_menssage(met_name:str, met_des:str) -> None:
    """
    Crea un formato para los mensajes que avisan cuando se está ejecutando una función y que hace.

    Args:
        met_name (str): Texto que da el nombre del método.
        met_des (str): Texto que describe lo que hace el método en cuestión.
    
    Returns:
        None: No se espera argumento de salída.
    """
    print(f'\nEjecutando el método "{met_name}": {met_des}:')
