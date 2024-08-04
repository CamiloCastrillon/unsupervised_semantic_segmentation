import sys

def error_message(err_des:str) -> None:
    """
    Define un formato para los mensajes de error.

    Args:
        err_des (str): Texto que describe el error.
    
    Returns:
        None: No se espera argumento de salida.
    """
    print(f'\n********************************\n⛔ Error:\n{err_des}\n********************************\n')
    sys.exit()

def warning_message(war_des:str) -> None:
    """
    Define un formato para los mensajes de advertencia con un tiempo de espera de 10 segundos.

    Args:
        war_des (str): Texto que describe la advertencia.
    
    Returns:
        None: No se espera argumento de salida.
    """
    return print(f'\n********************************\n ⚠️ Warning:\n{war_des}\n********************************\n')

def method_menssage(met_name:str, met_des:str) -> str:
    """
    Crea un formato para los mensajes que avisan cuando se está ejecutando una función y que hace.

    Args:
        met_name (str): Texto que da el nombre del método.
        met_des (str): Texto que describe lo que hace el método en cuestión.
    
    Returns:
        str: Texto que describe la ejecución de la función.
    """
    text = f'\n🟢 Ejecutando el método "{met_name}": {met_des}.'
    return print(text)
