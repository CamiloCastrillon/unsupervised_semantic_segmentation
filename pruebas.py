import time
import os
from resources.message import method_menssage

def fun():
    method_menssage(fun.__name__, 'La función hace lo que se le da la gana')
    print('     Ejecución de la función')
fun()
