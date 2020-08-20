"""
A custom class for exceptions without printing the traceback
"""

import sys


class QuietException(Exception):
    """
        An exception that when raised results in the error message
        being printed without the traceback
    """
    pass


def quiet_hook(kind, message, traceback):
    """
        Exception hook that reroutes all exceptions through this method
    
        Args:
            kind (type): the type of the exception
            message (obj): the exception instance
            traceback (traceback): traceback object
    """
   
    if kind.__name__ == "QuietException":
        # only print message
        print('ERROR: {}'.format(message))  
    else:
        # print error type, message & traceback
        sys.__excepthook__(kind, message, traceback)  


# customize handling of exceptions by assigning the exception hook
sys.excepthook = quiet_hook
