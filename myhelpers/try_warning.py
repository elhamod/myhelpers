import warnings

def try_running(a, msg):
    result = None
    # try:
    result= a()
    # except:
    #     warnings.warn(msg, Warning)
    return result
    