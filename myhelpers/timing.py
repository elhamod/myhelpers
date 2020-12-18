import time

def time_func(title, f, *args):
    start = time.time()
    ans = f(*args)
    end = time.time()
    print('time of ', title, ' : ' , end - start, ' ms')
    return ans