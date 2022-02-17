import time
import os
import shutil

def clock(func):
    r"""
    time counter
    """
    def clocked(*args, **kw):
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        print('[%0.8fs] %s' % (elapsed, name))
        return result
    return clocked

class TrainLogger():
    r"""
    Description:
    -----------
    Logger class. Function:
    - print all training hyperparameter setting
    - print all model    hyperparameter setting
    - save all the python file of model

    Args:
    -----------
    path: str
        Log path
    """
    
    def __init__(self, model_name, dataset):
        path        = 'log/'
        cur_time    = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cur_time    = cur_time.replace(" ", "-")
        # mkdir
        os.makedirs(path + cur_time)
        # pwd = os.getcwd() + "/"
        # copy model files
        shutil.copytree('models',  path + cur_time + "/models")      # copy models
        shutil.copytree('configs',  path + cur_time + "/configs")      # copy models
        shutil.copyfile('main.py',  path + cur_time + "/main.py")      # copy models
        # backup model parameters
        try:
            shutil.copyfile('output/' + model_name + "_" + dataset + ".pt", path + cur_time + "/"  + model_name + "_" + dataset + ".pt")
            shutil.copyfile('output/' + model_name + "_" + dataset + "_resume" + ".pt", path + cur_time + "/" + model_name + "_" + dataset + "_resume.pt")
        except:
            # No model_para.pt
            pass
    def __print(self, dic, note=None, ban=[]):
        print("=============== " + note + " =================")
        for key,value in dic.items():
            if key in ban:
                continue
            print('|%20s:|%20s|' % (key, value))
        print("--------------------------------------------")

    def print_model_args(self, model_args, ban=[]):
        self.__print(model_args, note='model args', ban=ban)

    def print_optim_args(self, optim_args, ban=[]):
        self.__print(optim_args, note='optim args', ban=ban)
