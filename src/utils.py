import pickle

def load_pickle(path:str):
    with open(file=path,mode='rb') as file:
        pickle_file=pickle.load(file)
    return pickle_file

