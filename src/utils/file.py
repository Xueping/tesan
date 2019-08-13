from src.utils.record_log import _logger
import json
import pickle
import os

def load_squad_dataset(filePath):
    with open(filePath, 'r', encoding='utf-8') as data_file:
        line = data_file.readline()
        dataset = json.loads(line)

    return dataset['data']

def save_file(data, filePath, dataName = 'data', mode='pickle'):
    _logger.add()
    _logger.add('Saving %s to %s' % (dataName,filePath))

    if mode == 'pickle':
        with open(filePath, 'wb') as f:
            pickle.dump(obj=data,file=f)
    elif mode == 'json':
        with open(filePath, 'w', encoding='utf-8') as f:
            json.dump(obj=data,fp=f)
    else:
        raise(ValueError,'Function save_file does not have mode %s' % (mode))
    _logger.add('Done')


def load_file(filePath, dataName = 'data', mode='pickle'):
    _logger.add()
    _logger.add('Trying to load %s from %s' % (dataName, filePath))
    data = None
    is_load = False
    if os.path.isfile(filePath):
        _logger.add('Have found the file, loading...')

        if mode == 'pickle':
            with open(filePath, 'rb') as f:
                data = pickle.load(f)
                is_load = True
        elif mode == 'json':
            with open(filePath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                is_load = True
        else:
            raise (ValueError, 'Function save_file does not have mode %s' % (mode))

    else:
        _logger.add('Have not found the file')
    _logger.add('Done')
    return is_load, data


def save_nn_model(modelFilePath, allParams, epoch):
    _logger.add()
    _logger.add('saving model file to %s' % modelFilePath)
    with open(modelFilePath,'wb') as f:
        pickle.dump(obj=[[param.get_value() for param in allParams ],
                         epoch],
                    file = f)
    _logger.add('Done')

def load_nn_model(modelFilePath):
    _logger.add()
    _logger.add('try to load model file %s' % modelFilePath)
    allParamValues = None
    epoch = 1
    isLoaded = False
    if os.path.isfile(modelFilePath):
        _logger.add('Have found model file, loading...')
        with open(modelFilePath, 'rb') as f:
            data = pickle.load(f)
            allParamValues = data[0]
            epoch = data[1]
            isLoaded = True

    else:
        _logger.add('Have not found model file')
    _logger.add('Done')
    return isLoaded, allParamValues, epoch