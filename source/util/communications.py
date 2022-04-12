import torch
from torch import nn, optim
import syft as sy
import copy
import matplotlib.pyplot as plt

def getParamWeight(store, request_block, delete_obj):
    from collections import OrderedDict
    params = OrderedDict()
    count = len(store.pandas)
    for i in range(count):
        param_tensor = store[0].tags[0]
        print(i, store[0].tags)
        params[param_tensor] = store[0].get(request_block=request_block, delete_obj=delete_obj)
    return params

def getWeightAggregations(dict1, dict2, dict3, **kwargs):
    from collections import OrderedDict
    avg_updates = OrderedDict()
    if len(dict1)!=len(dict2) and len(dict3)!=len(dict2):
        return avg_updates
    count = len(dict1)
    for param in dict1:
        params_avg = (dict1[param] + dict2[param] + dict3[param]) / 3.    
        avg_updates[param] = params_avg
    return avg_updates

def sendData(data, duet, tag, describe, searchable=True):
    data.tag(tag)
    data.describe(describe)
    data_ptr = data.send(duet, searchable=searchable)
    duet.store.pandas
    duet.requests.add_handler(action="accept", print_local=True, )
    return data_ptr

def sendAllData(dict, duet):
    data_ptrs = []
    for param_tensor in dict:
        data_ptr = sendData(dict[param_tensor], duet, "{}".format(param_tensor), "", searchable=True)
        data_ptrs.append(data_ptr)
    return data_ptrs

def getPredict(model, test_dl):
    import torch.nn.functional as F
    y_true = []
    y_pred = []
    y_pred_proba = []
    for inputs, labels in test_dl:
        with torch.no_grad():
            output = model(inputs) # Feed Network
            proba = F.softmax(output.reshape(-1), dim=0)[1]
            # print(output, proba)
            y_pred_proba.append(proba)
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth
    return y_true, y_pred, y_pred_proba
