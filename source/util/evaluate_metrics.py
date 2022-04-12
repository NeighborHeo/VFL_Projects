import torch
from torch import nn, optim
import syft as sy
import copy
import matplotlib.pyplot as plt

def draw_confusion_matrix(y_true, y_pred, output_path, output_name):
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import seaborn as sns
    import numpy as np
    cf = confusion_matrix(y_true, y_pred, labels=[1,0])
    cf_norm = confusion_matrix(y_true, y_pred, normalize='true', labels=[1,0])
    group_names = ['TP', 'FN', 'FP', 'TN']
    group_counts = ['{}'.format(value) for value in cf.flatten()]
    group_percentages = ['{:.2}'.format(value) for value in cf_norm.flatten()]
    labels = ['{}\n{}\n({})'.format(v1, v2, v3) for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    # pd.DataFrame((cf_matrix/np.sum(cf_matrix) *100), index = [i for i in range(config['classes'])], \
    # columns = [i for i in range(config['classes'])])
    cm_figure = sns.heatmap(cf_norm, annot=labels, fmt='', xticklabels=['1','0'], yticklabels=['1','0'], cmap='Blues')
    cm_figure.set_title('Confusion matrix')
    cm_figure.set_xlabel('Predicted label')
    cm_figure.set_ylabel('True label')
    plt.setp(cm_figure.get_yticklabels(), rotation=0)
    plt.savefig('{}/{}_confusion_matrix.png'.format(output_path,output_name), format='png',
                dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')
    return

def model_performance_evaluation(y_true, y_pred, y_pred_proba, output_path, output_name):
    from sklearn.metrics import confusion_matrix, roc_auc_score

    cf = confusion_matrix(y_true, y_pred, labels=[1,0])
    TP, FP, FN, TN = cf[0][0], cf[0][1], cf[1][0], cf[1][1]
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    result = {}
    result['TP'] = TP
    result['TN'] = FP
    result['FP'] = FN
    result['FN'] = TN
    result['precision'] = TP/(TP+FP)
    result['specificity'] = TN/(TN+FP)
    result['sensitivity'] = TP/(TP+FN) 
    result['recall'] = result['sensitivity'] # recall = sensitivity
    result['accuracy'] = (TP+TN) / (FP+FN+TP+TN)
    result['f1score'] = 2*result['precision']*result['recall']/(result['precision']+result['recall'])
    result['roc_auc'] = roc_auc
       
    out = open('{}/model_performance_evaluation.txt'.format(output_path),'a')
    out.write(str(output_name) + '///')
    out.write(str(TP) + '///')
    out.write(str(TN) + '///')
    out.write(str(FP) + '///')
    out.write(str(FN) + '///')
    out.write('{:.3}'.format(result['precision']) + '///')
    out.write('{:.3}'.format(result['specificity']) + '///')
    out.write('{:.3}'.format(result['accuracy']) + '///')
    out.write('{:.3}'.format(result['recall'])+ '///')
    out.write('{:.3}'.format(result['f1score'])+ '///' )
    out.write('{:.3}'.format(result['roc_auc']) )
    out.write('\n')        
    out.close()
    
    print("TP", TP) 
    print("TN", TN)
    print("FP", FP)
    print("FN", FN)
    print("precision", '{:.3%}'.format(result['precision']))
    print("specificity", '{:.3%}'.format(result['specificity']))
    print("accuracy", '{:.3%}'.format(result['accuracy']))
    print("recall", '{:.3%}'.format(result['recall']))
    print("f1_score", '{:.3%}'.format(result['f1score']))    
    print("roc_auc", '{:.3%}'.format(result['roc_auc'])) 
    # from pycm import *
    # cm = ConfusionMatrix(actual_vector=np.array(c.y_test.ravel()), predict_vector=np.array(c.y_pred.ravel()))
    # cm.classes
    # cm.table
    # print(cm)
    return

def draw_history_plot(hist_dict, output_path, output_name):
    """
    :param hist_dict: [dict,list] (train_acc, train_loss, valid_acc, valid_loss)
    :return: None
    """
    plt.figure()
    plt.plot( hist_dict['valid_loss'], marker='.', c='red', label="Valid-set Loss")
    plt.plot( hist_dict['train_loss'], marker='.', c='blue', label="Train-set Loss")
    plt.plot( hist_dict['valid_acc'], marker='.', c='red', label="Valid-set acc")
    plt.plot( hist_dict['train_acc'], marker='.', c='blue', label="Train-set acc")

    plt.legend(loc='upper right')
    plt.grid()
    plt.title('acc/loss plot')
    plt.xlabel('epoch')
    plt.ylabel('acc/loss')
    plt.show()
    plt.savefig("{}/{}_acc_loss.png".format(output_path, output_name))

    return

def draw_ROC_AUC(y_true, y_pred, output_path, output_name):
    
    # Compute ROC curve and ROC area for each class
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    AUC = round(roc_auc, 4)

    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    plt.savefig('{}/{}_ROC.png'.format(output_path, output_name) , format='png',
                dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')

    from sklearn.metrics import accuracy_score
    y_hat2 = np.where(y_hat>0.5, 1, 0)
    ACC = round(accuracy_score(y_hat2, y_test), 5)*100
    print("ACC:",ACC)
    
    # to draw mean ROC Curve
    np.savez('{}/{}_dat.npz'.format(output_path, output_name), y_hat=y_hat, y_test=y_test)
    
    return AUC, ACC

def draw_ROC_AUC(y_true, y_pred_proba, output_path, output_name):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.plot(fpr, tpr, label='ROC curve (area = {:.3f})'.format(roc_auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    plt.savefig('{}/{}_ROC.png'.format(output_path, output_name) , format='png',
                dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')