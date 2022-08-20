
import sys
import torch

class MyTools:

    def __init__(self):
        self.name = 'my'

    def func_cal_accu_recall(logits_list=None, y_list=None, y_len_list=None, is_logits_tag=False):
        eval_dict = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'others': 0, 'n_total': 0}
        n_total = 0
        for logit_i, logit in enumerate(logits_list):
            assert logit.shape[0:2] == y_list[logit_i].shape
            batch_len = len(y_len_list[logit_i])
            batch_max_seqlen = max(y_len_list[logit_i])
            y_curr = y_list[logit_i]
            logit_argmax = logit if is_logits_tag else torch.argmax(logit, dim=2)
            logit_argmax = logit_argmax.cuda()
            # get mask matrix
            mask = torch.zeros((batch_len, batch_max_seqlen))
            for mask_i in range(mask.shape[0]):
                mask[mask_i][0:y_len_list[logit_i][mask_i]] = 1
                
            assert sum(mask.sum(axis=1) == y_len_list[logit_i]) // batch_len == 1
            
            # 这一部分是因为加入了start和stop的tag，需要遮盖掉计算
            mask[:,0] = 0 # start
            for mask_i in range(mask.shape[0]):
                mask[mask_i][y_len_list[logit_i][mask_i]-1] = 0 # stop
            
            # cal the tp,tn,fp,fn in this batch
            N = mask.sum()
            TP = ((logit_argmax > 0) * (y_curr > 0) * (logit_argmax == y_curr) * mask.cuda()).sum()
            TN = ((logit_argmax == 0) * (y_curr == 0) * (logit_argmax == y_curr) * mask.cuda()).sum()
            FP = ((logit_argmax > 0) * (y_curr == 0) * (logit_argmax != y_curr) * mask.cuda()).sum()
            FN = ((logit_argmax == 0) * (y_curr > 0) * (logit_argmax != y_curr) * mask.cuda()).sum()
            others = ((logit_argmax > 0) * (y_curr > 0) * (logit_argmax != y_curr) * mask.cuda()).sum()

            eval_dict['tp'] += TP.item()
            eval_dict['tn'] += TN.item()
            eval_dict['fp'] += FP.item()
            eval_dict['fn'] += FN.item()
            eval_dict['others'] += others.item()
            eval_dict['n_total'] += N.item()

            accu = (TP + TN) / N
            recall = TP / (TP + FN)

        if False:
            print('Total accu = {:.2f}% recall = {:.2f}%'.format(
                (eval_dict['tp'] + eval_dict['tn']) / eval_dict['n_total'] * 100,
                eval_dict['tp'] / (eval_dict['tp'] + eval_dict['fn']) * 100))
        return eval_dict
    
    def func_cal_metrics(eval_dict):
        recall = eval_dict['tp'] / (eval_dict['tp'] + eval_dict['fn'] + eval_dict['others'])
        precision = eval_dict['tp'] / (eval_dict['tp'] + eval_dict['fp'])
        accuracy = (eval_dict['tp'] + eval_dict['tn']) / eval_dict['n_total']
        f1 = 2 * recall * precision / (recall + precision)
        print('recall = {:.2f}%, precision = {:.2f}%, accuracy = {:.2f}%, f1 = {:.2f}%'.format(recall*100,precision*100,accuracy*100,f1*100))
        return recall, precision, accuracy, f1