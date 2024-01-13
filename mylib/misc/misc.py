import numpy as np


def target_extract(_fnames, _trues, _preds, _probs, target, mode="pap5"):
    fnames, trues, preds, probs = [], [], [], []
    for i in range(len(_fnames)):
        if target == "all":
            fnames.append(_fnames[i])
            trues.append(_trues[i])
            preds.append(_preds[i])
            
            if mode == "pap5":
                probs.append(
                    [_probs["class1"][i], _probs["class2"][i], _probs["class3"][i], _probs["class4"][i], _probs["class5"][i]]
                )
            elif mode == "beth4":
                probs.append(
                    [_probs["NILM"][i], _probs["OLSIL"][i], _probs["OHSIL"][i], _probs["SCC"][i]]
                ) 
        else:
            if _fnames[i].split('_')[0] == target:
                fnames.append(_fnames[i])
                trues.append(_trues[i])
                preds.append(_preds[i])
                
                if mode == "pap5":
                    probs.append(
                        [_probs["class1"][i], _probs["class2"][i], _probs["class3"][i], _probs["class4"][i], _probs["class5"][i]]
                    )
                elif mode == "beth4":
                    probs.append(
                        [_probs["NILM"][i], _probs["OLSIL"][i], _probs["OHSIL"][i], _probs["SCC"][i]]
                    )
                
        print(f"\rprogress {i+1}/{len(_fnames)}", end='')
    print(" >> Done")
    print(fnames[0], trues[0], preds[0], probs[0])
    
    return fnames, trues, preds, probs


def s_to_hms(td):
    m, s = divmod(td, 60)
    h, m = divmod(m, 60)
    
    return h, m, s


def count_elements(li, num_classes):
    counter = np.zeros(num_classes)
    for l in li:
        counter[l] += 1
    return counter