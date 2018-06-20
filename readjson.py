import json


def extract_X_and_y(json_str):
    obj = json.loads(json_str)
    X = obj["content"]
    X = X[X.rfind("___")+3:]
    points = obj["annotation"][0]["points"]
    ar = []
    ar.append(points[0]["x"])
    ar.append(points[0]["y"])
    ar.append(points[1]["x"])
    ar.append(points[1]["y"])
    return X, ar
    
def iou(sqr1, sqr2):
    area1 = (sqr1[2]-sqr1[0])*(sqr1[3]-sqr1[1])
    area2 = (sqr2[2]-sqr2[0])*(sqr2[3]-sqr2[1])
    intersection=[]
    intersection.append(max(sqr1[0], sqr2[0]))
    intersection.append(max(sqr1[1], sqr2[1]))
    intersection.append(min(sqr1[2], sqr2[2]))
    intersection.append(min(sqr1[3], sqr2[3]))
    if(intersection[0] > intersection[2] or intersection[1] > intersection[3]):
        return 0
    inter = (intersection[2]-intersection[0])*(intersection[3]-intersection[1])
    return inter / (area1+area2-inter)
    
    

def calc_loss(y_true, y_pred):
    if y_true[0] == 0 and y_pred[0]>=0.7:
        return 1
    elif y_true[0] == 1 and y_pred[0]<0.7:
        return 1
    elif y_true[0] == 1 and y_pred[0]>=0.5:
        val_iou = iou(y_true[1:], y_pred[1:])
        if val_iou>0.5:
            return 0
        else:
            return 0.5-val_iou
    else:
        return 0
"""
print(iou([0.1,0.1,0.3,0.3], [0.7,0.7,0.9,0.9]))
print(iou([0.1,0.1,0.4,0.4], [0.2,0.2,0.5,0.5]))
print(iou([0.2,0.2,0.5,0.5], [0.3,0.1,0.4,0.6]))
print(iou([0.3,0.3,0.6,0.6], [0.25,0.35,0.65,0.55]))
print(iou([0.3,0.3,0.7,0.7], [0.35,0.35,0.65,0.65]))



f = open("Israeli license plates.json", "r")
for line in f:
    print(extract_X_and_y(line))
f.close()
"""
