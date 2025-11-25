import torch
import csv
import numpy


train = 0


def create_weight(path_to_split="/home/programmer/Bachelorarbeit/split/splits.csv"):
    weights = {}
    with open(path_to_split, 'r') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if (row[1] == "train") and i > 0:
                #print(row[0])
                weights[row[0]] = torch.full((1,1528),int(row[0]))
                #print(row[0])
            i = i +1
        print(len(weights))
    torch.save(weights,"/home/programmer/Bachelorarbeit/weights/init_mh.pt")



def new_create_weight(path_to_split="/home/programmer/Bachelorarbeit/split/splits.csv"):
    weights = {}
    with open(path_to_split, 'r') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if (row[1] == "train") and i > 0:
                # print(row[0])
                weights[row[0]] = [1/1528 * 160716 ]* 1528
                # print(row[0])
            i = i + 1
        print(len(weights))
    torch.save(weights, "../../weights/init_mh.pt")


def add_train_weights(ids):
    d = torch.load("/home/programmer/Bachelorarbeit/weights/init_mh.pt",weights_only=False)
    it = 0
    for i in ids:
        if it % 10000 == 0:
            print(it)
        ident = i["ident"]
        i["weight"] = d[str(ident)]
        it = it + 1
    return ids

def add_val_weights(ids):
    for i in ids:
        weight = 1
        #i["weight"] = torch.full((1,1528),1)
        i["weight"] = [1]*1528

    return ids

def create_data_weights(batchsize:int,dim:int,weights:dict[str,list[float,...]],idents:tuple[int,...])-> torch.tensor:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight = None
    index = 0
    for i in idents:
        w = torch.Tensor([weights[str(i)],]).to(device)
        if weight == None:
            weight = w
        else:
            weight = torch.cat((weight,w),0)
        index = index + 1
    return weight

#new_create_weight()
#create_weight()
