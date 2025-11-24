import torch
import csv


train = 0


def create_weight(path_to_split="../../split/splits.csv"):
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
    torch.save(weights,"../../weights/init_mh.pt")

def add_train_weights(ids):
    d = torch.load("/home/programmer/Bachelorarbeit/weights/init_mh.pt",weights_only=False)
    global train
    train = train + 1
    print(train)
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
        i["weight"] = torch.full((1,1528),1)
    return ids

def create_data_weights(batchsize:int,dim:int,weights:dict[str,torch.Tensor],idents:tuple[int,...])-> torch.tensor:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight = None
    index = 0
    for i in idents:
        w = weights[str(i)]
        if weight == None:
            weight = w
        else:
            weight = torch.cat((weight,w),0)
        index = index + 1
    return weight

