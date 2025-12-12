import torch
import csv
import numpy



class Ensemble_loader():

    def __init__(
            self,
            #True :bagging, False : boosting
            ensemble:bool,
            load_path:str,
            dim:int,
    ):
        self.ensemble=ensemble
        self.load_path=load_path
        self.dim=dim




    def add_train_weights(self,ids,load_path):
        d = torch.load(load_path,weights_only=False)
        print("start")
        it = 0
        for i in ids:
            if it % 10000 == 0:
                print(it)
            ident = i["ident"]
            i["weight"] = d[str(ident)]
            it = it + 1
        return ids

    def add_val_weights(self,ids,dim):
        for i in ids:
            i["weight"] = [1]*dim
        return ids
    #dict reverse to the dict created by the method bootstrapping in sample.py
    def add_duplicates(self,data,load_path):
        path_to_dict = load_path
        d = torch.load(path_to_dict,weights_only=False)
        length = len(data)
        for i in range(0,length):
            ident = data[i]["ident"]
            if(d[str(ident)] > 1):
                r = d[str(ident)]
                for j in range(0,r-1):
                    data.append(data[i])


        return data


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

# def create_weight(path_to_split="/home/programmer/Bachelorarbeit/split/splits.csv"):
#     weights = {}
#     with open(path_to_split, 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         i = 0
#         for row in reader:
#             if (row[1] == "train") and i > 0:
#                 #print(row[0])
#                 weights[row[0]] = torch.full((1,1528),int(row[0]))
#                 #print(row[0])
#             i = i +1
#         print(len(weights))
#     torch.save(weights,"/home/programmer/Bachelorarbeit/weights/init_mh.pt")


#for 1_ada_no_normal_weights weights =0.0001
def new_create_weight(path_to_split="/home/programmer/Bachelorarbeit/split/reworked_splits.csv"):
    weights = {}
    with open(path_to_split, 'r') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if (row[1] == "train") and i > 0:
                # print(row[0])
                weights[row[0]] = [(1/(1528*160677))*10000]* 1528
                # print(row[0])
            i = i + 1
        print(len(weights))
    torch.save(weights, "/home/programmer/Bachelorarbeit/weights/init_mh_10000.pt")




















#new_create_weight()

