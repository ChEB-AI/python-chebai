import torch
import csv
import copy



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

    def find_id(self,id, d):
        for i in range(0, len(d)):
            if int(id) == d[i]["ident"]:
                return i

    def resample_dataset(self,d, load_path):

        t = torch.load(load_path, weights_only=False)
        l = torch.load("",
                       weights_only=False)
        for i in t:
            # print(i)
            check = str(i).split("m")
            if len(check) == 1:
                if t[i] > 1:
                    index = self.find_id(check[0], d)
                    instance = d[index]
                    for k in range(0, t[i] - 1):
                        d.append(instance)
                    t[i] = 0
                else:
                    t[i] = 0

            else:
                if t[i] > 0:
                    # it is a split instance
                    index = self.find_id(check[0], d)

                    if t[check[0] + "mi"] > 0:
                        mi = copy.deepcopy(d[index])
                        # get majority labels to set weight to zero
                        positives = l[check[0] + "ma"]
                        for k in positives:
                            mi["weight"][k] = 0
                        for j in range(0, t[check[0] + "mi"]):
                            d.append(mi)

                    if t[check[0] + "ma"] > 0:
                        # get minority labels to set weight to zero
                        ma = copy.deepcopy(d[index])
                        positive = l[str(check[0]) + "mi"]
                        for k in positive:
                            ma["weight"][k] = 0
                        for j in range(0, t[check[0] + "ma"]):
                            d.append(ma)
                    t[check[0] + "ma"] = 0
                    t[check[0] + "mi"] = 0
                    d.pop(index)
        return d

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

