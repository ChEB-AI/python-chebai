import csv
import torch
import os

class_weights = True

#inint weights in a csv file
def init_weights(path="../weights/first_it.csv",path_to_split="../split/splits.csv"):
    if not os.path.exists("../weights/first_it.csv"):
        print("init weights ....")
        with open(path_to_split, 'r') as csvfile:
            with open(path, 'w') as to_file:
                fieldnames = ['idents','label','weights']
                writer = csv.writer(to_file)
                writer.writerow(fieldnames)
                reader = csv.reader(csvfile)
                weight = 1 / get_size(path_to_split)
                for row in reader:
                    if row[1] == "train":
                        #print(type(row[0]))
                        writer.writerow([int(row[0]),row[1],weight])

def mock_init_weights(path="../weights/first_it.csv",path_to_split="../split/splits.csv"):
    with open(path_to_split, 'r') as csvfile:
        with open(path, 'w') as to_file:
            fieldnames = ['idents','label','weights']
            writer = csv.writer(to_file)
            writer.writerow(fieldnames)
            reader = csv.reader(csvfile)
            weight = 1
            for row in reader:
                if row[1] == "train":
                    writer.writerow([int(row[0]),row[1],weight])
                    weight = weight + 1

#check the size of a csv file given a filter for the second object
# assumes csv file has a header      
def get_size(path="../split/splits.csv",filter=["train"]) -> int:
    with open(path,'r') as file:
        reader = csv.reader(file)
        size = -1
        for row in reader:
            if row[1] in filter:
                size = size + 1
        return size
#get a dictory with the ids and weights of the data points  
def get_weights(idents:tuple[int,...],path="../weights/first_it.csv")-> dict[str,float]:
    value = dict()
    for i in idents:
        weight = find_weight(path,i)
        value.update({str(i):weight})
    return value

#finds the weight for a specific datapoint  
def find_weight(path:str,ident:int)-> float:
    with open(path,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == str(ident):
                return float(row[2])
            
    label = find_label(id=ident)
    print(f"{ident} is not in file with {label} ")

def find_label(id:int,path="../split/splits.csv")-> str:
    with open(path,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == str(id):
                return row[1]


#to do 
# return should be a tuple of weigths matching the sequenece of the target and label tensor
def create_data_weights(batchsize:int,dim:int,weights:dict[str,float],idents:tuple[int,...])-> torch.tensor:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight = None
    index = 0
    for i in idents:
        w = torch.full((1,dim),float(weights[str(i)]),device=device)
        if weight == None:
            weight = w
        else:
            weight = torch.cat((weight,w),0)
        index = index + 1
    return weight

def testing():
    print("hello world")

#create a tensor that is size (1,n) where n is the amout of classes being predicted
def create_weight_tensor(weight:float)-> torch.tensor:
    pass

def add_val_weights(ids):
    for i in ids:
        weight = 1
        i["weight"] = weight
    return ids


def add_train_weights(ids):
    it = 0
    for i in ids:
        if it % 10000 == 0:
            print(it)
        ident = i["ident"]
        weight = find_weight("/home/programmer/Bachelorarbeit/weights/first_it.csv",ident=ident)
        i["weight"] = weight
        it = it +1
    return ids

def check_weights(data):
    for i in data:
        print(f"({i["ident"]} , {i["weight"]}")


def init_class_weights(class_path:str,weight_path:str,weight:float):
    with open(class_path,'r') as classes:
        with open(weight_path,'w') as weights:
            reader = csv.reader(classes)
            writer = csv.writer(weights)
            writer.writerow(["class","weight"])
            for row in reader:
                row = row + [weight,]
                writer.writerow(row)

def create_class_tensor(save_path:str)-> torch.Tensor:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = torch.empty(1,1528)
    with open("../../weights/class_first_it.csv",'r') as f:
        reader = csv.reader(f)
        index = 0
        for row in reader:
            if row[1] == "weight":
                continue
            t[0][index] = float(row[1])
            index = index + 1
    torch.save(t,save_path)

def create_weight_class_tensor(batch_size:int)-> torch.Tensor:
    t = torch.load("../../weights/test.pt")
    w = None
    for i in range(0,batch_size):
        if w is None:
            w = t
        else:
            w = torch.cat((w,t),dim=0)
    print(w.shape)
    return w





#init_class_weights("../../data/chebi_v241/ChEBI50/processed/classes.txt","../../weights/class_first_it.csv",1)
create_class_tensor("../../weights/test.pt")
create_weight_class_tensor(32)