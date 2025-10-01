import csv
import torch


#inint weights in a csv file
def init_weights(path="/home/programmer/Bachelorarbeit/weights/first_it.csv",path_to_split="/home/programmer/Bachelorarbeit/split/splits.csv"):
    with open(path_to_split, 'r') as csvfile:
        with open(path, 'w') as to_file:
            fieldnames = ['idents','label','weights']
            writer = csv.writer(to_file)
            writer.writerow(fieldnames)
            reader = csv.reader(csvfile)
            weight = 1 / get_size(path_to_split)
            for row in reader:
                if row[1] == "train" or row[1] == "validation":
                    #print(type(row[0]))
                    writer.writerow([int(row[0]),row[1],weight])

def mock_init_weights(path="/home/programmer/Bachelorarbeit/weights/first_it.csv",path_to_split="/home/programmer/Bachelorarbeit/split/splits.csv"):
    with open(path_to_split, 'r') as csvfile:
        with open(path, 'w') as to_file:
            fieldnames = ['idents','label','weights']
            writer = csv.writer(to_file)
            writer.writerow(fieldnames)
            reader = csv.reader(csvfile)
            weight = 1
            for row in reader:
                if row[1] == "train" or row[1] == "validation":
                    writer.writerow([int(row[0]),row[1],weight])
                    weight = weight + 1

#check the size of a csv file given a filter for the second object
# assumes csv file has a header      
def get_size(path="/home/programmer/Bachelorarbeit/split/splits.csv",filter=["train"]) -> int:
    with open(path,'r') as file:
        reader = csv.reader(file)
        size = -1
        for row in reader:
            if row[1] in filter:
                size = size + 1
        return size
#get a dictory with the ids and weights of the data points  
def get_weights(idents:tuple[int,...],path="/home/programmer/Bachelorarbeit/weights/first_it.csv")-> dict[str,float]:
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

def find_label(id:int,path="/home/programmer/Bachelorarbeit/split/splits.csv")-> str:
    with open(path,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == str(id):
                return row[1]


#to do 
# return should be a tuple of weigths matching the sequenece of the target and label tensor
def create_data_weights(batchsize:int,dim:int,weights:dict[str,float],idents:tuple[int,...])-> torch.tensor:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight = torch.empty(batchsize,dim,device=device)
    index = 0
    for i in idents:
        w = weights[str(i)]
        for j in range(0,dim):
            weight[index][j] = float(w)
        index = index + 1
    return weight



def testing():
    print("hello world")

#create a tensor that is size (1,n) where n is the amout of classes being predicted
def create_weight_tensor(weight:float)-> torch.tensor:
    pass







def create_class_weights()-> torch.tensor:
    pass

mock_init_weights()
# print(get_weights((233713,51990)))

