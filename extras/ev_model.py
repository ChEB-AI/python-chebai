def create_weight_dict(p,l,data_list):
    result = []
    i = 0
    for j in range(0,len(p)):
        for k in range(0,len(p[j])):
            d = {}
            pred = p[j][k]
            label = l[j][k]
            ident = data_list[i]["idents"]
            d["pred"]= pred
            d["label"]= label
            d["ident"]= ident
            result.append(d)
            i = i + 1
    return result