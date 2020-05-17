import os
import json

root = "D:/Project_Glaucoma/dataset/JSONS"
json1 = os.path.join(root,"ORIGA_glaucoma.json")
json2 = os.path.join(root,"ORIGA_sanas.json")

data = []
with open(json1,"r") as f:
    data1 = json.load(f)
with open(json2,"r") as f:
    data2 = json.load(f)

for a in data1:
    a["img"] = "ORIGA/" + a["img"]
    data.append(a)

for b in data2:
    b["img"] = "ORIGA/" + b["img"]
    data.append(b)


with open(os.path.join(root, "ORIGA.json"), "w") as f:
    json.dump(data,f)