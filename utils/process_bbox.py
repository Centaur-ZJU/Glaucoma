import numpy as np
import json
import os

def compute_meanBox(json_path = "D:/Project_Glaucoma/dataset/bbox_label.json"):
    total_box = np.zeros(4)

    with open(json_path, 'r') as f:
        dataset = json.load(f)
    for data in dataset:
        # show(os.path.join(root,data["img"]),[data["bbox"]])
        x,y,w,h = data['bbox']
        W,H = data['size']
        total_box += np.array([x/W, y/H, w/W, h/H])
    return 1.0*total_box/len(dataset)

if __name__ == "__main__":
    print(compute_meanBox())