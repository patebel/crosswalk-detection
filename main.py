import ultralytics
from ultralytics import YOLO


ultralytics.checks()



epochs = 300 #@param {type:"slider", min:100, max:1000, step:10}
training_option = "Train new CNN" #@param ["Train new CNN", "Resume an interrupted Training"]
weight = 'yolov8m.pt' #@param["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt", "yolov8x"]
batch = 32 #@param {type:"slider", min:1, max:64, step:1}

model = YOLO(weight)
data =  "/content/gdrive/MyDrive/"+folder+"/training/training_dataset/data.yaml"
project = "/content/gdrive/MyDrive/"+folder+"/training/training_results/"
model_folder = "/content/gdrive/MyDrive/"+folder+"/training/training_results/train/weights/last.pt"

print(type(model))


