import ultralytics
from ultralytics import YOLO


ultralytics.checks()

if __name__ == '__main__':
    epochs = 300 #@param {type:"slider", min:100, max:1000, step:10}
    training_option = "Train new CNN" #@param ["Train new CNN", "Resume an interrupted Training"]
    weight = 'yolov8m.pt' #@param["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt", "yolov8x"]
    batch = 32 #@param {type:"slider", min:1, max:64, step:1}

    model = YOLO(weight)
    data = "training/training_dataset/data.yaml"
    project = "training/training_results/"
    model_folder = "training/training_results/train/weights/last.pt"

    print(type(model))


    if training_option == 'Train new CNN':
      model.train(data=data, epochs=epochs, imgsz=512, batch=batch, cache=True, patience=50, project=project)
    else:
      model.train(data=data, epochs=epochs, imgsz=512, batch=batch, cache=True, patience=50, project=project, resume=True, model=model_folder)

    data_test = "/test/test_dataset/data.yaml"
    metrics = model.val(data=data_test)
