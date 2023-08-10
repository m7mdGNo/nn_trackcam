from ultralytics import YOLO
import cv2

def detect_persons(model,frame,conf=0.5):
    names = model.names
    results = model(frame,conf=conf,verbose=False)
    persons = []
    for result in results[0]:
        classes = result.boxes.cls.tolist()
        bboxs = result.boxes.xyxy.tolist()
        for i,bbox in enumerate(bboxs):
            index = int(classes[i])
            class_name = names[index]
            x1,y1,x2,y2 = [int(p) for p in bbox]
            if class_name == 'person':
                persons.append((x1,y1,x2,y2))
    return persons

# Example usage
def main():
    cap = cv2.VideoCapture(0)

    model = YOLO('yolov8s.pt')

    while True:
        ret,frame = cap.read()
        persons = detect_persons(model,frame)
        for person in persons:
            x1,y1,x2,y2 = person
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    main()