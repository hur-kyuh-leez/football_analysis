from ultralytics import YOLO 
from ENV import FILE_NAME
model = YOLO('models/best.pt')

results = model.predict(f'input_videos/{FILE_NAME}.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)