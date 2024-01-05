import torch
from ultralytics import YOLO
import cv2

def rescale_frame(frame, scale):    # works for image, video, live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

my_image = "C:\\Users\\bilgi\\OneDrive\\Masaüstü\\WhatsApp Image 2023-12-31 at 16.43.17_30f0f4a4.jpg"

# Load the YOLO model
model = YOLO('C:\\Users\\bilgi\\OneDrive\\Masaüstü\\code\\AI\go br\\AI v3\\weights\\best.pt')

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
modell = model.to(device)

# Load the image
img = cv2.imread(my_image)

# Run inference on your image
results = modell(my_image, conf=0.5)
result = results[0]
boxs = result.boxes

# Define the font type
font = cv2.FONT_HERSHEY_SIMPLEX

# Define the font scale factor that is multiplied by the font-specific base size
fontScale = 1

# Define the color of the text string to be drawn
color = (255, 0, 0)

# Define the thickness of the line in px
thickness = 2

total = 0

for box in boxs:
    if box.cls == 0:
        print("100_tl")
        text = "100_tl"
        value = 100
    elif box.cls == 1:
        print("10_tl")
        text = "10_tl"
        value = 10
    elif box.cls == 2:
        print("200_tl")
        text = "200_tl"
        value = 200
    elif box.cls == 3:
        print("20_tl")
        text = "20_tl"
        value = 20
    elif box.cls == 4:
        print("50_tl")
        text = "50_tl"
        value = 50
    elif box.cls == 5:
        print("5_tl")
        text = "5_tl"
        value = 5
    total = total + value
    xmin, ymin, xmax, ymax = box.xyxy[0]
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    org = (int(xmin), int(ymin))

    img = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    print(box)
    print("-----------------------------------------")
    
img = rescale_frame(img, 0.5)

print(total)

# Display the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
