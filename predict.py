from ultralytics import YOLO
from PIL import Image
model = YOLO(r'C:\Projects\gdsc_lemon\best (1).pt')

results = model(['C:\Projects\gdsc_lemon\sign.jpeg'])  # return a list of Results objects

# Process results list
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image