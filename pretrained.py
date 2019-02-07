import torch
from torchvision import transforms,models
import numpy as np
import cv2 as cv
from PIL import Image
from keras.applications.imagenet_utils import decode_predictions

cap=cv.VideoCapture(0)
model=models.vgg19(pretrained=True)
model.cuda()
model.to('cuda')
for param in model.parameters():
    param.requires_grad=False

while True:
    ret,frame=cap.read()
    f=Image.fromarray(frame)
    transform=transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    f=transform(f)
    f.unsqueeze_(0)
    with torch.no_grad():
        model.to('cuda')
        out=model(f.cuda())
    ix,cl,prob=decode_predictions(np.array(out.cpu()))[0][0]
    cv.putText(frame,cl+" Prob: "+str(prob),(10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv.imshow('object',frame)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break
