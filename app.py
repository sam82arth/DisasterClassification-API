import  flask
import torch
from flask import request
import numpy as np
from PIL import Image
import requests


app = flask.Flask(__name__)
app.config["DEBUG"] = True

classes = ["Cyclone", "Earthquake", "Flood", "Wildfire"]
@app.route('/img', methods=['GET'])
def login():
    url = request.form.get('url')
    print(url)
    model = (torch.load('./model2.h5'))
    img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    from torchvision import transforms
        
    preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        
    img_preprocessed = preprocess(img)
        
    batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
    out = model(batch_img_tensor)
    conv_int = (out.cpu().detach().numpy())
    _,index = np.where(conv_int == conv_int.max())
    index[0]
    return (classes[index[0]])

       
if __name__ == "__main__" :
    app.run()