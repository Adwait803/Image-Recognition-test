from torchvision import models, transforms
import torch

from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#model.load_state_dict(torch.load("/home/ak/flower/examples/advanced_pytorch/art/0/0a5c5009edbb43d8a710a57151b9b8b0/artifacts/model/state_dict.pth", map_location=device))

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def predict(image_path):
    model = torch.jit.load('scripted_dummymodel0.6758/data/model.pth')
    model.eval()

    #https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    
    out = model(batch_t)

    #with open('imagenet_classes.txt') as f:
       # classes = [line.strip() for line in f.readlines()]
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    #return [classes[0], prob[0].item() ]
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
