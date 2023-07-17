# !pip install torchvision
# !pip install kaggle
# !pip install tqdm
# !pip install colorama

# ! export KAGGLE_USERNAME="Your_Kaggle_username" && export KAGGLE_KEY="Your_Kaggle_APIKey" && kaggle datasets download --unzip amarinderplasma/alphabets-sign-language
# ! ls

# ! pwd && ls
# print("\nFolders(classess) in training folder: ...")
# ! cd asl_alphabet_train && ls

import torch
from torch import nn,optim
from torchvision import transforms, models ,datasets
import numpy as np
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.axes_grid1 import ImageGrid

%matplotlib inline

ASL=np.array(glob.glob('asl_alphabet_test/*')) 

fig = plt.figure(figsize=(30, 30))
grid = ImageGrid(fig, 111, 
                 nrows_ncols=(4, 7),  
                 axes_pad=0,  
                 )
l=0
for img in ASL:
        im=plt.imread(img)
        grid[l].imshow(im,cmap='gray',interpolation='nearest')
        grid[l].text(5,20, img.split('/')[1].split('_')[0] ,fontsize=30)
        l+=1

train_path='asl_alphabet_train'
valid_path='asl_alphabet_valid'

train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(p=0.3),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_path, transform=train_transforms)
test_data = datasets.ImageFolder(valid_path, transform=test_transforms)

testsamples=torch.utils.data.Subset(test_data, list(range(0, len(test_data), 5)))

trainloader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True)
testloader = torch.utils.data.DataLoader(testsamples,batch_size=512)


print(f"Device used: cpu")


print(f"class to index mapping: {train_data.class_to_idx}")
len(testloader)

model = models.mobilenet_v2(pretrained=True)


for param in model.parameters():
    param.requires_grad = False
    

print (model.classifier)

model.classifier= nn.Sequential(nn.Dropout(p=0.6, inplace=False),
                                nn.Linear(in_features=1280, out_features=29, bias=True),
                                nn.LogSoftmax(dim=1))


for p in model.features[-1].parameters():
    p.requires_grad = True  

    

criterion = nn.NLLLoss()


optimizer = optim.Adam([{'params':model.features[-1].parameters()},
                        {'params':model.classifier.parameters()}], lr=0.0005)


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)


print(model.classifier)


%matplotlib inline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


model.eval()


test_data_t = datasets.ImageFolder(valid_path,transforms.Compose([transforms.ToTensor()]))
testloader_t = torch.utils.data.DataLoader(test_data_t, batch_size=200,shuffle=True)
images_t , labels_t=next( iter(testloader_t) )

index = np.random.randint(0, 199)
test_img=images_t[index]


t=transforms.ToPILImage()
plt.imshow(t(test_img))

t_n=transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
test_img=t_n(test_img).unsqueeze(0)


res = torch.exp(model(test_img))


classes=train_data.class_to_idx
classes = {value:key for key, value in classes.items()}

print(f"image number {index}")
print("---------------------")


print("label:",classes[labels_t[index].item()])


print("prediction:", classes[res.argmax().item()])

import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
epochs = 1
step = 0
running_loss = 0
print_every = 30
trainlossarr=[]
testlossarr=[]
oldacc=0

steps=math.ceil(len(train_data)/(trainloader.batch_size))

%%time
from tqdm import tqdm
import sys
from colorama import Fore,Style

for epoch in range(epochs):
    print(Style.RESET_ALL)
    print(f"--------------------------------- START OF EPOCH [ {epoch+1} ] >>> LR =  {optimizer.param_groups[-1]['lr']} ---------------------------------\n")
    for inputs, labels in tqdm(trainloader,desc=Fore.GREEN + f"* progess in EPOCH {epoch+1} ",file=sys.stdout):
        model.train()
        step += 1
        inputs=inputs.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        props = model.forward(inputs)
        loss = criterion(props, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (step % print_every == 0) or (step==steps):
            test_loss = 0
            accuracy = 0
            model.eval()
            tqdm._instances.clear()
            with torch.no_grad():
                for inputs, labels in tqdm(testloader,desc=Fore.BLUE + f"* CALCULATING TESTING LOSS {epoch+1} ",file=sys.stdout,leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    props = model.forward(inputs)
                    batch_loss = criterion(props, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(props)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
 
                    
                        
            print(Style.RESET_ALL)
            tqdm.write(f"Epoch ({epoch+1} of {epochs}) ... "
                  f"Step  ({step:3d} of {steps}) ... "
                  f"Train loss: {running_loss/print_every:.3f} ... "
                  f"Test loss: {test_loss/len(testloader):.3f} ... "
                  f"Test accuracy: {accuracy/len(testloader):.3f} ")
            trainlossarr.append(running_loss/print_every)
            testlossarr.append(test_loss/len(testloader))
            running_loss = 0
            
        
    scheduler.step()
    step=0

%matplotlib inline


model.eval()

test_data = datasets.ImageFolder(valid_path,transforms.Compose([transforms.ToTensor()]))
testloader = torch.utils.data.DataLoader(test_data, batch_size=200,shuffle=True)
images , labels=next( iter(testloader) )

index = np.random.randint(0, 199)
test_img=images[index]


t=transforms.ToPILImage()
plt.imshow(t(test_img))


t_n=transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
test_img=t_n(test_img).unsqueeze(0)


res = torch.exp(model(test_img))

classes=train_data.class_to_idx
classes = {value:key for key, value in classes.items()}

print(f"image number {index}")
print("---------------------")


print("label:",classes[labels[index].item()])

print("prediction:", classes[res.argmax().item()])
                                                
