import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = "data"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=10)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)


def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def plot(val_loss,train_loss,typ):
    plt.title("{} after epoch: {}".format(typ,len(train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel(typ)
    plt.plot(list(range(len(train_loss))),train_loss,color="r",label="Train "+typ)
    plt.plot(list(range(len(val_loss))),val_loss,color="b",label="Validation "+typ)
    plt.legend()
    plt.savefig(os.path.join(data_dir,typ+".png"))
    plt.close()

val_loss_gph=[]
train_loss_gph=[]
val_acc_gph=[]
train_acc_gph=[]

def train_model(model, criterion, optimizer, scheduler, num_epochs=25,model_name = "kaggle"):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) #was (outputs,1) for non-inception and (outputs.data,1) for inception
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
              train_loss_gph.append(epoch_loss)
              train_acc_gph.append(epoch_acc)
            if phase == 'val':
              val_loss_gph.append(epoch_loss)
              val_acc_gph.append(epoch_acc)
            
            plot(val_loss_gph,train_loss_gph, "Loss")
            plot(val_acc_gph,train_acc_gph, "Accuracy")

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, data_dir+"/"+model_name+".h5")
                print('==>Model Saved')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model = models.resnet18(pretrained = True)

num_ftrs = model.fc.in_features  ##for googlenet, resnet18
#num_ftrs = model.classifier.in_features  ## for densenet169
print("Number of features: "+str(num_ftrs))


model.fc = nn.Linear(num_ftrs, num_classes)  ##for googlenet, resnet18
#model.classifier = nn.Linear(num_ftrs, num_classes) ## for densenet169
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# Decay LR by a factor of 0.1 every 7 epochs
# Learning rate scheduling should be applied after optimizer’s update
# e.g., you should write your code this way:
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=30, model_name = "resnet18")


# Getting Proba distribution
print("\nGetting the Probability Distribution")
trainloader=torch.utils.data.DataLoader(image_datasets['train'],batch_size=1)
testloader=torch.utils.data.DataLoader(image_datasets['val'],batch_size=1)


model=model.eval()
correct = 0
total = 0
import csv
import numpy as np


#Train Probabilities
f = open(data_dir+"/resnet18_train.csv",'w+',newline = '')
writer = csv.writer(f)

saving = []
with torch.no_grad():
      num = 0
      temp_array = np.zeros((len(trainloader),num_classes))
      for i,data in enumerate(trainloader):
          images, labels = data
          sample_fname, _ = trainloader.dataset.samples[i]
          labels=labels.cuda()
          outputs = model(images.cuda())
          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels.cuda()).sum().item()
          prob = torch.nn.functional.softmax(outputs, dim=1)
          saving.append(sample_fname.split('/')[-1])
          temp_array[num] = np.asarray(prob[0].tolist()[0:num_classes])
          num+=1
print("Train Accuracy = ",100*correct/total)

for i in range(len(trainloader)):
  k = temp_array[i].tolist()
  k.append(saving[i])
  writer.writerow(k)

f.close()

f = open(data_dir+"/train_labels.csv",'w+',newline = '')
writer = csv.writer(f)
for i,data in enumerate(testloader):
  _, labels = data
  sample_fname, _ = testloader.dataset.samples[i]
  sample = sample_fname.split('/')[-1]
  lab = labels.tolist()[0]
  writer.writerow([sample,lab])
f.close()


#Test Probabilities
f = open(data_dir+"/resnet18_test.csv",'w+',newline = '')
writer = csv.writer(f)

saving = []
with torch.no_grad():
      num = 0
      temp_array = np.zeros((len(testloader),num_classes))
      for i,data in enumerate(testloader):
          images, labels = data
          sample_fname, _ = testloader.dataset.samples[i]
          labels=labels.cuda()
          outputs = model(images.cuda())
          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels.cuda()).sum().item()
          prob = torch.nn.functional.softmax(outputs, dim=1)
          saving.append(sample_fname.split('/')[-1])
          temp_array[num] = np.asarray(prob[0].tolist()[0:num_classes])
          num+=1
print("Test Accuracy = ",100*correct/total)

for i in range(len(testloader)):
  k = temp_array[i].tolist()
  k.append(saving[i])
  writer.writerow(k)

f.close()

f = open(data_dir+"/test_labels.csv",'w+',newline = '')
writer = csv.writer(f)
for i,data in enumerate(testloader):
  _, labels = data
  sample_fname, _ = testloader.dataset.samples[i]
  sample = sample_fname.split('/')[-1]
  lab = labels.tolist()[0]
  writer.writerow([sample,lab])
f.close()
