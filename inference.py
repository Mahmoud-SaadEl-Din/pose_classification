import torch, torch.nn as nn
from torchvision import models
from dataloading import *

model_path = "best_weights/best.pt"
output_folder_class0 = "model_results/not_pose"
output_folder_class1 = "model_results/pose"

# Make sure the output folders exist
os.makedirs(output_folder_class0, exist_ok=True)
os.makedirs(output_folder_class1, exist_ok=True)
resnet = models.resnet18(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 2)

resnet.load_state_dict(torch.load(model_path))

resnet.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device=device)
# Iterate over data.
for inputs, labels in dataloaders["val"]:#tqdm(self.val_dataloader,desc="Validation loop"):
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = resnet(inputs)
    _, preds = torch.max(outputs, 1)
    for i in range(inputs.size(0)):
        input_image = transforms.ToPILImage()(inputs[i].cpu())
        label = labels[i].item()
        pred = preds[i].item()

        # Define the path to save the image
        if pred == 0:
            save_path = os.path.join(output_folder_class0, f"image_{i}_class{label}_pred{pred}.png")
        else:
            save_path = os.path.join(output_folder_class1, f"image_{i}_class{label}_pred{pred}.png")

        # Save the image
        input_image.save(save_path)