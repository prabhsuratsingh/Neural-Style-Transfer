import torch
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

from nst.gram_matrix import gram_matrix
from nst.VGG import VGGFeatures

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(path, max_size=512):
    image = Image.open(path).convert("RGB")
    size = min(max(image.size), max_size)
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = transform(image).unsqueeze(0)
    return image.to(DEVICE)

content_layers = ["conv4_2"]
style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

alpha = 1
beta = 1e4

content = load_image("content.jpg")
style = load_image("style.jpg")

generated = content.clone().requires_grad_(True)

model = VGGFeatures().to(DEVICE)

content_features = model(content)
style_features = model(style)
style_grams = {l: gram_matrix(style_features[l]) for l in style_layers}

optimizer = optim.Adam([generated], lr=0.003)

for step in range(300):
    gen_features = model(generated)

    content_loss = torch.mean(
        (gen_features["conv4_2"] - content_features["conv4_2"])**2
    )

    style_loss = 0
    for l in style_layers:
        G = gram_matrix(gen_features[l])
        A = style_grams[l]
        style_loss += torch.mean((G-A)**2)
    
    loss = alpha * content_loss + beta * style_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        generated.clamp_(0, 1)


    if step % 50 == 0:
        print(f"Step: {step} | Loss: {loss.item():.2f}")