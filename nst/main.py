import torch
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image, ImageDraw

from nst.gram_matrix import gram_matrix
from nst.VGG import VGGFeatures
from nst.live_viewer import  tensor_to_pil

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

content = load_image("content-1.jpg")
style = load_image("style.jpg")

# generated = content.clone().requires_grad_(True)
generated = torch.randn_like(content).to(DEVICE).requires_grad_(True)


model = VGGFeatures().to(DEVICE)

content_features = model(content)
style_features = model(style)
style_grams = {l: gram_matrix(style_features[l]) for l in style_layers}

optimizer_adam = optim.Adam([generated], lr=0.003)
optimizer = optim.LBFGS([generated], max_iter=20, lr=1.0)


frames = []
CAPTURE_EVERY = 5

num_steps = 300
step = 0

while step < num_steps:

    def closure():
        optimizer.zero_grad()

        gen_features = model(generated)

        content_loss = torch.mean(
            (gen_features["conv4_2"] - content_features["conv4_2"]) ** 2
        )

        style_loss = 0
        for l in style_layers:
            G = gram_matrix(gen_features[l])
            A = style_grams[l]
            style_loss += torch.mean((G - A) ** 2)

        loss = alpha * content_loss + beta * style_loss
        loss.backward()

        return loss

    loss = optimizer.step(closure)

    # with torch.no_grad():
    #     generated.clamp_(0, 1)

    if step % CAPTURE_EVERY == 0:
        img = tensor_to_pil(generated)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"iter {step}", fill=(255, 255, 255))
        frames.append(img)
        print(f"Captured frame at step {step}")

    if step % 50 == 0:
        # loss_val = closure().item()
        loss_val = loss
        print(f"Step: {step} | Loss: {loss_val:.2f}")

    step += 1



frames[0].save(
    "style_transfer-2.gif",
    save_all=True,
    append_images=frames[1:],
    duration=2000,   
    loop=0
)
