import os
from tinygrad import Tensor, nn, TinyJit
from tinygrad.nn.state import torch_load, load_state_dict, get_parameters, get_state_dict
from PIL import Image
import numpy as np

os.environ["DEV"]="CL"
os.environ["DEBUG"]="4"

def download_model():
    url = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
    pass

class VGGFeatures:
    def __init__(self):
        self.features = {
            "0":  nn.Conv2d(3,   64,  3, padding=1),
            "2":  nn.Conv2d(64,  64,  3, padding=1),
            "5":  nn.Conv2d(64,  128, 3, padding=1),
            "7":  nn.Conv2d(128, 128, 3, padding=1),
            "10": nn.Conv2d(128, 256, 3, padding=1),
            "12": nn.Conv2d(256, 256, 3, padding=1),
            "14": nn.Conv2d(256, 256, 3, padding=1),
            "16": nn.Conv2d(256, 256, 3, padding=1),
            "19": nn.Conv2d(256, 512, 3, padding=1),
            "21": nn.Conv2d(512, 512, 3, padding=1),
            "23": nn.Conv2d(512, 512, 3, padding=1),
            "25": nn.Conv2d(512, 512, 3, padding=1),
            "28": nn.Conv2d(512, 512, 3, padding=1),
        }

    def __call__(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        f = self.features
        s1 = f["0"](x).relu()
        x  = f["2"](s1).relu()
        x  = x.avg_pool2d(2)
        s2 = f["5"](x).relu()
        x  = f["7"](s2).relu()
        x  = x.avg_pool2d(2)
        s3 = f["10"](x).relu()
        x  = f["12"](s3).relu()
        x  = f["14"](x).relu()
        x  = f["16"](x).relu()
        x  = x.avg_pool2d(2)
        s4 = f["19"](x).relu()
        c  = f["21"](s4).relu()
        x  = f["23"](c).relu()
        x  = f["25"](x).relu()
        x  = x.avg_pool2d(2)
        s5 = f["28"](x).relu()

        return c, [s1, s2, s3, s4, s5]


def load_image(path: str, size: int = 512) -> Tensor:
    img = Image.open(path).convert("RGB").resize((size, size))
    x   = Tensor(np.array(img).astype(np.float32) / 255.0)
    x   = x.permute(2, 0, 1).unsqueeze(0)
    return x

def save_image(t: Tensor, path: str):
    x = t.squeeze(0).permute(1, 2, 0)
    x = (x.numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(x).save(path)


def gram(f: Tensor) -> Tensor:
    _, C, H, W = f.shape
    F = f.reshape(C, H * W)
    return F.matmul(F.T) / (C * H * W)


def content_loss(gen_c: Tensor, tgt_c: Tensor) -> Tensor:
    return ((gen_c - tgt_c) ** 2).mean()

def style_loss(gen_s: list[Tensor], tgt_grams: list[Tensor]) -> Tensor:
    return sum(((gram(g) - t) ** 2).mean() for g, t in zip(gen_s, tgt_grams))


if __name__ == "__main__":
    CONTENT_PATH = "./img1.jpg"
    STYLE_PATH   = "./img2.jpg"
    OUT_PATH     = "./output.jpg"
    SIZE         = 512
    STEPS        = 500
    ALPHA        = 1.0
    BETA         = 1e5
    LR           = 0.02

    model = VGGFeatures()
    load_state_dict(model, torch_load("./vgg19-dcbb9e9d.pth"), strict=False)

    for p in get_parameters(model):
        p.requires_grad = False

    content_img = load_image(CONTENT_PATH, SIZE)
    style_img   = load_image(STYLE_PATH,   SIZE)

    Tensor.no_grad = True
    content_target, _ = model(content_img)
    _, style_feats = model(style_img)
    style_targets = [gram(s) for s in style_feats]
    Tensor.no_grad = False

    gen = Tensor(content_img.numpy(), requires_grad=True)

    optimizer = nn.optim.Adam([gen], lr=LR)
    @TinyJit
    def step():
        Tensor.training = True
        optimizer.zero_grad()
        gen_c, gen_s = model(gen)
        c_loss = content_loss(gen_c, content_target)
        s_loss = style_loss(gen_s, style_targets)
        loss   = ALPHA * c_loss + BETA * s_loss
        loss.backward()
        optimizer.step()
        gen.realize()
        return loss, c_loss, s_loss

    for _step in range(1, STEPS + 1):
        loss, c_loss, s_loss = step()
        if _step % 50 == 0:
            print(f"[{_step:4d}/{STEPS}] loss={loss.numpy():.4f}  content={c_loss.numpy():.4f}  style={s_loss.numpy():.4f}")

    save_image(gen, OUT_PATH)
    print(f"saved: {OUT_PATH}")