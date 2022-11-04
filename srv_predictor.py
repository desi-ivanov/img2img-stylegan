import json
from typing import Optional
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import torch
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import numpy as np
import dnnlib
import legacy
from training.dataset import ZipDataset, WithSegSplit

class Wrapper(torch.nn.Module):
    def __init__(self, E, G):
        super().__init__()
        self.E = E
        self.G = G
    def forward(self, z, x, c, **G_kwargs):
        skips = self.E(x)
        return self.G(z, c, imgs=skips, **G_kwargs)

def load_ae(model_path):
    with dnnlib.util.open_url(model_path) as f:
        data = legacy.load_network_pkl(f) 
    if 'AE' not in data and 'VAE' not in data:
        model = Wrapper(data['E'], data['G'])
    else:
        model = (data['AE'] if 'AE' in data else data['VAE'])
    return model.requires_grad_(False).cuda().eval()

def load_clf(clf_path):
    clf = resnet18()
    clf.fc = torch.nn.Linear(clf.fc.in_features, 2, bias=True)
    stdict = {**torch.load(clf_path)['model']}
    stdict = {k.replace('module.', ''): v for k, v in stdict.items()}
    clf.load_state_dict(stdict)
    clf = clf.cuda().eval()
    return clf
    
def torch_to_b64(tensor):
    image = Image.fromarray(tensor.permute(1,2,0).cpu().detach().numpy().astype(np.uint8))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    result_str = base64.b64encode(buffered.getvalue())
    result_str = 'data:image/PNG;base64,' + result_str.decode('utf-8')
    return result_str

class Predictor:
    def __init__(self, seg_dataset_path: str, real_dataset_path: str, model_path: str, seg_colors: str, clf_path: Optional[str] = None):
        
        print("Loading datasets")
        self.training_set_seg = WithSegSplit(seg_dataset_path, json.loads(seg_colors))
        self.training_set_imgs = ImageFolder(real_dataset_path, transform=transforms.ToTensor())
        self.full_datset = ZipDataset(self.training_set_seg, self.training_set_imgs)

        print("Loading models")
        self.model = load_ae(model_path)
        self.clf = load_clf(clf_path) if clf_path else None
        self.T_clf = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        print("Running model to preload CUDA plugins")
        self.example(np.random.randn(512).tolist())
        print("Done")

    def preprocess_in_img(self, img):
        img = transforms.ToTensor()(img)
        img = img[:3, :, :] # remove alpha channel
        colors = torch.tensor(self.training_set_seg.colors).float()
        img = img.mul(255).permute(1, 2, 0)
        dists = img.reshape(-1, 3).unsqueeze(1).sub(colors).pow(2).sum(-1)
        img = colors[dists.argmin(1)]
        img = img.reshape(512, 512, 3)
        img = img.permute(2, 0, 1)
        seg = torch.from_numpy(self.training_set_seg.split(img))
        return seg

    def reconstruct(self, seg, z):
        image = self.model(z.unsqueeze(0).cuda(), seg.unsqueeze(0).cuda(), None, noise_mode="const")[0]
        return image.add(1).div(2)
    
    def classify(self, image):
        return self.clf(self.T_clf(image).unsqueeze(0).cuda()).softmax(1).cpu().detach()[0]

    def random_example(self):
        (seg_img, c), (real_img, c) = self.full_datset[np.random.randint(0, len(self.full_datset))]
        return torch.tensor(seg_img), real_img, c

    def infer(self, seg_base64, raw_z):
        in_img = Image.open(BytesIO(base64.b64decode(seg_base64 + "=" * (4 - len(seg_base64) % 4))))
        mask = self.preprocess_in_img(in_img)
        z = torch.tensor(raw_z)
        image = self.reconstruct(mask, z)

        proba = self.classify(image) if self.clf else None
        return {
            'img': torch_to_b64(image.mul(255).clip(0, 255)),
            'rec_pred_proba': proba.max().item() if proba is not None else None,
            'rec_pred_label': proba.argmax().item() if proba is not None else None,
        }

    def example(self, raw_z):
        z = torch.tensor(raw_z)
        seg_mask, real_img, c = self.random_example()
        rec_image = self.reconstruct(seg_mask, z)
        if self.clf:
            rec_pred = self.classify(rec_image)
            real_pred = self.classify(real_img)
        return {
            'segmented': torch_to_b64(torch.from_numpy(self.training_set_seg.to_rgb(seg_mask.unsqueeze(0).cpu()))[0].clip(0, 255)), 
            'reconstructed': torch_to_b64(rec_image.mul(255).clip(0, 255)),
            'real': torch_to_b64(real_img.mul(255).clip(0,255)),
            'rec_pred_proba': rec_pred.max().item() if self.clf else None,
            'rec_pred_label': rec_pred.argmax().item() if self.clf else None,
            'real_pred_proba': real_pred.max().item() if self.clf else None,
            'real_pred_label': real_pred.argmax().item() if self.clf else None,
            'label': c,
            'z': z.cpu().detach().numpy().tolist(),
        }
