import torchvision
from torchvision import transforms


def getMNIST(size, dl_path, is_gray=True, need_dl=False):
    if is_gray: dim = 1
    else: dim = 3
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Grayscale(dim),
            transforms.ToTensor(),
        ]
    )
    
    trainset = torchvision.datasets.MNIST(root=dl_path, train=True, download=need_dl, transform=transform)
    testset = torchvision.datasets.MNIST(root=dl_path, train=False, download=need_dl, transform=transform)
    
    return trainset, testset