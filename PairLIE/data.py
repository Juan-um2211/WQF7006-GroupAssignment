from torchvision.transforms import Compose, ToTensor, RandomCrop
from dataset import DatasetFromFolderEval, DatasetFromFolder

def transform1():
    return Compose([
        RandomCrop((128, 128)),
        ToTensor(),
    ])

def transform2():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir):
    return DatasetFromFolder(data_dir, transform=transform1())


def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    train_set = get_training_set(r'C:/Users/HP-VICTUS/Documents/Masters/PairLIE/PairLIE-training-dataset/')
    training_data_loader = DataLoader(dataset=train_set, batch_size=1,
                                      shuffle=True)
    examples = next(iter(training_data_loader))
    print(len(training_data_loader))

    for label, img in enumerate(examples):
        plt.imshow(img.squeeze(0).permute(1, 2, 0))
        plt.show()
        print(f"Label: {label}")

    print(train_set)