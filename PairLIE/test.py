from data import get_eval_set

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    train_set = get_eval_set(r'C:/Users/HP-VICTUS/Documents/Masters/PairLIE/demo_results/epoch_500/I')
    training_data_loader = DataLoader(dataset=train_set, batch_size=1,
                                      shuffle=True)
    examples = next(iter(training_data_loader))

    for label, img in enumerate(examples):
        plt.imshow(img.squeeze(0).permute(1, 2, 0))
        plt.show()
        print(f"Label: {label}")

    print(train_set)