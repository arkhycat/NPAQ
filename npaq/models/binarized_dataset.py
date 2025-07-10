from torchvision import datasets, transforms

class BinarizedDataset[T: datasets.VisionDataset]:
    def __init__(self, original_dataset: T, threshold=0.5):
        self.original_dataset = original_dataset
        self.threshold = threshold

    def __getitem__(self, index):
        x, y = self.original_dataset[index]
    
        x = x > self.threshold
        x = x.float() # Cast back to float, since x is a ByteTensor now
    
        # Apply mask here
        if hasattr(self.original_dataset, "masks") and self.original_dataset.masks is not None:
            x = x * self.original_dataset.masks[index] 
    
        return x, y
    
    def __len__(self):
        return self.original_dataset.__len__()
    

if __name__ == "__main__":
    test_dataset_instance  = BinarizedDataset(datasets.MNIST('../data', download=True, train=False, transform=transforms.Compose([transforms.Resize(4, 4),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])))
    print(test_dataset_instance[0])