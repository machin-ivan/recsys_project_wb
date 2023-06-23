import pytorch_lightning as pl

from dataset import PurchaseHistory, PredictionDataset

from torch.utils.data import DataLoader
from typing import Optional

PATH = 'data/filtered_data.csv'


class DataModule(pl.LightningDataModule):
    """
    DataModule
    - Создание train/valid/test dataloader
    """
    def __init__(self, max_len=70, data_dir=PATH, neg_sample_size=100,
                pin_memory=True, num_workers=4, batch_size=256):
        """
        Initialize DataModule
        - Dataset args
        """
        super(DataModule, self).__init__()
        # Dataset related settings
        self.max_len = max_len
        self.neg_sample_size = neg_sample_size
        self.data_dir = data_dir
        # DataLoader related settings
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.batch_size = batch_size
        # Assign vocab size
        self.train_data = PurchaseHistory(
            mode="Train", max_len=self.max_len, data_dir=self.data_dir)
        self.item_size = self.train_data.item_size

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create train/valid/test datasets
        """
        if stage == "fit" or stage is None:
            self.valid_data = PurchaseHistory(
                mode="Valid", max_len=self.max_len, data_dir=self.data_dir,
                neg_sample_size=self.neg_sample_size)
        
        if stage == "test" or stage is None:
            self.test_data = PurchaseHistory(
                mode="Test", max_len=self.max_len, data_dir=self.data_dir,
                neg_sample_size=self.neg_sample_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)
    
    def predict_dataloader(self) -> DataLoader:
        predict_data = PredictionDataset()
        return DataLoader(predict_data, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)
