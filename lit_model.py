import pytorch_lightning as pl
import torch.nn as nn
import torch
import numpy as np

from model import BERT

from torchmetrics import RetrievalHitRate, RetrievalNormalizedDCG, RetrievalMAP, RetrievalRecall


class BERT4REC(pl.LightningModule):
    """
    LightningModule for BERT4REC.
    - Defines train, valid, test, and predict steps.
    - Defines optimizer and scheduler.
    - Defines Module-related arguments.
    """
    def __init__(self, learning_rate=1e-3, max_len=70, hidden_dim=256,
                encoder_num=2, head_num=4, dropout_rate=.1, dropout_rate_attn=.1,
                item_size=5957, initializer_range=.02, weight_decay=.01, decay_step=25,
                gamma=.1, batch_size=256):
        """
        Initializes the LightningModule.
        - Creates the model.
        - Defines the criterion and metrics.
        - Defines train/validate-test steps.
        - Defines prediction (inference).

        :param args: Arguments for the model and training.
        """
        super(BERT4REC, self).__init__()
        # Training-related parameters
        self.learning_rate = learning_rate
        # BERT-related parameters
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.encoder_num = encoder_num
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        self.dropout_rate_attn = dropout_rate_attn
        self.vocab_size = item_size + 2
        self.initializer_range = initializer_range
        # Optimizer-related parameters
        self.weight_decay = weight_decay
        self.decay_step = decay_step
        self.gamma = gamma
        # BERT creation
        self.model = BERT(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            hidden_dim=self.hidden_dim,
            encoder_num=self.encoder_num,
            head_num=self.head_num,
            dropout_rate=self.dropout_rate,
            dropout_rate_attn=self.dropout_rate_attn,
            initializer_range=self.initializer_range
        )
        self.out = nn.Linear(self.hidden_dim, item_size + 1)  # Mask prediction: 1 ~ args.item_size + 1
        self.batch_size = batch_size  # Used for step calculation
        # Criterion, metrics
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.HR = RetrievalHitRate(k=10)  # HR@10
        self.NDCG = RetrievalNormalizedDCG(k=10)  # NDCG@10
        self.MAP = RetrievalMAP(k=10) # MAP@10
        self.RECALL = RetrievalRecall(k=10) # Recall@10

    def forward(self, x):
        """
        Used for debugging.
        """
        pass

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step.
        - Must return the loss for backpropagation.

        :param batch: Input batch tensor.
        :param batch_idx: Batch index.
        :return: Loss tensor.
        """
        seq, labels = batch
        logits = self.model(seq)  # Logits: [batch_size, max_len, hidden_dim]
        preds = self.out(logits)  # Predictions: [batch_size, max_len, vocab_size]

        loss = self.criterion(preds.transpose(1, 2), labels)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Validation step.

        :param batch: Input batch tensor.
        :param batch_idx: Batch index.
        """
        seq, candidates, labels = batch
        logits = self.model(seq)
        preds = self.out(logits)

        preds = preds[:, -1, :]  # Extract only the MASK part, preds: [batch_size, vocab_size]
        targets = candidates[:, 0]  # The 0th index of each batch's Candidates is the Label
        loss = self.criterion(preds, targets)

        recs = torch.gather(preds, 1, candidates)  # recs: [batch_size, neg_sample + 1]
        # Calculate HR, NDCG
        steps = batch_idx * self.batch_size
        indexes = torch.arange(steps, steps + seq.size(0), dtype=torch.long).unsqueeze(1).repeat(1, 101)
        hr = self.HR(recs, labels, indexes)  # dim recs = labels = indexes
        ndcg = self.NDCG(recs, labels, indexes)
        # Log after the completion of the validation_step, on_step=False, on_epoch=True
        self.log("val_loss", loss, on_step=False,
                on_epoch=True, prog_bar=True, logger=True)
        self.log("HR_val", hr, on_step=False,
                on_epoch=True, prog_bar=True, logger=True)
        self.log("NDCG_val", ndcg, on_step=False,
                on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Test step.

        :param batch: Input batch tensor.
        :param batch_idx: Batch index.
        """
        seq, candidates, labels = batch
        logits = self.model(seq)
        preds = self.out(logits)

        preds = preds[:, -1, :]  # Extract only the MASK part, preds: [batch_size, vocab_size]
        targets = candidates[:, 0]  # The 0th index of each batch's Candidates is the Label
        loss = self.criterion(preds, targets)

        recs = torch.gather(preds, 1, candidates)  # recs: [batch_size, neg_sample + 1]
        # Calculate HR, NDCG
        steps = batch_idx * self.batch_size
        indexes = torch.arange(steps, steps + seq.size(0), dtype=torch.long).unsqueeze(1).repeat(1, 101)
        hr = self.HR(recs, labels, indexes)  # dim recs = labels = indexes
        ndcg = self.NDCG(recs, labels, indexes)
        map = self.MAP(recs, labels, indexes)
        recall = self.RECALL(recs, labels, indexes)
        # Log after the completion of the test_step, on_step=False, on_epoch=True
        self.log("test_loss", loss, on_step=False,
                on_epoch=True, prog_bar=True, logger=True)
        self.log("HR_test", hr, on_step=False,
                on_epoch=True, prog_bar=True, logger=True)
        self.log("NDCG_test", ndcg, on_step=False,
                on_epoch=True, prog_bar=True, logger=True)
        self.log("MAP_test", map, on_step=False,
                on_epoch=True, prog_bar=True, logger=True)
        self.log("Recall_test", recall, on_step=False,
                on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx=0) -> np.array:
        """
        Prediction step.

        :param batch: Input batch tensor.
        :param batch_idx: Batch index.
        :param dataloader_idx: Dataloader index (default=0).
        :return: Numpy array of predicted indexes.
        """
        seq, uid = batch
        logits = self.model(seq)
        preds = self.out(logits)

        preds = preds[:, -1, :]  # Extract only the MASK part, preds: [batch_size, vocab_size]
        _, items = torch.topk(preds, 10)

        return uid.cpu().numpy(), items.cpu().numpy()

    def configure_optimizers(self):
        """
        Initialize optimizer and scheduler.

        :return: Dictionary containing optimizer, lr_scheduler, and monitor.
        """
        # No decay for bias and LayerNorm.weight
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
        ]
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.decay_step, gamma=self.gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

