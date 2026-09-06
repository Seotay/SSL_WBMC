import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes} min {seconds:.2f} sec"    


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path="./checkpoint/best_model.pt"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_val_f1 = None
        self.early_stop = False

    def __call__(self, current_val_f1, model):
        if self.best_val_f1 is None:
            self.best_val_f1 = current_val_f1
            self.save_checkpoint(model)

        elif current_val_f1 >= self.best_val_f1 + self.delta:
            self.save_checkpoint(model)
            self.best_val_f1 = current_val_f1
            self.counter = 0

        else:
            self.counter += 1
            if self.verbose:
                print(f"\tEarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        if self.verbose and self.best_val_f1 is not None:
            print(f"\tValidation F1 improved Saving model: {self.path}")
        torch.save(model.state_dict(), self.path)

    def load_best_model(self, model, device=None):
        map_location = device if device is not None else "cpu"
        state_dict = torch.load(self.path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "classification_report": report, "confusion_matrix": conf_matrix}


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = float(alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = self.CE(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)[targets]
                focal_loss = focal_loss * alpha_t
            else:
                focal_loss = focal_loss * self.alpha

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskLossWeighting(nn.Module):
    """AWL-style multi-task loss weighting with trainer_v3 compatibility."""

    def __init__(self, num=2):
        super().__init__()
        params = torch.ones(num, dtype=torch.float32)
        self.params = nn.Parameter(params)
    
    def mtl_term(self, loss, param):
        weight = 0.5 / (param ** 2)
        weighted_loss = weight * loss + torch.log(1 + param ** 2)
        return weighted_loss, weight

    def forward(self, loss_s, loss_u):
        loss_s_weighted, alpha = self.mtl_term(loss_s, self.params[0])
        loss_u_weighted, beta = self.mtl_term(loss_u, self.params[1])
        total_loss = loss_s_weighted + loss_u_weighted
    
        return total_loss, alpha.detach(), beta.detach()
    


