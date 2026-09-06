import torch
import torch.nn.functional as F
import numpy as np
from utils.utils import format_time, compute_metrics
from tqdm import tqdm
import time

class Trainer:
    def __init__(self, model, labeled_trainloader, unlabeled_trainloader,
        val_loader, test_loader, epochs, optimizer, 
        early_stopping=None, scheduler=None,
        temperature=1.0, threshold=0.95,
        mtl_weighting=None,
        use_amp=True, device=None
        
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.labeled_trainloader = labeled_trainloader
        self.unlabeled_trainloader = unlabeled_trainloader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.scheduler = scheduler

        # Hyperparameters for semi-supervised learning
        self.temperature = temperature
        self.threshold = threshold

        self.mtl_weighting = mtl_weighting
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        
    @staticmethod
    def _next_batch(iterator, loader):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch, iterator
    

    def training(self):
        total_start = time.perf_counter()
        assert self.early_stopping is not None, "early_stopping must be provided."
        assert self.val_loader is not None, "val_loader must be provided."

        for epoch in range(self.epochs):
            train_loss, train_metrics = self.train_one_epoch(epoch)
            val_loss, val_metrics = self.evaluate(data_loader=self.val_loader, loader_name="Validation Evaluating...")
            self._print_epoch_summary(epoch, train_loss, train_metrics, val_loss, val_metrics)

            # Early stopping check
            self.early_stopping(val_metrics["f1"], self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered...")
                break

        # Computational time
        total_time = time.perf_counter() - total_start
        print(f"Total training time: {format_time(total_time)}")

        # Load best model
        self.early_stopping.load_best_model(self.model, self.device)
        print("Loaded best model.")
        

        # Best model evaluation(train, val, test)
        self._evaluate_and_log(self.labeled_trainloader, "Best Model Train")
        self._evaluate_and_log(self.val_loader, "Best Model Validation")
        self._evaluate_and_log(self.test_loader, "Test")


    def train_one_epoch(self, epoch=None):
        self.model.train()
        labeled_iter = iter(self.labeled_trainloader)
        unlabeled_iter = iter(self.unlabeled_trainloader)
        num_steps = min(len(self.labeled_trainloader), len(self.unlabeled_trainloader))

        total_loss, total_alpha, total_loss_x, total_beta, total_loss_u, total_mask_ratio = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        all_labels, all_preds = [], []
        
        desc = "Training" if epoch is None else f"Training {epoch + 1}/{self.epochs}"
        progress_bar = tqdm(range(num_steps), desc=desc, leave=False)

        for step, _ in enumerate(progress_bar, start=1):
            (labeled_x, label), labeled_iter = self._next_batch(labeled_iter, self.labeled_trainloader)
            unlabeled_batch, unlabeled_iter = self._next_batch(unlabeled_iter, self.unlabeled_trainloader)
            (inputs_u_w, inputs_u_s), _ = unlabeled_batch

            labeled_x = labeled_x.to(self.device)
            label = label.to(self.device).long()
            inputs_u_w = inputs_u_w.to(self.device)
            inputs_u_s = inputs_u_s.to(self.device)

            self.optimizer.zero_grad()   
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits_labeled_x, logits_u_s = self.model(labeled_x), self.model(inputs_u_s)
                loss_x = F.cross_entropy(logits_labeled_x, label)
                #loss_x = self.focal_loss(logits_labeled_x, label).mean()
                
                with torch.no_grad():
                    logits_u_w = self.model(inputs_u_w) # [B*mu, num_classes]
                    prob_u_w = torch.softmax(logits_u_w.detach() / self.temperature, dim=-1) # [B*mu, num_classes]
                    max_probs, pseudo_labels  = torch.max(prob_u_w, dim=-1) # max_probs: [B*mu, 1], pesudo_labels: [B*mu, 1]
                    mask = max_probs.ge(self.threshold).float() # [B*mu, 1]: True / False

                loss_u = (F.cross_entropy(logits_u_s, pseudo_labels, reduction="none") * mask).mean()
                
                #loss_u = (F.cross_entropy(logits_u_s, pesudo_labels, reduction="none") * mask).mean()
                loss, alpha, beta = self.mtl_weighting(loss_s=loss_x, loss_u=loss_u)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
             
            total_loss_x += loss_x.item()
            total_loss_u += loss_u.item()
            total_alpha += alpha.item()
            total_beta += beta.item()
            total_loss += loss.item()
            total_mask_ratio += mask.mean().item()

            preds = torch.argmax(logits_labeled_x, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())

            progress_bar.set_postfix(total_loss=f"{total_loss / step:.4f}", loss_x=f"{total_loss_x / step:.2f}", alpha=f"{total_alpha / step:.2f}", 
                                     loss_u=f"{total_loss_u / step:.4f}", beta=f"{total_beta / step:.4f}", mask=f"{total_mask_ratio / step:.2f}")
        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / num_steps
        metrics = compute_metrics(all_labels, all_preds)
        metrics["alpha"] = total_alpha / num_steps
        metrics["supervised_loss"] = total_loss_x / num_steps
        metrics["beta"] = total_beta / num_steps
        metrics["unsupervised_loss"] = total_loss_u / num_steps
        metrics["mask_ratio"] = total_mask_ratio / num_steps
        return avg_loss, metrics


    def evaluate(self, data_loader, loader_name="Evaluating..."):
        self.model.eval()
        total_loss = 0.0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for x, labels in tqdm(data_loader, desc=loader_name, leave=False):
                x = x.to(self.device)
                labels = labels.to(self.device).long()

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(x)
                    loss = F.cross_entropy(outputs, labels)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                total_loss += loss.item()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        metrics = compute_metrics(all_labels, all_preds)
        return avg_loss, metrics


    def _evaluate_and_log(self, data_loader, title):
        loss, metrics = self.evaluate(data_loader=data_loader, loader_name=f"{title} Evaluating...")
        print(f"[{title}] Loss: {loss:.4f}, " f"Acc: {metrics['accuracy']:.4f}, " f"Prec: {metrics['precision']:.4f}, " f"Rec: {metrics['recall']:.4f}, " f"F1: {metrics['f1']:.4f}")
        print(f"[{title}] Classification Report:\n{metrics['classification_report']}")
        print(f"[{title}] Confusion Matrix:\n{metrics['confusion_matrix']}\n")


    def _print_epoch_summary(self, epoch, train_loss, train_metrics, val_loss, val_metrics):
        print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}, "
              f"Alpha: {train_metrics['alpha']:.2f}, "
              f"Loss_s: {train_metrics['supervised_loss']:.4f}, "
              f"Beta: {train_metrics['beta']:.2f}, "
              f"Loss_u: {train_metrics['unsupervised_loss']:.4f}, "
              f"Mask: {train_metrics['mask_ratio']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"Prec: {train_metrics['precision']:.4f}, "
              f"Rec: {train_metrics['recall']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        
        print(f"[Epoch {epoch + 1}] Val Loss: {val_loss:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"Prec: {val_metrics['precision']:.4f}, "
              f"Rec: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
