import torch
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import optim
from dataset.dataset import get_wm811k
from model.model import ResnetModel
from utils.utils import set_seed, EarlyStopping, FocalLoss, MultiTaskLossWeighting
from utils.trainer_MTL import Trainer



logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)


if __name__ == "__main__":
    set_seed(42)
    label_ratio = 1.00
    batch_size = 256
    mu = 4
    epochs = 150
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset = get_wm811k(
        labeled_path="./data/wm811k/preprocessing/labeled.pkl",
        unlabeled_path="./data/wm811k/preprocessing/unlabeled.pkl",
        train_ratio=0.75,
        val_ratio=0.15,
        test_ratio=0.10,
        image_size= 96,
        data_seed=1,
        label_ratio=label_ratio,

        cutout_num_holes=4,
        cutout_ratio=0.2,
        noise_prob=0.05,
    )
    

    logger.info(f"train_labeled_dataset: {len(train_labeled_dataset)}")
    logger.info(f"val_dataset: {len(val_dataset)}")
    logger.info(f"test_dataset: {len(test_dataset)}")
    logger.info(f"train_unlabeled_dataset: {len(train_unlabeled_dataset)}")
    
    
    labeled_trainloader = DataLoader(
        train_labeled_dataset,
        sampler=RandomSampler(train_labeled_dataset),
        batch_size=batch_size,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )

    unlabeled_trainloader = DataLoader(
        train_unlabeled_dataset,
        sampler=RandomSampler(train_unlabeled_dataset),
        batch_size=batch_size * mu,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size,
        drop_last=False,
        )
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),    
        batch_size=batch_size,
        drop_last=False
        )
    
    model = ResnetModel(model_name="resnet18", num_classes=9, pretrained=True).to(device)
    
    # 1. ResNet backbone freeze
    for param in model.backbone.parameters():
        param.requires_grad = True

    # # 2. ResNet backbone layer3 unfreeze
    # for param in model.backbone.layer3.parameters():
    #     param.requires_grad = True

    # # # 3. ResNet backbone layer4 unfreeze
    # for param in model.backbone.layer4.parameters():
    #     param.requires_grad = True

    # 4. fc classifier layer unfreeze
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    early_stopping = EarlyStopping(patience=30, verbose=True, delta=0.0, path="./checkpoints/best_model.pt")
    mtl_weighting = MultiTaskLossWeighting(num=2).to(device)
    optimizer = optim.Adam([{"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": 1e-3, "weight_decay": 1e-5},
                            {"params": mtl_weighting.parameters(), "lr": 1e-4, "weight_decay": 0.0}
                            ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    


    trainer = Trainer(
        model=model, 
        labeled_trainloader=labeled_trainloader, unlabeled_trainloader=unlabeled_trainloader,
        val_loader=val_loader, test_loader=test_loader,
        epochs=epochs, optimizer=optimizer, scheduler=scheduler, early_stopping=early_stopping, 
        temperature=1.0, threshold=0.90,
        mtl_weighting=mtl_weighting,
        use_amp=True, device=device)

    trainer.training()
