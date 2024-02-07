import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import T5ForConditionalGeneration
from tqdm.auto import tqdm
import os


def train_loop(
    model: T5ForConditionalGeneration,
    train_dl: DataLoader,
    optimizer: Adam,
    device: str,
) -> float:
    avg_loss = 0
    for X, y in tqdm(train_dl):
        X, y = X.to(device), y.to(device)
        loss = model(input_ids=X, labels=y).loss
        avg_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_dl)
    return avg_loss


def val_loop(
    model: T5ForConditionalGeneration, val_dl: DataLoader, device: str
) -> float:
    avg_loss = 0
    with torch.no_grad():
        for X, y in tqdm(val_dl):
            X, y = X.to(device), y.to(device)
            loss = model(input_ids=X, labels=y).loss
            avg_loss += loss.item()
    avg_loss /= len(val_dl)
    return avg_loss


def save(
    model,
    optimizer,
    best_epoch,
    best_loss,
    best_model_weights,
    best_optimizer_weights,
    epoch,
    final_loss,
    save_dir,
):
    save_obj = {
        "best": {
            "epoch": best_epoch,
            "loss": best_loss,
            "model": best_model_weights,
            "optimizer": best_optimizer_weights,
        },
        "final": {
            "epoch": epoch,
            "loss": final_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
    }
    torch.save(save_obj, f"{save_dir}/states.tar")


def fit(
    train_dl: DataLoader,
    val_dl: DataLoader,
    model: T5ForConditionalGeneration,
    epochs: int = 100,
    patience: int = 10,
    save_dir: str = ".",
):
    assert os.path.exists(save_dir), f"Save directory ({save_dir}) not found"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = model.to(device)
    for param in model.encoder.parameters():
        param.requires_grad = False
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()))

    # fine tume the model
    best_epoch = -1
    best_loss = float("inf")
    best_model_weights = model.state_dict()
    best_optimizer_weights = optimizer.state_dict()
    for epoch in range(epochs):
        print(f"----- Epoch {str(epoch+1).rjust(len(str(epochs)), '0')}/{epochs} -----")
        train_loss = train_loop(model, train_dl, optimizer, device)
        val_loss = val_loop(model, val_dl, device)

        # update the best states
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = model.state_dict()
            best_optimizer_weights = optimizer.state_dict()
            best_epoch = epoch

        # save checkpoint
        save(
            model,
            optimizer,
            best_epoch,
            best_loss,
            best_model_weights,
            best_optimizer_weights,
            epoch,
            val_loss,
            save_dir,
        )

        # check for overfitting
        if best_epoch + patience < epoch:
            print(
                f"Stoping early at epoch {epoch+1}, best loss ({best_loss}) observed at epcoh {best_epoch+1}"
            )
            break
