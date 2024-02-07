from train.t5 import fit
from datasets import CFGenerativeDataset
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
import os

if __name__ == "__main__":
    # Directory variables
    data_dir = os.environ["SM_CHANNEL_TRAIN"]
    intermediate_data_dir = os.environ["SM_OUTPUT_INTERMEDIATE_DIR"]
    model_output_dir = os.environ["SM_MODEL_DIR"]
    output_data_dir = os.environ["SM_OUTPUT_DATA_DIR"]

    BATCH_SIZE = 4
    EPOCHS = 100
    PATIENCE = 20
    SAVE_DIR = model_output_dir
    MODEL_NAME = "t5-small"

    train_ds = CFGenerativeDataset(f"{data_dir}/snli_1.0_contra.yaml", split="train")
    val_ds = CFGenerativeDataset(f"{data_dir}/snli_1.0_contra.yaml", split="val")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    fit(
        train_dl,
        val_dl,
        model,
        epochs=EPOCHS,
        patience=PATIENCE,
        save_dir=SAVE_DIR,
    )
