import torch
from datetime import datetime

from steps import (
    load_data_step,
    preprocess_step,
    train_model_step,
    save_model_step,
    evaluate_step
)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = '/home/marinda/Документы/ml_projects/ProductsMatching/data/processed/ebay_expanded.csv'
LR = 2e-5
BATCH_SIZE = 4
EPOCHS = 3
TEST_SIZE = 0.2
VERBOSE = True


def get_time_str() -> str:
    now = datetime.now()
    formatted = now.strftime("%H:%M")
    return formatted


def training_pipeline(data_path: str = DATA_PATH,
                      lr: float = LR,
                      batch_size: int = BATCH_SIZE,
                      test_size: float = TEST_SIZE,
                      epochs: int = EPOCHS,
                      verbose: bool = VERBOSE,
                      device: str = DEVICE) -> None:
    """
    Training pipeline

    Args:
        data_path: path to data
        lr: learning rate
        batch_size: size of batch
        test_size: size of test data
        device: device
    """
    print()
    print('Config:')
    print(f' - data: {data_path}')
    print(f' - learining rate: {lr}')
    print(f' - batch size: {batch_size}')
    print(f' - test size: {test_size}')
    print(f' - device: {device}')
    print()


    print(f'[{get_time_str()}] Getting data...')
    df = load_data_step(data_path=data_path)

    print(f'[{get_time_str()}] Processing data...')
    train_dataset, val_dataset, tokenizer = preprocess_step(df=df, 
                                                            test_size=test_size)
    print(f'[{get_time_str()}] Training model...')
    model = train_model_step(train_dataset=train_dataset, 
                             val_dataset=val_dataset, 
                             tokenizer=tokenizer, 
                             batch_size=batch_size, 
                             verbose=verbose,
                             epochs=epochs,
                             lr=lr, 
                             device=device)

    print(f'[{get_time_str()}] Evaluate model...')
    metrics = evaluate_step(model=model, 
                            val_dataset=val_dataset, 
                            tokenizer=tokenizer, 
                            device=device)

    print(f'[{get_time_str()}] Metrics:')
    for key in metrics.keys():
        print(f'{key}: {metrics[key]:.3f}')

    if metrics['precision'] > 0.85:
        print(f'[{get_time_str()}] Saving model')
        save_model_step(model)