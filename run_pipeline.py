import argparse
import torch
from datetime import datetime

from pipelines.train_pipeline import training_pipeline

DEFAULT_DATA_PATH = '/home/marinda/Документы/ml_projects/ProductsMatching/data/processed/ebay_expanded.csv'
DEFAULT_LR = 2e-5
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 3
DEFAULT_TEST_SIZE = 0.2
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():

    parser = argparse.ArgumentParser(
        description='Training pipeline for BERT sequence classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        default=DEFAULT_DATA_PATH,
        help='Path to the training data CSV file'
    )
    
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float, 
        default=DEFAULT_LR,
        help='Learning rate for optimizer'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int, 
        default=DEFAULT_BATCH_SIZE,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int, 
        default=DEFAULT_EPOCHS,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--test-size',
        type=float, 
        default=DEFAULT_TEST_SIZE,
        help='Fraction of data to use for validation (0.0-1.0)'
    )
    
    parser.add_argument(
        '--device',
        type=str, 
        default=DEFAULT_DEVICE,
        choices=['cuda', 'cpu', 'auto'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Disable verbose output'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    verbose = args.verbose and not args.no_verbose
    
    print('=' * 60)
    print('Starting training pipeline')
    print('=' * 60)
    
    training_pipeline(
        data_path=args.data_path,
        lr=args.lr,
        batch_size=args.batch_size,
        test_size=args.test_size,
        epochs=args.epochs,
        verbose=verbose,
        device=device
    )
    
    print('=' * 60)
    print('Pipeline completed successfully!')
    print('=' * 60)


if __name__ == '__main__':
    main()
