import argparse


def get_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--do_train", action="store_true", default=False)
    arg_parser.add_argument("--do_eval", action="store_true", default=False)

    # Input
    arg_parser.add_argument(
        "--types_path", type=str, help="Path to type specifications",
        required=True
    )
    arg_parser.add_argument("--train_path", type=str, help="Path to train dataset")
    arg_parser.add_argument("--valid_path", type=str, help="Path to validation dataset")
    arg_parser.add_argument("--test_path", type=str, help="Path to test dataset")

    # Preprocessing
    arg_parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer")
    arg_parser.add_argument(
        "--lowercase",
        action="store_true",
        default=False,
        help="If true, input is lowercased during preprocessing",
    )
    arg_parser.add_argument(
        "--sampling_processes",
        type=int,
        default=4,
        help="Number of sampling processes. 0 = no multiprocessing for sampling",
    )

    arg_parser.add_argument(
        "--save_path",
        type=str,
        help="Path to directory where model checkpoints are stored",
        required=True
    )
    arg_parser.add_argument(
        "--overwrite_save_path",
        action="store_true",
        default=False,
    )

    # Model / Training
    arg_parser.add_argument(
        "--se_train_batch_size", type=int, default=1, help="Training batch size for seeding and expanding"
    )
    arg_parser.add_argument("--entail_train_batch_size", type=int, default=16, help="Training batch size for entailing")

    arg_parser.add_argument("--epochs", type=int, default=35, help="Number of epochs")
    arg_parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    arg_parser.add_argument(
        "--lr_warmup",
        type=float,
        default=0.1,
        help="Proportion of total train iterations to warmup in linear increase/decrease schedule",
    )
    arg_parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply"
    )
    arg_parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm"
    )
    arg_parser.add_argument("--focal_loss_gamma", type=int, default=2)
    arg_parser.add_argument("--seeding_loss_weight", type=float, default=1.0)
    arg_parser.add_argument("--expansion_loss_weight", type=float, default=1.0)
    arg_parser.add_argument("--entailing_loss_weight", type=float, default=1.0)

    # Model / Training / Evaluation
    arg_parser.add_argument(
        "--plm_path",
        type=str,
        help="Path to directory that contains model checkpoints",
    )
    arg_parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="If true, train/evaluate on CPU even if a CUDA device is available",
    )
    arg_parser.add_argument(
        "--se_eval_batch_size", type=int, default=1, help="Evaluation batch size for seeding and expanding"
    )
    arg_parser.add_argument("--entail_eval_batch_size", type=int, default=16, help="Evaluation batch size for entailing")
    arg_parser.add_argument(
        "--prop_drop",
        type=float,
        default=0.5,
        help="Probability of dropout",
    )
    
    arg_parser.add_argument("--seed_threshold", type=float, default=0.7)
    arg_parser.add_argument("--offset_limit", type=int, default=5)

    # Misc
    arg_parser.add_argument("--seed", type=int, default=-1, help="Seed")

    return arg_parser
