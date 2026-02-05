from __future__ import annotations

import argparse
from pathlib import Path

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from roc_time_parser.normalizer.dataset import load_normalizer_jsonl, prepare_seq2seq_features
from roc_time_parser.normalizer.model import init_normalizer_from_base


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="roc_time_parser.normalizer.train")
    ap.add_argument("--train", required=True, help="Train jsonl (span/refdate/target format).")
    ap.add_argument("--dev", required=True, help="Dev jsonl (span/refdate/target format).")
    ap.add_argument("--base-model", default="google/mt5-small")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-source-length", type=int, default=128)
    ap.add_argument("--max-target-length", type=int, default=128)
    ap.add_argument("--resume-from-checkpoint", default=None)
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = init_normalizer_from_base(args.base_model)

    ds_train = load_normalizer_jsonl(args.train)
    ds_dev = load_normalizer_jsonl(args.dev)

    ds_train = ds_train.map(
        prepare_seq2seq_features,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_length": int(args.max_source_length),
            "max_target_length": int(args.max_target_length),
        },
    )
    ds_dev = ds_dev.map(
        prepare_seq2seq_features,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_length": int(args.max_source_length),
            "max_target_length": int(args.max_target_length),
        },
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        seed=args.seed,
        report_to=[],
        max_steps=args.max_steps,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

