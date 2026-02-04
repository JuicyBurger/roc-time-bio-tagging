from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import evaluate
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from roc_time_parser.extractor.dataset import load_spans_jsonl, tokenize_and_align_labels
from roc_time_parser.extractor.labels import ID2LABEL
from roc_time_parser.extractor.model import init_extractor_from_base


def _compute_metrics_factory(id2label: dict[int, str]):
    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)

        true_predictions = [
            [id2label[int(p_)] for (p_, l_) in zip(pred, lab) if int(l_) != -100]
            for pred, lab in zip(preds, labels)
        ]
        true_labels = [
            [id2label[int(l_)] for (p_, l_) in zip(pred, lab) if int(l_) != -100]
            for pred, lab in zip(preds, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="roc_time_parser.extractor.train")
    ap.add_argument("--train", required=True, help="Train jsonl (spans_labeled format).")
    ap.add_argument("--dev", required=True, help="Dev jsonl (spans_labeled format).")
    ap.add_argument("--base-model", default="microsoft/Multilingual-MiniLM-L12-H384")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--augment-ocr", action="store_true", help="Inject OCR-like noise into model input.")
    ap.add_argument("--augment-prob", type=float, default=0.15)
    ap.add_argument("--augment-seed", type=int, default=42)
    ap.add_argument("--resume-from-checkpoint", default=None)
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = init_extractor_from_base(args.base_model)

    ds_train = load_spans_jsonl(args.train)
    ds_dev = load_spans_jsonl(args.dev)

    ds_train = ds_train.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "augment_ocr": bool(args.augment_ocr),
            "augment_prob": float(args.augment_prob),
            "augment_seed": int(args.augment_seed),
        },
    )
    ds_dev = ds_dev.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer})

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics_factory(ID2LABEL),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

