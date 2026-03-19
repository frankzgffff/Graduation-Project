from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict


def setup_logger(log_dir: str | Path, name: str = "rle_nas") -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class ExperimentLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.log_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.logs_dir / "metrics.jsonl"
        self.record_paths = {
            "search_steps": {
                "json": self.logs_dir / "search_steps.json",
                "csv": self.logs_dir / "search_steps.csv",
            },
            "evaluated_architectures": {
                "json": self.logs_dir / "evaluated_architectures.json",
                "csv": self.logs_dir / "evaluated_architectures.csv",
            },
            "final_retrain_history": {
                "json": self.logs_dir / "final_retrain_history.json",
                "csv": self.logs_dir / "final_retrain_history.csv",
            },
            "ablation_summary": {
                "json": self.logs_dir / "ablation_summary.json",
                "csv": self.logs_dir / "ablation_summary.csv",
            },
        }
        self._buffers: dict[str, list[Dict[str, Any]]] = {key: [] for key in self.record_paths}

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._normalize_value(metrics), ensure_ascii=False) + "\n")

    def log_search_step(self, record: Dict[str, Any]) -> None:
        self._append_record("search_steps", record)

    def log_architecture_evaluation(self, record: Dict[str, Any]) -> None:
        self._append_record("evaluated_architectures", record)

    def log_final_retrain_epoch(self, record: Dict[str, Any]) -> None:
        self._append_record("final_retrain_history", record)

    def log_ablation_result(self, record: Dict[str, Any]) -> None:
        self._append_record("ablation_summary", record)

    def _append_record(self, name: str, record: Dict[str, Any]) -> None:
        normalized = self._normalize_value(record)
        self._buffers[name].append(normalized)
        self._write_record_set(name)

    def _write_record_set(self, name: str) -> None:
        records = self._buffers[name]
        paths = self.record_paths[name]
        paths["json"].write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        flat_records = [self._flatten_record(item) for item in records]
        fieldnames = sorted({key for record in flat_records for key in record})
        with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
            if not fieldnames:
                handle.write("")
                return
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in flat_records:
                writer.writerow(record)

    def _flatten_record(self, record: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        flattened: Dict[str, Any] = {}
        for key, value in record.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_record(value, full_key))
            elif isinstance(value, list):
                flattened[full_key] = json.dumps(value, ensure_ascii=False)
            else:
                flattened[full_key] = value
        return flattened

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._normalize_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize_value(item) for item in value]
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except (TypeError, ValueError):
                return str(value)
        if isinstance(value, Path):
            return str(value)
        return value
