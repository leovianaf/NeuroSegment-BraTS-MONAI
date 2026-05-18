#!/usr/bin/env python3
"""Cria um sample do dataset BraTS2020 mantendo a estrutura de diretórios.

Usa symlinks para apontar para os arquivos originais, economizando espaço
em disco. Mantém a proporção de treino/validação do pipeline original
(últimos ~11% para validação).

Uso :
    python src/sample_dataset.py -p 1 -s 42

"""

import argparse
import csv
import os
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SRC = PROJECT_ROOT / "data" / "raw" / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
VAL_SRC = PROJECT_ROOT / "data" / "raw" / "BraTS2020_ValidationData" / "MICCAI_BraTS2020_ValidationData"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODALITIES = ["flair", "t1", "t1ce", "t2"]
SEG_SUFFIX = "seg"


def get_patient_dirs(base_path: Path, pattern: str) -> list[Path]:
    """Retorna lista de diretórios de pacientes ordenados."""
    return sorted(base_path.glob(pattern))


def sample_patients(patients: list[Path], percentage: float, seed: int) -> list[Path]:
    """Amostra uma porcentagem dos pacientes, garantindo pelo menos 1."""
    n = max(1, round(len(patients) * percentage / 100))
    rng = random.Random(seed)
    return sorted(rng.sample(patients, n))


def create_symlinked_sample(src_patients: list[Path], dst_dir: Path, has_seg: bool = True) -> None:
    """Cria diretório de destino com symlinks para os pacientes selecionados."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for patient_dir in src_patients:
        patient_id = patient_dir.name
        dst_patient = dst_dir / patient_id
        dst_patient.mkdir(exist_ok=True)

        for modality in MODALITIES:
            src_file = patient_dir / f"{patient_id}_{modality}.nii"
            dst_file = dst_patient / f"{patient_id}_{modality}.nii"
            if src_file.exists():
                create_symlink(src_file, dst_file)

        if has_seg:
            src_seg = patient_dir / f"{patient_id}_{SEG_SUFFIX}.nii"
            dst_seg = dst_patient / f"{patient_id}_{SEG_SUFFIX}.nii"
            if src_seg.exists():
                create_symlink(src_seg, dst_seg)


def create_symlink(src: Path, dst: Path) -> None:
    """Cria symlink relativo, sobrescrevendo se já existir."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def copy_csv_files(src_dir: Path, dst_dir: Path) -> None:
    """Copia arquivos CSV do diretório original para o sample."""
    for csv_file in src_dir.glob("*.csv"):
        shutil.copy2(csv_file, dst_dir / csv_file.name)


def filter_csv_by_patients(csv_path: Path, dst_path: Path, patient_ids: set[str]) -> None:
    """Filtra linhas de um CSV mantendo apenas os pacientes do sample.

    Busca o ID do paciente em qualquer coluna da linha.
    """
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader
                if any(pid in cell for pid in patient_ids for cell in row)]
    with open(dst_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria um sample do dataset BraTS2020")
    parser.add_argument("--percentage", "-p", type=float, required=True,
                        help="Porcentagem do dataset (ex: 1 para 1%%, 5 para 5%%)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Seed para reprodutibilidade (default: 42)")
    args = parser.parse_args()

    pct = args.percentage
    seed = args.seed

    if pct <= 0 or pct > 100:
        parser.error("Porcentagem deve estar entre 0 e 100")

    # --- Training data ---
    print(f"=== Sample de {pct}% do dataset de treino (seed={seed}) ===")
    all_train = get_patient_dirs(TRAIN_SRC, "BraTS20_Training_*")
    if not all_train:
        raise FileNotFoundError(f"Nenhum paciente encontrado em {TRAIN_SRC}")

    sampled_train = sample_patients(all_train, pct, seed)

    # Manter a proporção original: últimos ~11% para validação
    val_ratio = 40 / len(all_train)
    n_val = max(1, round(len(sampled_train) * val_ratio))
    # Garantir que sobra pelo menos 1 para treino
    n_val = min(n_val, len(sampled_train) - 1) if len(sampled_train) > 1 else 0

    sampled_val_from_train = sampled_train[-n_val:] if n_val > 0 else []
    sampled_train_only = sampled_train[:-n_val] if n_val > 0 else sampled_train

    tag = f"{pct:.0f}pct" if pct == int(pct) else f"{pct}pct"

    # Dataset de treino (pacientes de treino + validação, para manter
    # compatibilidade com get_dataloaders que faz o split internamente)
    train_dst = PROCESSED_DIR / f"MICCAI_BraTS2020_TrainingData_sample_{tag}"
    print(f"  Treino: {len(sampled_train_only)} pacientes")
    print(f"  Val (do treino): {len(sampled_val_from_train)} pacientes")
    print(f"  Total: {len(sampled_train)} pacientes -> {train_dst.name}")
    create_symlinked_sample(sampled_train, train_dst, has_seg=True)
    copy_csv_files(TRAIN_SRC, train_dst)

    # Filtrar CSVs pelos pacientes do sample
    train_patient_ids = {p.name for p in sampled_train}
    for csv_file in train_dst.glob("*.csv"):
        filter_csv_by_patients(csv_file, csv_file, train_patient_ids)

    # --- Validation (test) data ---
    print(f"\n=== Sample de {pct}% do dataset de validação/teste (seed={seed}) ===")
    all_val = get_patient_dirs(VAL_SRC, "BraTS20_Validation_*")
    if all_val:
        sampled_val = sample_patients(all_val, pct, seed)
        val_dst = PROCESSED_DIR / f"MICCAI_BraTS2020_ValidationData_sample_{tag}"
        print(f"  Total: {len(sampled_val)} pacientes -> {val_dst.name}")
        create_symlinked_sample(sampled_val, val_dst, has_seg=False)
        copy_csv_files(VAL_SRC, val_dst)

        val_patient_ids = {p.name for p in sampled_val}
        for csv_file in val_dst.glob("*.csv"):
            filter_csv_by_patients(csv_file, csv_file, val_patient_ids)
    else:
        print("  Diretório de validação não encontrado, pulando.")

    print(f"\nPronto! Para usar o sample, passe o caminho para get_dataloaders():")
    print(f'  data_path="{train_dst}"')


if __name__ == "__main__":
    main()