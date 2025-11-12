# make_imagefolder_from_csv.py
import csv, json, re, shutil
from pathlib import Path
import argparse

INVALID = r'[<>:"/\\|?*]'  # Windows-invalid path chars

def sanitize_label(name: str) -> str:
    # trim, replace invalid chars with underscore, collapse spaces, strip dots
    s = re.sub(INVALID, "_", name.strip())
    s = re.sub(r"\s+", " ", s).strip().strip(".")
    return s

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", type=Path, default=Path("data/train"))
    p.add_argument("--csv", type=Path, default=Path("data/Training_set.csv"))
    p.add_argument("--copy", action="store_true", help="Copy files instead of moving")
    args = p.parse_args()

    train_dir: Path = args.train_dir.resolve()
    csv_path: Path = args.csv.resolve()

    if not train_dir.exists() or not train_dir.is_dir():
        raise SystemExit(f"[ERR] train_dir not found: {train_dir}")
    if not csv_path.exists():
        raise SystemExit(f"[ERR] CSV not found: {csv_path}")

    # Read mapping filename -> label
    mapping = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        # Expect columns: filename,label
        for i, row in enumerate(rdr, 1):
            fn = row.get("filename")
            lbl = row.get("label")
            if not fn or not lbl:
                print(f"[WARN] Row {i} missing filename/label, skipping: {row}")
                continue
            mapping[fn] = lbl

    # Build label sets and create dirs
    labels = sorted({sanitize_label(lbl) for lbl in mapping.values()})
    for lbl in labels:
        (train_dir / lbl).mkdir(parents=True, exist_ok=True)

    moved, copied, missing, skipped = 0, 0, 0, 0
    for fn, raw_label in mapping.items():
        src = train_dir / fn
        if not src.exists():
            # maybe CSV has case mismatch; try case-insensitive search
            candidates = [p for p in train_dir.iterdir() if p.is_file() and p.name.lower() == fn.lower()]
            if candidates:
                src = candidates[0]
            else:
                print(f"[MISS] {fn} not found in {train_dir}")
                missing += 1
                continue

        dst_dir = train_dir / sanitize_label(raw_label)
        dst = dst_dir / src.name

        if dst.exists():
            # already organized? skip
            skipped += 1
            continue

        if args.copy:
            shutil.copy2(src, dst)
            copied += 1
        else:
            shutil.move(str(src), str(dst))
            moved += 1

    # Create a deterministic class_to_idx map (ImageFolder sorts class names)
    class_to_idx = {lbl: i for i, lbl in enumerate(sorted(labels))}
    out_map = train_dir.parent / "class_to_idx.json"
    out_map.write_text(json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- Summary ---")
    print(f"Train root : {train_dir}")
    print(f"CSV        : {csv_path.name}")
    print(f"Labels     : {len(labels)}")
    print(f"Moved      : {moved}")
    print(f"Copied     : {copied}")
    print(f"Skipped    : {skipped} (already in place)")
    print(f"Missing    : {missing} (not found in train dir)")
    print(f"Wrote      : {out_map}")

if __name__ == "__main__":
    main()
