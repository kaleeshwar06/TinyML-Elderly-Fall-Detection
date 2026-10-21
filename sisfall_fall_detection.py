#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sisfall_fall_detection.py
=========================
Auditing + leakage-safe fall-detection experiments for a Kaggle "SisFall Enhanced" download.

The script makes NO assumption about the Kaggle file layout. It inspects every file
recursively, decides (and prints) what it found, and then follows one of two branches:

  * TABULAR branch : processed/engineered CSV/XLSX/Parquet/TXT tables with a detectable label.
  * RAW branch     : header-less numeric SisFall-style recordings (one file per trial), which are
                     windowed *inside* each recording and then featurised.

Every number written to disk is computed at run time. Nothing is hard-coded or estimated.

Usage:
    python sisfall_fall_detection.py --data_dir /path/to/kaggle/dataset --output_dir outputs
"""
import argparse
import inspect
import os
import platform
import re
import sys
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, roc_curve,
                             classification_report)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:  # pragma: no cover
    HAVE_XGB = False

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")

NOTES = []          # leakage / preprocessing / limitation notes collected at run time
FACTS = OrderedDict()  # facts used by the Markdown reports


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def hdr(title):
    print("\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100, flush=True)


def log(msg, note=False):
    print(msg, flush=True)
    if note:
        NOTES.append(msg)


def sd(a, b):
    return float(a) / float(b) if b else np.nan


def md_table(df, floatfmt=4):
    if df is None or len(df) == 0:
        return "_(none)_"

    def f(v):
        if isinstance(v, (float, np.floating)):
            return "NA" if np.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="SisFall Enhanced fall-detection audit + experiments")
    p.add_argument("--data_dir", default="data", help="Root folder of the extracted Kaggle dataset")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--mode", default="auto", choices=["auto", "tabular", "raw"])
    # fallbacks if automatic detection fails
    p.add_argument("--target_col", default=None, help="FALLBACK: force the label column")
    p.add_argument("--group_col", default=None, help="FALLBACK: force the grouping column (subject/recording)")
    p.add_argument("--table_file", default=None, help="FALLBACK: relative path of the table to use")
    p.add_argument("--binary_positive", default=None,
                   help="FALLBACK: comma-separated raw label values that mean FALL (all others = non-fall)")
    p.add_argument("--prefall_as", default="exclude", choices=["exclude", "nonfall"],
                   help="How to treat 'prefall' rows in the binary task")
    p.add_argument("--keep_features", default="", help="Comma-separated columns to protect from the name-based drop")
    p.add_argument("--filename_regex", default=r"^(?P<activity>[DF]\d{2})_(?P<subject>S[AE]\d{2})_R(?P<trial>\d{2})",
                   help="RAW branch: regex with named groups activity, subject, trial applied to the file stem")
    # raw-branch windowing (sampling rate is an ASSUMPTION taken from the original SisFall paper)
    p.add_argument("--sampling_rate", type=float, default=200.0)
    p.add_argument("--window_sec", type=float, default=2.0)
    p.add_argument("--step_sec", type=float, default=1.0)
    p.add_argument("--window_labeling", default="impact", choices=["impact", "recording"],
                   help="impact: fall windows = windows containing the acceleration peak of a fall recording, "
                        "other windows of fall recordings are dropped; recording: every window inherits the recording label")
    # evaluation
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--cv_folds", type=int, default=3, help="grouped CV folds on train+val (0 disables)")
    p.add_argument("--svm_max_train", type=int, default=20000)
    p.add_argument("--select_k", type=int, default=0, help="Optional SelectKBest inside pipeline (0 = off)")
    p.add_argument("--min_class_count", type=int, default=10)
    p.add_argument("--drop_duplicates", action="store_true", help="Remove exact duplicate rows (reported either way)")
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--max_files", type=int, default=0, help="Debug: limit number of raw files")
    # event-level metrics (only if you can assert the data are continuous recordings)
    p.add_argument("--time_col", default=None)
    p.add_argument("--time_unit", default=None, choices=[None, "s", "ms", "datetime"])
    p.add_argument("--assume_continuous", action="store_true",
                   help="Assert that rows are time-ordered samples of continuous recordings")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# file discovery, loading and audit
# --------------------------------------------------------------------------------------
def discover(root):
    root = Path(root)
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts):
            out.append(p)
    return out


def try_read_raw_numeric(path):
    """Header-less numeric signal (SisFall style: comma/space separated, trailing ';'). None if not."""
    try:
        with open(path, "r", errors="ignore") as f:
            lines = f.read().splitlines()
    except Exception:
        return None
    rows = []
    for ln in lines:
        ln = ln.strip().rstrip(";").strip()
        if not ln:
            continue
        parts = [x for x in re.split(r"[,;\s]+", ln) if x != ""]
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            return None
    if len(rows) < 2:
        return None
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        return None
    return np.asarray(rows, dtype=float)


def read_tables(path):
    """Return list of (name, DataFrame). Raises on failure."""
    ext = path.suffix.lower()
    if ext == ".parquet":
        return [(path.name, pd.read_parquet(path))]
    if ext in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        return [(f"{path.name}::{k}", v) for k, v in sheets.items()]
    if ext == ".tsv":
        return [(path.name, pd.read_csv(path, sep="\t"))]
    # csv / txt / dat
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=None, engine="python")
    return [(path.name, df)]


def audit_all(root, args):
    files = discover(root)
    hdr(f"STEP 1 - FILE INVENTORY ({len(files)} files under {root})")
    rows, col_rows = [], []
    tables, raw_files = [], []
    n_raw_printed = 0
    for p in files:
        rel = str(p.relative_to(root))
        ext = p.suffix.lower()
        rec = dict(path=rel, extension=ext, size_bytes=p.stat().st_size, kind="other", n_rows=np.nan,
                   n_cols=np.nan, columns="", dtypes="", missing_values=np.nan, duplicate_rows=np.nan, note="")
        try:
            if ext in (".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".txt", ".dat"):
                raw = None
                if ext in (".txt", ".dat"):
                    raw = try_read_raw_numeric(p)
                if raw is not None:
                    rec.update(kind="raw_numeric_signal", n_rows=raw.shape[0], n_cols=raw.shape[1],
                               columns="(no header) col0..col%d" % (raw.shape[1] - 1), dtypes="float64",
                               missing_values=int(np.isnan(raw).sum()),
                               duplicate_rows=int(pd.DataFrame(raw).duplicated().sum()))
                    raw_files.append(p)
                else:
                    for name, df in read_tables(p):
                        r2 = dict(rec)
                        r2.update(path=rel if name == p.name else f"{rel}::{name.split('::')[-1]}", kind="table",
                                  n_rows=df.shape[0], n_cols=df.shape[1],
                                  columns="; ".join(map(str, df.columns[:80])) + (" ..." if df.shape[1] > 80 else ""),
                                  dtypes="; ".join(f"{k}={v}" for k, v in df.dtypes.astype(str).value_counts().items()),
                                  missing_values=int(df.isna().sum().sum()),
                                  duplicate_rows=int(df.duplicated().sum()))
                        rows.append(r2)
                        tables.append((r2["path"], df))
                        for c in df.columns:
                            col_rows.append(dict(file=r2["path"], column=str(c), dtype=str(df[c].dtype),
                                                 n_missing=int(df[c].isna().sum()),
                                                 n_unique=int(df[c].nunique(dropna=True))))
                    continue
            else:
                rec["note"] = "unsupported extension (not read)"
        except Exception as e:
            rec["kind"] = "unreadable"
            rec["note"] = f"{type(e).__name__}: {e}"
        rows.append(rec)

    audit = pd.DataFrame(rows)
    # console report
    tab = audit[audit.kind == "table"]
    for _, r in tab.iterrows():
        print(f"\n[TABLE] {r.path}  shape=({r.n_rows}, {r.n_cols})  missing={r.missing_values}  duplicates={r.duplicate_rows}")
        print(f"        dtypes: {r.dtypes}")
        print(f"        columns: {r.columns}")
    rawdf = audit[audit.kind == "raw_numeric_signal"]
    if len(rawdf):
        print(f"\n[RAW NUMERIC SIGNAL FILES] {len(rawdf)} files; first 15 shown (all listed in dataset_audit.csv)")
        for _, r in rawdf.head(15).iterrows():
            print(f"   {r.path}  shape=({r.n_rows}, {r.n_cols})  missing={r.missing_values}  duplicate_rows={r.duplicate_rows}")
        print(f"   n_cols distribution: {rawdf.n_cols.value_counts().to_dict()}  | total rows: {int(rawdf.n_rows.sum())}")
    other = audit[~audit.kind.isin(["table", "raw_numeric_signal"])]
    if len(other):
        print(f"\n[OTHER / UNREADABLE FILES] {len(other)}")
        for _, r in other.head(40).iterrows():
            print(f"   {r.path}  kind={r.kind}  {r.note}")
    print("\nFile-type counts:", audit.groupby(["extension", "kind"]).size().to_dict())
    return audit, pd.DataFrame(col_rows), tables, raw_files


# --------------------------------------------------------------------------------------
# column role detection
# --------------------------------------------------------------------------------------
TARGET_STRONG = re.compile(r"(^|[_\-\s])(label|labels|target|class|y|is_fall|fall|falls|outcome)($|[_\-\s])", re.I)
TARGET_WEAK = re.compile(r"(activity|act|category|event|type|adl|status|code|motion|movement|action)", re.I)
SUBJECT_PAT = re.compile(r"(subject|subj|participant|volunteer|person|patient|user|^sid$|^pid$)", re.I)
RECORD_PAT = re.compile(r"(trial|session|recording|record|file|filename|source|^run$|sequence|episode|^rec)", re.I)
TIME_PAT = re.compile(r"^(time|timestamp|datetime|date|t|ts|seconds|sec|ms|elapsed|time_?stamp)([_\-\s].*)?$|(timestamp|datetime)", re.I)
INDEX_PAT = re.compile(r"(^unnamed|^index$|^idx$|^id$|_id$|^row|row_?id|^window_?id|^sample_?id)", re.I)
NAME_DROP_PAT = re.compile(r"(label|target|class|fall|activity|adl|subject|participant|filename|file_name|trial|session|recording|timestamp|^id$|_id$|^index$|^unnamed)", re.I)
ACC_PAT = re.compile(r"(acc|accel|accelerometer|adxl|mma|(^|[_\W])a[xyz]($|[_\W]))", re.I)
GYR_PAT = re.compile(r"(gyr|gyro|gyroscope|itg|angular|(^|[_\W])g[xyz]($|[_\W]))", re.I)


def detect_roles(df):
    n = len(df)
    tc, sc, rc, tmc, ic = [], [], [], [], []
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        name = str(c)
        is_float_cont = pd.api.types.is_float_dtype(s) and nun > 50
        score = 0.0
        if TARGET_STRONG.search(name):
            score += 3
        if TARGET_WEAK.search(name):
            score += 1.5
        if score > 0 and 2 <= nun <= 100 and not is_float_cont:
            score += 1.0
            if nun == 2:
                score += 0.5
            tc.append((c, score, nun))
        vals_sub = s.dropna().astype(str).head(2000)
        if SUBJECT_PAT.search(name) or (len(vals_sub) and vals_sub.str.match(r"^S[AE]\d{2}$").mean() > 0.9):
            sc.append((c, nun))
        if RECORD_PAT.search(name):
            rc.append((c, nun))
        if TIME_PAT.search(name):
            tmc.append((c, nun))
        if INDEX_PAT.search(name) or (pd.api.types.is_integer_dtype(s) and nun == n and n > 20 and not TARGET_STRONG.search(name)):
            ic.append((c, nun))
    tc.sort(key=lambda t: -t[1])
    return dict(target=tc, subject=sc, record=rc, time=tmc, index=ic)


def norm_label(v):
    return re.sub(r"[\s\-]+", "_", str(v).strip().lower())


def label_semantics(v, pos_set=None):
    """Return 'fall' | 'nonfall' | 'prefall' | 'unknown' using ONLY the label text/code."""
    s = norm_label(v)
    if pos_set is not None:
        return "fall" if s in pos_set else "nonfall"
    if re.search(r"pre_?fall|pre_?impact|impending", s):
        return "prefall"
    if s in ("1", "1.0", "true", "yes", "fall", "falls", "f") or re.fullmatch(r"f\d{1,2}", s):
        return "fall"
    if s in ("0", "0.0", "false", "no", "nonfall", "non_fall", "adl", "adls", "d", "n", "nofall", "no_fall", "not_fall", "normal") \
            or re.fullmatch(r"d\d{1,2}", s):
        return "nonfall"
    if re.search(r"non_?fall|no_?fall|not_?fall|adl|daily", s):
        return "nonfall"
    if "fall" in s:
        return "fall"
    return "unknown"


# --------------------------------------------------------------------------------------
# dataset construction
# --------------------------------------------------------------------------------------
def build_tabular(args, tables, audit):
    hdr("STEP 2 - TABULAR BRANCH: target / group / time candidates")
    cand = []
    for name, df in tables:
        if args.table_file and args.table_file not in name:
            continue
        if df.shape[0] < 20 or df.shape[1] < 3:
            continue
        roles = detect_roles(df)
        best = roles["target"][0][1] if roles["target"] else 0
        if args.target_col and args.target_col in df.columns:
            best = 99
        cand.append((best, df.shape[0], name, df, roles))
    if not cand:
        return None
    cand.sort(key=lambda t: (-t[0], -t[1]))
    best_score = cand[0][0]
    if best_score == 0:
        log("No table with a plausible label column found.")
        return None
    # tables sharing the same columns as the chosen one are concatenated
    chosen_cols = list(cand[0][3].columns)
    parts = [(c[2], c[3]) for c in cand if list(c[3].columns) == chosen_cols]
    for nm, d in parts:
        print(f"  using table: {nm} shape={d.shape}")
    if len(parts) > 1:
        log(f"{len(parts)} tables with identical columns were concatenated; any pre-existing train/test partition "
            f"implied by file names is IGNORED and re-split by this script.", note=True)
    df = pd.concat([d.assign(__source_file=nm) for nm, d in parts], ignore_index=True)
    roles = detect_roles(df.drop(columns="__source_file"))
    FACTS["tables_used"] = [nm for nm, _ in parts]

    print("\n  TARGET candidates (column, score, n_unique):", roles["target"])
    print("  SUBJECT/PARTICIPANT candidates (column, n_unique):", roles["subject"])
    print("  SESSION/TRIAL/RECORDING candidates (column, n_unique):", roles["record"])
    print("  TIMESTAMP candidates (column, n_unique):", roles["time"])
    print("  INDEX-like candidates:", roles["index"])

    # ---- target decision
    if args.target_col:
        if args.target_col not in df.columns:
            sys.exit(f"[FALLBACK ERROR] --target_col '{args.target_col}' not in columns.")
        target = args.target_col
        log(f"TARGET DECISION: user-forced fallback column '{target}'.")
    elif roles["target"]:
        target = roles["target"][0][0]
        log(f"TARGET DECISION: automatic -> '{target}' (score {roles['target'][0][1]}, {roles['target'][0][2]} classes).")
    else:
        sys.exit("[AUTO-DETECTION FAILED] No target column found. Re-run with --target_col <name>.")
    print("  Class distribution of target (raw values):")
    print(df[target].value_counts(dropna=False).to_string())
    if df[target].isna().any():
        log(f"{int(df[target].isna().sum())} rows with missing target were removed.", note=True)
        df = df[df[target].notna()].reset_index(drop=True)

    # ---- label semantics
    pos_set = None
    if args.binary_positive:
        pos_set = {norm_label(x) for x in args.binary_positive.split(",")}
        log(f"BINARY MAPPING: user-forced fallback; FALL = {sorted(pos_set)}, everything else = non-fall.")
    uniq = list(pd.unique(df[target]))
    mapping = {u: label_semantics(u, pos_set) for u in uniq}
    mp = pd.DataFrame({"raw_value": [str(k) for k in mapping], "interpreted_as": list(mapping.values()),
                       "count": [int((df[target] == k).sum()) for k in mapping]})
    print("\n  LABEL MAPPING (derived from label text/code only):")
    print(mp.to_string(index=False))
    if (mp.interpreted_as == "unknown").any():
        sys.exit("[AUTO-MAPPING FAILED] Some label values cannot be interpreted as fall/non-fall.\n"
                 "Re-run with --binary_positive \"<value1,value2>\" listing the raw values that mean FALL.")
    if all(norm_label(k) in ("0", "1", "0.0", "1.0", "true", "false") for k in mapping):
        log("NOTE: numeric/boolean target interpreted as 1 = fall, 0 = non-fall. Verify with the dataset documentation.",
            note=True)
    sem = df[target].map(mapping)
    if (sem == "prefall").any():
        if args.prefall_as == "exclude":
            log(f"{int((sem == 'prefall').sum())} 'prefall' rows excluded from the binary task (--prefall_as exclude).", note=True)
        else:
            sem = sem.replace("prefall", "nonfall")
    keep = sem.isin(["fall", "nonfall"]) if args.prefall_as == "exclude" else np.ones(len(df), bool)
    df, sem = df[keep].reset_index(drop=True), sem[keep].reset_index(drop=True)
    y_bin = (sem == "fall").astype(int).values
    if len(np.unique(y_bin)) < 2:
        sys.exit("Only one binary class remains after mapping; cannot train a detector.")

    # ---- duplicates
    feat_cols_tmp = [c for c in df.columns if c != "__source_file"]
    ndup = int(df.duplicated(subset=feat_cols_tmp).sum())
    log(f"DUPLICATES: {ndup} exact duplicate rows found (all columns).")
    if args.drop_duplicates and ndup:
        keep_mask = ~df.duplicated(subset=feat_cols_tmp)
        log(f"DUPLICATES: removing {ndup} rows (--drop_duplicates).", note=True)
        df, y_bin = df[keep_mask.values].reset_index(drop=True), y_bin[keep_mask.values]
    elif ndup:
        log("DUPLICATES: kept (use --drop_duplicates to remove). Cross-partition duplicate leakage is tested later.", note=True)

    # ---- group decision
    group_col, group_kind = None, None
    if args.group_col:
        if args.group_col not in df.columns:
            sys.exit(f"[FALLBACK ERROR] --group_col '{args.group_col}' not in columns.")
        group_col, group_kind = args.group_col, "user-forced"
    else:
        for c, nun in sorted(roles["subject"], key=lambda t: -t[1]):
            if 3 <= nun < len(df) * 0.5 and c != target:
                group_col, group_kind = c, "subject/participant"
                break
        if group_col is None:
            for c, nun in sorted(roles["record"], key=lambda t: -t[1]):
                if 3 <= nun < len(df) * 0.9 and c != target:
                    group_col, group_kind = c, "recording/session/trial"
                    break
    if group_col:
        log(f"GROUP DECISION: grouping column '{group_col}' ({group_kind}); {df[group_col].nunique()} groups. Grouped splitting will be used.")
        groups = df[group_col].astype(str).values
    else:
        groups = None
        log("GROUP DECISION: NO subject/participant/session/recording identifier detected. Stratified row-wise splitting "
            "will be used. LIMITATION: subject-independent generalisation cannot be established and, if rows are "
            "overlapping windows, train/test leakage cannot be excluded.", note=True)

    # ---- secondary (multiclass) target
    multi_col = None
    if df[target].nunique() > 2:
        multi_col = target
    else:
        for c, sc_, nun in roles["target"]:
            if c != target and 3 <= nun <= 100:
                multi_col = c
                break
    if multi_col is not None and multi_col in df.columns:
        log(f"MULTICLASS TARGET: '{multi_col}' ({df[multi_col].nunique()} classes).")
    else:
        multi_col = None
        log("MULTICLASS TARGET: unavailable - no label column with >2 valid classes was found.", note=True)
    y_multi = df[multi_col].astype(str).values if multi_col else None

    # ---- feature matrix: remove all non-predictors
    protect = {c.strip() for c in args.keep_features.split(",") if c.strip()}
    excluded = OrderedDict()
    drop_roles = {target: "target"}
    if multi_col:
        drop_roles[multi_col] = "secondary target"
    for c, _, _ in roles["target"]:
        drop_roles.setdefault(c, "label-like column")
    for k, tag in [("subject", "subject/participant id"), ("record", "session/trial/recording id"),
                   ("time", "timestamp"), ("index", "index/row id")]:
        for c, _ in roles[k]:
            drop_roles.setdefault(c, tag)
    if group_col:
        drop_roles.setdefault(group_col, "group id")
    drop_roles["__source_file"] = "source filename"
    for c in df.columns:
        if c in drop_roles:
            excluded[c] = drop_roles[c]
        elif c not in protect and NAME_DROP_PAT.search(str(c)):
            excluded[c] = "name suggests label/identifier/leakage"
    X = df.drop(columns=[c for c in excluded if c in df.columns])
    print("\n  COLUMNS REMOVED FROM FEATURE MATRIX:")
    for c, why in excluded.items():
        print(f"    - {c}: {why}")
    # numeric coercion
    dropped_nonnum = []
    for c in list(X.columns):
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            conv = pd.to_numeric(X[c], errors="coerce")
            if conv.notna().mean() >= 0.95 * X[c].notna().mean() and X[c].notna().any():
                X[c] = conv
            else:
                dropped_nonnum.append(c)
    if dropped_nonnum:
        log(f"Non-numeric feature columns dropped (no encoding applied): {dropped_nonnum}", note=True)
        X = X.drop(columns=dropped_nonnum)
    X = X.replace([np.inf, -np.inf], np.nan).astype(float)
    if X.shape[1] == 0:
        sys.exit("No numeric feature columns remain after removing labels/identifiers.")

    meta = pd.DataFrame({"row_index": np.arange(len(df))})
    meta["group"] = groups if groups is not None else "NA"
    for k in ("subject", "record"):
        for c, _ in roles[k][:1]:
            if c in df.columns:
                meta[f"{k}_col:{c}"] = df[c].astype(str).values
    meta["raw_label"] = df[target].astype(str).values
    if multi_col:
        meta["activity"] = df[multi_col].astype(str).values
    tcol = args.time_col or (roles["time"][0][0] if roles["time"] else None)
    rcol = group_col if group_kind and "recording" in group_kind else (roles["record"][0][0] if roles["record"] else None)
    ev = None
    if tcol and tcol in df.columns:
        ev = dict(time=df[tcol].values, rec=df[rcol].astype(str).values if rcol and rcol in df.columns else None,
                  time_col=tcol, rec_col=rcol)
    FACTS.update(mode="tabular", target_col=target, group_col=group_col, group_kind=group_kind,
                 label_mapping=mp, excluded=excluded, multi_col=multi_col, windowing="none (rows used as provided)",
                 time_col=tcol, record_col=rcol)
    if rcol is None and groups is not None:
        pass
    log("NOTE: rows are used exactly as provided. If each row is a single time sample (rather than a window), "
        "row-level classification is NOT equivalent to windowed fall detection.", note=True)
    rec_for_check = df[rcol].astype(str).values if rcol and rcol in df.columns else None
    return dict(X=X, y_bin=y_bin, y_multi=y_multi, groups=groups, meta=meta, ev=ev, rec_check=rec_for_check)


def window_features(arr, names, W, S):
    """arr: (n, c). returns (feature matrix [n_win, n_feat], feature names, window start indices)."""
    n = arr.shape[0]
    if n < W:
        return None, None, None
    win = np.lib.stride_tricks.sliding_window_view(arr, W, axis=0)[::S]  # (nw, c, W)
    starts = np.arange(0, n - W + 1, S)[: win.shape[0]]
    feats, fn = [], []

    def add(label, vals):
        feats.append(vals)
        fn.append(label)
    for j, nm in enumerate(names):
        w = win[:, j, :]
        add(f"{nm}__mean", w.mean(-1)); add(f"{nm}__std", w.std(-1))
        add(f"{nm}__min", w.min(-1)); add(f"{nm}__max", w.max(-1))
        add(f"{nm}__range", np.ptp(w, axis=-1)); add(f"{nm}__rms", np.sqrt((w ** 2).mean(-1)))
        add(f"{nm}__median", np.median(w, axis=-1))
        add(f"{nm}__skew", sps.skew(w, axis=-1)); add(f"{nm}__kurt", sps.kurtosis(w, axis=-1))
        add(f"{nm}__mean_abs_diff", np.abs(np.diff(w, axis=-1)).mean(-1))
    return np.stack(feats, axis=1), fn, starts


def build_raw(args, raw_files):
    hdr("STEP 2 - RAW BRANCH: header-less numeric recordings -> windows")
    rx = re.compile(args.filename_regex)
    files = raw_files[: args.max_files] if args.max_files else raw_files
    parsed, unparsed = [], []
    for p in files:
        m = rx.search(p.stem)
        (parsed if m else unparsed).append((p, m))
    log(f"Filename regex '{args.filename_regex}' matched {len(parsed)} / {len(files)} files.")
    if not parsed:
        sys.exit("[AUTO-DETECTION FAILED] No file name matches the regex, so labels/subjects cannot be derived. "
                 "Re-run with --filename_regex containing named groups (activity, subject, trial).")
    if unparsed:
        log(f"{len(unparsed)} raw files do not match and are excluded (e.g. {unparsed[0][0].name}).", note=True)
    ncols_all = {}
    fs = args.sampling_rate
    W, S = int(round(args.window_sec * fs)), int(round(args.step_sec * fs))
    log(f"WINDOWING (ASSUMPTION): sampling rate {fs} Hz is taken from the original SisFall description, NOT verified "
        f"from the files; window = {W} samples ({args.window_sec}s), step = {S} samples ({args.step_sec}s); "
        f"windows are created inside each recording only.", note=True)
    X_blocks, meta_rows, fnames = [], [], None
    short = 0
    n_dropped_nonimpact = 0
    for p, m in parsed:
        arr = try_read_raw_numeric(p)
        if arr is None:
            continue
        nc = arr.shape[1]
        ncols_all[nc] = ncols_all.get(nc, 0) + 1
        if nc == 9:   # original SisFall layout: ADXL345 acc, ITG3200 gyro, MMA8451Q acc
            names = ["acc1_x", "acc1_y", "acc1_z", "gyro_x", "gyro_y", "gyro_z", "acc2_x", "acc2_y", "acc2_z"]
            mags = {"acc1_svm": arr[:, 0:3], "gyro_svm": arr[:, 3:6], "acc2_svm": arr[:, 6:9]}
        else:
            names = [f"ch{j}" for j in range(nc)]
            mags = {}
        ext = np.column_stack([arr] + [np.sqrt((v.astype(float) ** 2).sum(1)) for v in mags.values()])
        cn = names + list(mags.keys())
        F, fn, starts = window_features(ext, cn, W, S)
        if F is None:
            short += 1
            continue
        gd = m.groupdict()
        act, subj, trial = gd.get("activity", p.stem), gd.get("subject", str(p.parent.name)), gd.get("trial", "")
        is_fall_rec = label_semantics(act) == "fall"
        lab = np.full(len(F), int(is_fall_rec))
        keep = np.ones(len(F), bool)
        if is_fall_rec and args.window_labeling == "impact":
            ref = ext[:, cn.index("acc1_svm")] if "acc1_svm" in cn else np.linalg.norm(arr[:, :3], axis=1)
            pk = int(np.argmax(ref))
            keep = (starts <= pk) & (pk < starts + W)
            n_dropped_nonimpact += int((~keep).sum())
        if keep.sum() == 0:
            continue
        X_blocks.append(F[keep])
        fnames = fn
        for st in starts[keep]:
            meta_rows.append((str(p.relative_to(args.data_dir)), subj, act, trial, int(st)))
    if not X_blocks:
        sys.exit("No windows could be produced from the raw files.")
    if len(ncols_all) > 1:
        log(f"WARNING: raw files have different column counts: {ncols_all}", note=True)
    if 9 in ncols_all:
        log("SENSOR MAPPING (from original SisFall documentation, only the column COUNT = 9 was verified): "
            "cols 0-2 = ADXL345 accelerometer, 3-5 = ITG3200 gyroscope, 6-8 = MMA8451Q accelerometer.", note=True)
    else:
        log("Raw files do not have 9 columns: sensor type cannot be assigned, so sensor ablations are unavailable.", note=True)
    if short:
        log(f"{short} recordings shorter than one window were skipped.", note=True)
    if args.window_labeling == "impact":
        log(f"WINDOW LABELLING: fall = window containing the peak accelerometer magnitude of a fall recording "
            f"(heuristic, not an annotated impact time); {n_dropped_nonimpact} other windows of fall recordings were dropped "
            f"because their label is ambiguous. ADL windows are all non-fall.", note=True)
    else:
        log("WINDOW LABELLING: every window inherits its recording label (windows before/after the fall are labelled "
            "fall -> label noise).", note=True)
    X = pd.DataFrame(np.vstack(X_blocks), columns=fnames).replace([np.inf, -np.inf], np.nan)
    meta = pd.DataFrame(meta_rows, columns=["file", "subject", "activity", "trial", "win_start"])
    meta.insert(0, "row_index", np.arange(len(meta)))
    meta["group"] = meta["subject"]
    y_bin = np.array([int(label_semantics(a) == "fall") for a in meta["activity"]])
    mp = pd.DataFrame({"raw_value": sorted(meta["activity"].unique())})
    mp["interpreted_as"] = [label_semantics(v) for v in mp.raw_value]
    mp["count"] = [int((meta.activity == v).sum()) for v in mp.raw_value]
    print(mp.to_string(index=False))
    FACTS.update(mode="raw", target_col="activity code parsed from file name (first letter D = ADL, F = fall)",
                 group_col="subject (parsed from file name)", group_kind="subject",
                 label_mapping=mp, excluded={"file/subject/trial/activity/win_start": "metadata, never used as predictors"},
                 multi_col="activity", windowing=f"{W}-sample windows, {S}-sample step, labelling={args.window_labeling}",
                 time_col=None, record_col="file")
    return dict(X=X, y_bin=y_bin, y_multi=meta["activity"].values.astype(str), groups=meta["subject"].values.astype(str),
                meta=meta, ev=None, rec_check=meta["file"].values.astype(str))


# --------------------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------------------
def make_splits(y, groups):
    idx = np.arange(len(y))
    if groups is not None:
        ng = len(np.unique(groups))
        try:
            if ng >= 8:
                dev, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(idx, y, groups))
                a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=SEED).split(dev, y[dev], groups[dev]))
                return dev[a], dev[b], te, "StratifiedGroupKFold (group-wise 60/20/20 approx.)"
            if ng >= 3:
                log(f"Only {ng} groups: using GroupShuffleSplit (class balance per partition not guaranteed).", note=True)
                dev, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(idx, y, groups))
                a, b = next(GroupShuffleSplit(1, test_size=0.34, random_state=SEED).split(dev, y[dev], groups[dev]))
                return dev[a], dev[b], te, "GroupShuffleSplit"
        except Exception as e:
            log(f"Grouped split failed ({e}); falling back to stratified row split.", note=True)
        log("Too few groups for grouped splitting -> stratified row split (LIMITATION).", note=True)
    dev, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    tr, va = train_test_split(dev, test_size=0.25, stratify=y[dev], random_state=SEED)
    return tr, va, te, "stratified row split 60/20/20 (NO grouping)"


def summarize_splits(tr, va, te, y, groups, rec, outdir):
    hdr("STEP 4 - SPLIT SUMMARY AND LEAKAGE CHECKS")
    rows = []
    sets = {}
    for nm, ix in (("train", tr), ("validation", va), ("test", te)):
        g = sorted(set(groups[ix])) if groups is not None else []
        sets[nm] = set(g)
        rows.append(dict(partition=nm, n_rows=len(ix), pct_rows=100 * len(ix) / len(y), n_fall=int(y[ix].sum()),
                         n_nonfall=int((y[ix] == 0).sum()), n_groups=len(g), groups="; ".join(g) if g else "NA (no grouping)"))
        print(f"\n[{nm.upper()}] rows={len(ix)}  fall={int(y[ix].sum())}  nonfall={int((y[ix]==0).sum())}  groups={len(g)}")
        if g:
            print("   groups:", ", ".join(g))
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "split_summary.csv", index=False)
    checks = []
    if groups is not None:
        ov = {f"{a}&{b}": sorted(sets[a] & sets[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
        for k, v in ov.items():
            checks.append(f"Group overlap {k}: {len(v)} {v if v else ''}")
        if any(ov.values()):
            sys.exit("LEAKAGE: a group appears in more than one partition. Aborting.")
    else:
        checks.append("No group identifier: group-overlap check not possible (limitation).")
    if rec is not None:
        rs = {nm: set(rec[ix]) for nm, ix in (("train", tr), ("validation", va), ("test", te))}
        n_ov = len(rs["train"] & rs["test"]) + len(rs["train"] & rs["validation"]) + len(rs["validation"] & rs["test"])
        checks.append(f"Recording/file-id overlap across partitions: {n_ov}")
        if n_ov:
            log("WARNING: recording ids overlap across partitions.", note=True)
    for c in checks:
        log("LEAKAGE CHECK: " + c, note=True)
    return df


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------
def make_models(args, spw, k_feat):
    def sel():
        return [("sel", SelectKBest(f_classif, k=min(args.select_k, k_feat)))] if args.select_k > 0 else []
    m = OrderedDict()
    m["Dummy_majority"] = Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", DummyClassifier(strategy="most_frequent"))])
    m["LogisticRegression"] = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())] + sel() +
                                       [("clf", LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=SEED))])
    m["RandomForest"] = Pipeline([("imp", SimpleImputer(strategy="median"))] + sel() +
                                 [("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                                 n_jobs=args.n_jobs, random_state=SEED))])
    m["SVM_rbf"] = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())] + sel() +
                            [("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED))])
    hgb_kw = dict(random_state=SEED, max_iter=200, learning_rate=0.1)
    if "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters:
        hgb_kw["class_weight"] = "balanced"
    m["HistGradientBoosting"] = Pipeline([("imp", SimpleImputer(strategy="median"))] + sel() +
                                         [("clf", HistGradientBoostingClassifier(**hgb_kw))])
    if HAVE_XGB:
        kw = dict(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                  eval_metric="logloss", random_state=SEED, n_jobs=args.n_jobs if args.n_jobs != 0 else 1, verbosity=0)
        if spw is not None:
            kw["scale_pos_weight"] = spw
        m["XGBoost"] = Pipeline([("imp", SimpleImputer(strategy="median"))] + sel() + [("clf", XGBClassifier(**kw))])
    return m


def fit_model(name, model, X, y, args):
    mdl = clone(model)
    if name == "SVM_rbf" and len(y) > args.svm_max_train:
        rng = np.random.RandomState(SEED)
        try:
            sub, _ = train_test_split(np.arange(len(y)), train_size=args.svm_max_train, stratify=y, random_state=SEED)
        except Exception:
            sub = rng.choice(len(y), args.svm_max_train, replace=False)
        FACTS["svm_subsampled"] = f"SVM trained on a stratified subsample of {args.svm_max_train} training rows (of {len(y)})."
        mdl.fit(X[sub], y[sub])
    else:
        mdl.fit(X, y)
    return mdl


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def binary_metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens, spec = sd(tp, tp + fn), sd(tn, tn + fp)
    m = OrderedDict(TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
                    accuracy=sd(tp + tn, tp + tn + fp + fn), precision=sd(tp, tp + fp), recall_sensitivity=sens,
                    specificity=spec, f1=sd(2 * tp, 2 * tp + fp + fn),
                    balanced_accuracy=float(np.nanmean([sens, spec])), mcc=float(matthews_corrcoef(y, pred)),
                    npv=sd(tn, tn + fn), fpr=sd(fp, fp + tn), fnr=sd(fn, fn + tp))
    if len(np.unique(y)) == 2:
        m["roc_auc"] = float(roc_auc_score(y, score))
        m["pr_auc"] = float(average_precision_score(y, score))
    else:
        m["roc_auc"] = m["pr_auc"] = np.nan
    return m


def multi_core(y, pred):
    labels = sorted(set(y) | set(pred))
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
    try:
        kap = float(cohen_kappa_score(y, pred)) if len(labels) > 1 else np.nan
    except Exception:
        kap = np.nan
    return OrderedDict(accuracy=float(np.mean(y == pred)), macro_precision=float(p), macro_recall=float(r), macro_f1=float(f),
                       weighted_f1=float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
                       balanced_accuracy=float(balanced_accuracy_score(y, pred)), mcc=float(matthews_corrcoef(y, pred)),
                       cohen_kappa=kap)


def ovr_auc(y, proba, classes_idx):
    """Per-class OvR AUC for classes present (with both pos and neg) in y; returns (macro, n_classes_used)."""
    aucs = []
    for j, c in enumerate(classes_idx):
        yy = (y == c).astype(int)
        if 0 < yy.sum() < len(yy):
            aucs.append(roc_auc_score(yy, proba[:, j]))
    return (float(np.mean(aucs)) if aucs else np.nan), len(aucs)


def bootstrap_ci(y, pred, score, groups, fn, n_boot, keys):
    rng = np.random.RandomState(SEED)
    n = len(y)
    if groups is not None:
        ug = np.unique(groups)
        by = {g: np.where(groups == g)[0] for g in ug}
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        if groups is not None:
            gs = rng.choice(ug, len(ug), replace=True)
            ix = np.concatenate([by[g] for g in gs])
        else:
            ix = rng.randint(0, n, n)
        if len(np.unique(y[ix])) < 2:
            continue
        try:
            m = fn(y[ix], pred[ix], None if score is None else score[ix])
        except Exception:
            continue
        for k in keys:
            if k in m and not (isinstance(m[k], float) and np.isnan(m[k])):
                vals[k].append(m[k])
    out = {}
    for k in keys:
        v = np.array(vals[k])
        out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)) if len(v) >= 20 else (np.nan, np.nan, len(v))
    return out


# --------------------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------------------
def plot_confusion(cm, labels, title, path):
    k = len(labels)
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * k + 3), max(4.5, 0.45 * k + 2.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=90 if k > 6 else 0); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    if k <= 15:
        for i in range(k):
            for j in range(k):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------------------
# one full task (binary or multiclass)
# --------------------------------------------------------------------------------------
def run_task(task, X, y, groups, meta, tr, va, te, class_names, args, outdir, ev=None):
    binary = task == "binary"
    hdr(f"STEP 5 - TASK: {task.upper()}")
    R = dict(task=task)
    # ---- multiclass: keep only classes seen in train, re-encode
    if not binary:
        ytr_classes = sorted(np.unique(y[tr]))
        keepv, keept = np.isin(y[va], ytr_classes), np.isin(y[te], ytr_classes)
        if (~keepv).sum() or (~keept).sum():
            log(f"[multiclass] removed {int((~keepv).sum())} validation and {int((~keept).sum())} test rows whose class "
                f"never occurs in train.", note=True)
        va, te = va[keepv], te[keept]
        remap = {c: i for i, c in enumerate(ytr_classes)}
        y = np.array([remap.get(c, -1) for c in y])
        class_names = [class_names[c] for c in ytr_classes]
    labels_idx = [0, 1] if binary else list(range(len(class_names)))
    names = ["nonfall", "fall"] if binary else class_names

    Xdf = X
    # ---- feature cleaning on TRAIN only
    Xtr_df = Xdf.iloc[tr]
    allnan = [c for c in Xdf.columns if Xtr_df[c].isna().all()]
    const = [c for c in Xdf.columns if c not in allnan and Xtr_df[c].nunique(dropna=True) <= 1]
    feat = [c for c in Xdf.columns if c not in allnan and c not in const]
    log(f"Feature cleaning (train statistics only): {len(allnan)} all-NaN and {len(const)} constant columns removed; "
        f"{len(feat)} features remain.", note=True)
    R["n_features"], R["removed_const"], R["removed_allnan"] = len(feat), len(const), len(allnan)
    if not feat:
        log("No usable features.")
        return None
    A = Xdf[feat].to_numpy(float)
    R["feat"] = feat

    # ---- single-feature leakage screen (binary, train only)
    if binary:
        flagged = []
        for j, c in enumerate(feat[:1000]):
            col = np.nan_to_num(A[tr, j], nan=np.nanmedian(A[tr, j]) if np.isfinite(A[tr, j]).any() else 0.0)
            if len(np.unique(y[tr])) == 2 and np.unique(col).size > 1:
                a = roc_auc_score(y[tr], col)
                if max(a, 1 - a) >= 0.999:
                    flagged.append((c, round(max(a, 1 - a), 4)))
        msg = f"Single-feature train-AUC >= 0.999 (possible label leakage): {flagged if flagged else 'none found'}"
        log("LEAKAGE CHECK: " + msg, note=True)
    # cross-partition identical feature vectors
    h = pd.util.hash_pandas_object(pd.DataFrame(A).round(6), index=False).values
    dup_te = int(np.isin(h[te], h[tr]).sum()); dup_va = int(np.isin(h[va], h[tr]).sum())
    log(f"LEAKAGE CHECK: test rows whose feature vector is identical to a train row: {dup_te}; validation: {dup_va}", note=True)
    R["dup_test_in_train"] = dup_te

    spw = float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())) if binary else None
    models = make_models(args, spw, len(feat))
    if not HAVE_XGB:
        log("XGBoost not installed -> skipped (HistGradientBoosting from scikit-learn is used).", note=True)
    sel_key = "f1" if binary else "macro_f1"
    sel_name = "validation fall-class F1" if binary else "validation macro-F1"
    dev = np.concatenate([tr, va])

    # ---- model comparison
    rows, fitted_val = [], {}
    for name, mdl in models.items():
        t0 = time.time()
        row = dict(task=task, model=name)
        try:
            fm = fit_model(name, mdl, A[tr], y[tr], args)
            pv = fm.predict(A[va])
            pr = fm.predict_proba(A[va])
            if binary:
                m = binary_metrics(y[va], pv, pr[:, 1])
                row.update({f"val_{k}": v for k, v in m.items() if k in ("accuracy", "precision", "recall_sensitivity", "specificity",
                                                                           "f1", "balanced_accuracy", "mcc", "roc_auc", "pr_auc")})
            else:
                m = multi_core(y[va], pv)
                row.update({f"val_{k}": v for k, v in m.items()})
            row["fit_seconds"] = round(time.time() - t0, 2)
            row["selection_metric_value"] = m[sel_key]
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            row["selection_metric_value"] = np.nan
            log(f"  [{name}] FAILED: {row['error']}")
        # grouped CV on train+val
        if args.cv_folds and args.cv_folds >= 2 and "error" not in row:
            try:
                if groups is not None:
                    splitter = StratifiedGroupKFold(args.cv_folds, shuffle=True, random_state=SEED).split(A[dev], y[dev], groups[dev])
                else:
                    splitter = StratifiedKFold(args.cv_folds, shuffle=True, random_state=SEED).split(A[dev], y[dev])
                sc = []
                for a_, b_ in splitter:
                    f_ = fit_model(name, mdl, A[dev][a_], y[dev][a_], args)
                    p_ = f_.predict(A[dev][b_])
                    sc.append(f1_score(y[dev][b_], p_, pos_label=1, zero_division=0) if binary
                              else f1_score(y[dev][b_], p_, average="macro", zero_division=0))
                row["cv_mean_" + sel_key], row["cv_std_" + sel_key] = float(np.mean(sc)), float(np.std(sc))
                row["cv_type"] = "StratifiedGroupKFold" if groups is not None else "StratifiedKFold (no groups)"
            except Exception as e:
                row["cv_note"] = f"CV skipped: {type(e).__name__}: {e}"
        rows.append(row)
        print(f"  {name:22s} {sel_name}={row.get('selection_metric_value', np.nan):.4f}  "
              f"(cv={row.get('cv_mean_' + sel_key, np.nan):.4f})  {row.get('error', '')}")
    comp = pd.DataFrame(rows)
    cand = comp[(comp.model != "Dummy_majority") & comp.selection_metric_value.notna()]
    if cand.empty:
        log("No model could be trained.")
        return None
    best = cand.sort_values("selection_metric_value", ascending=False).iloc[0]["model"]
    comp["selected"] = comp.model == best
    log(f"MODEL SELECTION: '{best}' selected by {sel_name} (Dummy baseline excluded from selection).", note=True)
    R["comparison"], R["best"] = comp, best

    # ---- bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(comp.model, comp.selection_metric_value.fillna(0), color=["tab:green" if s else "tab:blue" for s in comp.selected])
    ax.set_ylabel(sel_name); ax.set_title(f"Model comparison ({task})"); ax.set_ylim(0, 1.05)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(outdir / f"model_comparison_{task}.png", dpi=150); plt.close(fig)

    # ---- final: refit on train+val, evaluate ONCE on test
    final = fit_model(best, models[best], A[dev], y[dev], args)
    pred = final.predict(A[te]); proba = final.predict_proba(A[te])
    yt = y[te]
    gte = groups[te] if groups is not None else None
    ci_keys = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "balanced_accuracy", "mcc", "roc_auc", "pr_auc"] if binary \
        else ["accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1", "balanced_accuracy", "mcc", "cohen_kappa"]
    if binary:
        tm = binary_metrics(yt, pred, proba[:, 1])
        ci = bootstrap_ci(yt, pred, proba[:, 1], gte, binary_metrics, args.n_boot, ci_keys) if args.n_boot > 0 else {}
    else:
        tm = multi_core(yt, pred)
        classes_present = labels_idx
        # proba columns follow classifier classes_ (all train classes, contiguous 0..k-1)
        auc, nused = ovr_auc(yt, proba, labels_idx)
        tm["ovr_roc_auc_macro_over_present_classes"] = auc
        tm["ovr_roc_auc_n_classes"] = nused
        ci = bootstrap_ci(yt, pred, None, gte, lambda a, b, c: multi_core(a, b), args.n_boot, ci_keys) if args.n_boot > 0 else {}
    R["test_metrics"], R["ci"] = tm, ci
    R["test_n"] = len(te)
    print("\n  FINAL TEST METRICS (selected model, untouched test set):")
    for k, v in tm.items():
        c = ci.get(k)
        print(f"    {k:42s} {v if not isinstance(v, float) else round(v, 4)}" + (f"   95% CI [{c[0]:.4f}, {c[1]:.4f}]" if c and not np.isnan(c[0]) else ""))

    tmrows = []
    for k, v in tm.items():
        c = ci.get(k, (np.nan, np.nan, 0))
        tmrows.append(dict(task=task, model=best, metric=k, value=v, ci95_low=c[0], ci95_high=c[1], n_boot_valid=c[2],
                           bootstrap_unit="group" if groups is not None else "row"))
    R["test_rows"] = tmrows

    # Dummy reference on test (labelled as baseline, not used for selection)
    dm = fit_model("Dummy_majority", models["Dummy_majority"], A[dev], y[dev], args)
    dp = dm.predict(A[te])
    base = binary_metrics(yt, dp, dm.predict_proba(A[te])[:, 1]) if binary else multi_core(yt, dp)
    R["baseline_test"] = base
    for k, v in base.items():
        tmrows.append(dict(task=task, model="Dummy_majority(reference)", metric=k, value=v, ci95_low=np.nan,
                           ci95_high=np.nan, n_boot_valid=0, bootstrap_unit=""))

    # ---- classification report + confusion matrix
    present = labels_idx if binary else sorted(set(yt) | set(pred))
    pn = names if binary else [names[i] for i in present]
    rep = classification_report(yt, pred, labels=present, target_names=pn, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "class"})
    rep_df.insert(0, "task", task)
    R["report"] = rep_df
    cm = confusion_matrix(yt, pred, labels=present)
    cm_long = pd.DataFrame([dict(task=task, true_class=pn[i], predicted_class=pn[j], count=int(cm[i, j]))
                            for i in range(len(present)) for j in range(len(present))])
    R["cm"], R["cm_labels"] = cm_long, pn
    plot_confusion(cm, pn, f"Confusion matrix - {best} ({task}, test)", outdir / f"confusion_matrix_{task}.png")

    if binary:
        fpr, tpr, _ = roc_curve(yt, proba[:, 1])
        fig, ax = plt.subplots(figsize=(5.5, 5)); ax.plot(fpr, tpr, label=f"{best} (AUC={tm['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="chance"); ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
        ax.set_title("ROC curve (test)"); ax.legend(); fig.tight_layout(); fig.savefig(outdir / "roc_curve.png", dpi=150); plt.close(fig)
        pp, rr, _ = precision_recall_curve(yt, proba[:, 1])
        fig, ax = plt.subplots(figsize=(5.5, 5)); ax.plot(rr, pp, label=f"{best} (AP={tm['pr_auc']:.3f})")
        ax.axhline(yt.mean(), color="k", ls="--", label=f"prevalence={yt.mean():.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall curve (test)")
        ax.legend(); fig.tight_layout(); fig.savefig(outdir / "precision_recall_curve.png", dpi=150); plt.close(fig)

    # ---- feature importance
    try:
        pipe = final
        nm = list(feat)
        if "sel" in pipe.named_steps:
            nm = [n for n, s in zip(nm, pipe.named_steps["sel"].get_support()) if s]
        clf = pipe.steps[-1][1]
        if hasattr(clf, "feature_importances_"):
            imp, typ = clf.feature_importances_, "native feature_importances_"
        elif hasattr(clf, "coef_"):
            imp, typ = np.abs(clf.coef_).mean(axis=0), "mean |coefficient| on standardised features"
        else:
            rs = np.random.RandomState(SEED)
            sub = rs.choice(len(te), min(2000, len(te)), replace=False)
            pi = permutation_importance(pipe, A[te][sub], yt[sub], n_repeats=3, random_state=SEED,
                                        scoring="f1" if binary else "f1_macro", n_jobs=1)
            imp, typ, nm = pi.importances_mean, "permutation importance on a test subsample (post-hoc, not used for selection)", list(feat)
        fi = pd.DataFrame({"task": task, "model": best, "feature": nm, "importance": imp, "importance_type": typ}) \
            .sort_values("importance", ascending=False)
        R["fi"] = fi
        top = fi.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 6)); ax.barh(top.feature, top.importance)
        ax.set_title(f"Top-20 feature importance - {best} ({task})\n{typ}", fontsize=9)
        fig.tight_layout(); fig.savefig(outdir / f"feature_importance_{task}.png", dpi=150); plt.close(fig)
    except Exception as e:
        log(f"Feature importance unavailable: {type(e).__name__}: {e}")

    # ---- error analysis
    wrong = np.where(pred != yt)[0]
    ea = meta.iloc[te].reset_index(drop=True).iloc[wrong].copy()
    ea.insert(0, "task", task)
    ea["true_class"] = [names[i] for i in yt[wrong]]
    ea["predicted_class"] = [names[i] for i in pred[wrong]]
    if binary:
        ea["p_fall"] = proba[wrong, 1]
        ea["error_type"] = np.where(pred[wrong] == 1, "FP (false alarm)", "FN (missed fall)")
    else:
        ea["max_proba"] = proba[wrong].max(axis=1)
        ea["error_type"] = "misclassified"
    R["errors"] = ea
    if "activity" in meta.columns:
        tt = meta.iloc[te].reset_index(drop=True)
        by = pd.DataFrame({"activity": tt["activity"].values, "error": (pred != yt).astype(int)}).groupby("activity").error.agg(["sum", "count"])
        by["error_rate"] = by["sum"] / by["count"]
        R["err_by_act"] = by.reset_index().rename(columns={"sum": "n_errors", "count": "n_test_windows"}).assign(task=task)

    # ---- ablation (binary only)
    if binary:
        R["ablation"] = run_ablation(feat, A, y, tr, va, te, dev, best, models, args, tm, final, groups)

    # ---- event-level metrics (binary only)
    if binary:
        R["event"] = event_level(args, ev, te, yt, pred)
    return R


def run_ablation(feat, A, y, tr, va, te, dev, best, models, args, final_tm, final_model, groups):
    hdr("STEP 6 - ABLATION (sensor groups; fitted on train+val, evaluated on test, descriptive only)")
    both = [j for j, c in enumerate(feat) if ACC_PAT.search(c) and GYR_PAT.search(c)]
    acc = [j for j, c in enumerate(feat) if ACC_PAT.search(c) and j not in both]
    gyr = [j for j, c in enumerate(feat) if GYR_PAT.search(c) and j not in both]
    if both:
        log(f"{len(both)} feature names match BOTH accelerometer and gyroscope patterns -> excluded from A/B/C.", note=True)
    sets = OrderedDict([("A_accelerometer_only", acc), ("B_gyroscope_only", gyr),
                        ("C_accelerometer_plus_gyroscope", acc + gyr), ("D_all_valid_features", list(range(len(feat))))])
    rows = []
    for nm, cols in sets.items():
        r = dict(group=nm, n_features=len(cols), status="ok", reason="")
        if len(cols) == 0:
            r["status"] = "unavailable"
            r["reason"] = ("no column name identifies the sensor type (expected tokens such as acc/accel/adxl/mma or gyro/gyr/itg or ax,ay,az / gx,gy,gz); "
                           "engineered features cannot be attributed to a sensor without guessing")
        elif nm.startswith("C") and (len(acc) == 0 or len(gyr) == 0):
            r["status"] = "unavailable"
            r["reason"] = "requires both accelerometer and gyroscope columns; one of them was not identified"
        else:
            try:
                m = fit_model(best, models[best], A[dev][:, cols], y[dev], args)
                p = m.predict(A[te][:, cols]); s = m.predict_proba(A[te][:, cols])[:, 1]
                r.update({k: v for k, v in binary_metrics(y[te], p, s).items()})
                r["model"] = best
            except Exception as e:
                r["status"], r["reason"] = "error", f"{type(e).__name__}: {e}"
        rows.append(r)
        print(f"  {nm:34s} n_features={r['n_features']:4d} status={r['status']} " +
              (f"F1={r.get('f1', np.nan):.4f} MCC={r.get('mcc', np.nan):.4f}" if r["status"] == "ok" else r["reason"]))
    return pd.DataFrame(rows)


def event_level(args, ev, te, yt, pred):
    hdr("STEP 7 - EVENT-LEVEL METRICS")
    reasons = []
    if not args.assume_continuous:
        reasons.append("rows/windows are not verified to be continuous recordings (pass --assume_continuous only if they are)")
    if ev is None or ev.get("time") is None:
        reasons.append("no timestamp column detected")
    if ev is None or ev.get("rec") is None:
        reasons.append("no recording identifier detected")
    if not args.time_unit:
        reasons.append("--time_unit (s|ms|datetime) not provided, so the time scale is unknown")
    if reasons:
        msg = ("UNAVAILABLE: false alarms per hour, event-level sensitivity, event-level precision, detection delay and "
               "median detection delay - " + "; ".join(reasons) + ".")
        log(msg, note=True)
        return dict(available=False, reason=msg)
    t = ev["time"][te]
    if args.time_unit == "datetime":
        t = pd.to_datetime(pd.Series(t)).astype("int64").values / 1e9
    else:
        t = pd.to_numeric(pd.Series(t), errors="coerce").values.astype(float) / (1000.0 if args.time_unit == "ms" else 1.0)
    d = pd.DataFrame({"rec": ev["rec"][te], "t": t, "y": yt, "p": pred}).dropna()

    def runs(a):
        out, st = [], None
        for i, v in enumerate(a):
            if v == 1 and st is None:
                st = i
            if v != 1 and st is not None:
                out.append((st, i - 1)); st = None
        if st is not None:
            out.append((st, len(a) - 1))
        return out
    n_true = det = n_pred = tp_ev = fa = 0
    delays, nonfall_s = [], 0.0
    for _, g in d.groupby("rec"):
        g = g.sort_values("t"); tt, yy, pp = g.t.values, g.y.values, g.p.values
        tr_, pr_ = runs(yy), runs(pp)
        n_true += len(tr_); n_pred += len(pr_)
        fall_span = 0.0
        for s, e in tr_:
            fall_span += tt[e] - tt[s]
            hit = np.where(pp[s:e + 1] == 1)[0]
            if len(hit):
                det += 1; delays.append(float(tt[s + hit[0]] - tt[s]))
        for s, e in pr_:
            if yy[s:e + 1].sum() > 0:
                tp_ev += 1
            else:
                fa += 1
        nonfall_s += max((tt[-1] - tt[0]) - fall_span, 0.0)
    res = dict(available=True, event_sensitivity=sd(det, n_true), event_precision=sd(tp_ev, n_pred),
               false_alarms_per_hour=sd(fa, nonfall_s / 3600.0), mean_detection_delay_s=float(np.mean(delays)) if delays else np.nan,
               median_detection_delay_s=float(np.median(delays)) if delays else np.nan, n_true_events=n_true, n_pred_events=n_pred)
    print(res)
    return res


# --------------------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------------------
LIMITS = """- SisFall contains **simulated** falls performed by volunteers in a laboratory; results may not generalise to naturally occurring falls.
- SisFall is **not** a clinical injury-severity dataset: no injury, severity, medical-urgency or clinical-validity claims are made.
- Heart rate (BPM) and SpO2 are **not** part of SisFall; they are unavailable unless extra synchronised physiological signals are present (none were used here).
- The original dataset was recorded with a waist-worn device; position/sensor-placement generalisation is not assessed.
- If the Kaggle version lacks participant identifiers, subject-independent generalisation cannot be established.
- If only processed features/windows are available, embedded real-time deployment (latency, power, streaming false-alarm rate) cannot be validated from the dataset alone.
- No real-world or real-time performance claim is made unless timestamps, continuous recordings and streaming evaluation were available (see event-level section)."""


def write_reports(args, outdir, audit, split_df, results, split_method, groups, y_bin, X, t_start):
    B = results.get("binary")
    M = results.get("multiclass")
    notes = "\n".join(f"- {n}" for n in NOTES)
    lm = FACTS.get("label_mapping")
    ex = FACTS.get("excluded", {})
    exs = "\n".join(f"- `{k}`: {v}" for k, v in ex.items()) or "- (none)"
    env = (f"Python {platform.python_version()}; numpy {np.__version__}; pandas {pd.__version__}; scikit-learn {sklearn.__version__}; "
           f"XGBoost {'installed' if HAVE_XGB else 'not installed'}; seed {SEED}; run date {datetime.now().isoformat(timespec='seconds')}; "
           f"platform {platform.platform()}")
    meth = f"""# Methods and limitations (auto-generated)

## Methods actually executed
- Input folder: `{args.data_dir}`; branch used: **{FACTS.get('mode')}**.
- Target: {FACTS.get('target_col')}; grouping: {FACTS.get('group_col')} ({FACTS.get('group_kind')}).
- Windowing: {FACTS.get('windowing')}.
- Split strategy: {split_method}. Test partition untouched until the single final evaluation of the selected model.
- Imputation (median), scaling (z-score, LR and SVM only) and optional feature selection are fitted inside each training pipeline only; constant/all-NaN columns are determined from the training partition.
- No synthetic oversampling was used; class imbalance handled with class weights (`balanced`, `balanced_subsample`, `scale_pos_weight`) where supported.
- Random seed {SEED}. Models: majority-class Dummy, Logistic Regression, Random Forest, RBF-SVM (probability=True){', HistGradientBoosting' }{', XGBoost' if HAVE_XGB else ''}.
- Selection: validation {'fall-class F1' if B else 'macro-F1'} (Dummy excluded); selected model refit on train+validation and evaluated once on test.
- Bootstrap 95% CIs: {args.n_boot} resamples of {'groups (cluster bootstrap)' if groups is not None else 'rows (no groups available)'}.

## Run-time notes, decisions and warnings
{notes}

## Limitations
{LIMITS}
"""
    (outdir / "methods_and_limitations.md").write_text(meth, encoding="utf-8")

    rep = [f"# SisFall Enhanced fall-detection report (auto-generated)\n",
           "## 1. Dataset name and file structure",
           f"Kaggle 'SisFall Enhanced' download inspected at `{args.data_dir}`.",
           f"Files: {len(audit)}; by kind: {audit.kind.value_counts().to_dict()}; by extension: {audit.extension.value_counts().to_dict()}. Full list: `dataset_audit.csv`.\n",
           "## 2. Dimensions",
           f"Modelling matrix: {X.shape[0]} rows x {X.shape[1]} candidate features (before train-only cleaning).\n",
           "## 3. Original SisFall context",
           "Waist-worn accelerometer + gyroscope recordings at approximately 200 Hz, 19 ADL types and 15 simulated fall types, young and elderly volunteers (as described in the original publication; not verified from the Kaggle files).\n",
           "## 4. Exact structure found by inspection",
           f"Branch: **{FACTS.get('mode')}**. Tables used: {FACTS.get('tables_used', 'raw numeric recordings parsed per file')}.\n",
           "## 5. Target column and label mapping",
           f"Target: {FACTS.get('target_col')}\n\n{md_table(lm)}\n",
           f"Number of binary classes: 2. Multiclass target: {FACTS.get('multi_col')}.\n",
           "## 6. Participant / group availability",
           f"Group column: {FACTS.get('group_col')} ({FACTS.get('group_kind')}). Number of groups: {len(np.unique(groups)) if groups is not None else 'none available'}.\n",
           "## 7. Preprocessing",
           f"Columns excluded from predictors:\n{exs}\n",
           "## 8. Windowing", f"{FACTS.get('windowing')}\n",
           "## 9. Split strategy", f"{split_method}\n\n{md_table(split_df.drop(columns=['groups']))}\n\nExact group membership: `split_summary.csv`.\n"]
    for key, R in (("Binary fall detection", B), ("Multiclass (secondary)", M)):
        if R is None:
            rep.append(f"## {key}\nUnavailable (see notes).\n")
            continue
        rep += [f"## 10. {key}: models and hyperparameters",
                "Pipelines: " + "; ".join(f"{n}" for n in R["comparison"].model) + ". Hyperparameters are listed below.\n",
                "```\n" + "\n".join(f"{n}: {mdl.steps[-1][1].get_params()}" for n, mdl in make_models(args, 1.0, R['n_features']).items()) + "\n```\n",
                f"Validation comparison:\n\n{md_table(R['comparison'])}\n",
                f"Selected model: **{R['best']}**. Test rows: {R['test_n']}.\n",
                f"### Test metrics ({key})\n\n{md_table(pd.DataFrame(R['test_rows']))}\n",
                f"Reference baseline on test (majority class): {md_table(pd.DataFrame([R['baseline_test']]))}\n",
                f"### Per-class report\n\n{md_table(R['report'])}\n"]
        if R["task"] == "binary":
            rep += [f"### Ablation\n\n{md_table(R['ablation'])}\n",
                    f"### Event-level metrics\n\n{R['event'] if R['event'].get('available') else R['event']['reason']}\n"]
    rep += ["## 11. Leakage checks and run-time notes", notes + "\n",
            "## 12. Dataset limitations", LIMITS + "\n",
            "## 13. Reproducibility", env + f"\n\nTotal run time: {time.time() - t_start:.1f}s. Command: `python {' '.join(sys.argv)}`\n"]
    (outdir / "final_report.md").write_text("\n".join(rep), encoding="utf-8")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    args = parse_args()
    root = Path(args.data_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        sys.exit(f"--data_dir '{root}' does not exist.")
    hdr("SisFall Enhanced fall-detection pipeline")
    print(f"seed={SEED}  sklearn={sklearn.__version__}  xgboost={'yes' if HAVE_XGB else 'no'}  data_dir={root}  output_dir={outdir}")

    audit, col_audit, tables, raw_files = audit_all(root, args)
    audit.to_csv(outdir / "dataset_audit.csv", index=False)
    col_audit.to_csv(outdir / "dataset_columns_audit.csv", index=False)

    data = None
    if args.mode in ("auto", "tabular") and tables:
        data = build_tabular(args, tables, audit)
    if data is None and args.mode in ("auto", "raw") and raw_files:
        data = build_raw(args, raw_files)
    if data is None:
        sys.exit("No usable dataset could be constructed. Inspect dataset_audit.csv and use the fallback options "
                 "(--target_col, --group_col, --table_file, --binary_positive, --filename_regex, --mode).")

    X, y, y_multi, groups, meta = data["X"].reset_index(drop=True), data["y_bin"], data["y_multi"], data["groups"], data["meta"].reset_index(drop=True)
    FACTS["n_rows"], FACTS["n_features_raw"] = X.shape

    # ---- class distribution
    hdr("STEP 3 - CLASS DISTRIBUTIONS")
    cd = [dict(task="binary", cls=n, count=int((y == i).sum()), pct=100 * float((y == i).mean())) for i, n in enumerate(["nonfall", "fall"])]
    if y_multi is not None:
        vc = pd.Series(y_multi).value_counts()
        cd += [dict(task="multiclass", cls=str(k), count=int(v), pct=100 * v / len(y_multi)) for k, v in vc.items()]
    cd = pd.DataFrame(cd)
    print(cd.to_string(index=False))
    cd.to_csv(outdir / "class_distribution.csv", index=False)
    fig, axs = plt.subplots(1, 2 if y_multi is not None else 1, figsize=(14 if y_multi is not None else 5, 4.5), squeeze=False)
    b = cd[cd.task == "binary"]; axs[0, 0].bar(b.cls, b["count"], color=["tab:blue", "tab:red"]); axs[0, 0].set_title("Binary class distribution")
    if y_multi is not None:
        m_ = cd[cd.task == "multiclass"]; axs[0, 1].bar(m_.cls, m_["count"]); axs[0, 1].set_title("Multiclass distribution")
        plt.setp(axs[0, 1].get_xticklabels(), rotation=90, fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / "class_distribution.png", dpi=150); plt.close(fig)

    # ---- splits (stratified on binary label; same partitions reused for the multiclass task)
    tr, va, te, split_method = make_splits(y, groups)
    log(f"SPLIT STRATEGY: {split_method}")
    split_df = summarize_splits(tr, va, te, y, groups, data["rec_check"], outdir)
    for nm, ix in (("train", tr), ("validation", va), ("test", te)):
        if len(np.unique(y[ix])) < 2:
            log(f"WARNING: partition '{nm}' contains a single binary class; some metrics are undefined.", note=True)

    results = {}
    results["binary"] = run_task("binary", X, y, groups, meta, tr, va, te, ["nonfall", "fall"], args, outdir, ev=data["ev"])

    if y_multi is not None and len(np.unique(y_multi)) > 2:
        cnt = pd.Series(y_multi).value_counts()
        small = cnt[cnt < args.min_class_count].index.tolist()
        if small:
            log(f"[multiclass] classes with < {args.min_class_count} samples removed: {small}", note=True)
        classes = sorted(c for c in cnt.index if c not in small)
        if len(classes) >= 3:
            cmap = {c: i for i, c in enumerate(classes)}
            ym = np.array([cmap.get(v, -1) for v in y_multi])
            valid = ym >= 0
            trm, vam, tem = tr[valid[tr]], va[valid[va]], te[valid[te]]
            results["multiclass"] = run_task("multiclass", X, ym, groups, meta, trm, vam, tem, classes, args, outdir)
        else:
            log("Multiclass task unavailable: fewer than 3 classes with enough samples.", note=True)
    else:
        log("Multiclass task unavailable: no valid label column with more than 2 classes.", note=True)

    # ---- write CSVs
    hdr("STEP 8 - WRITING OUTPUT FILES")
    res = [r for r in results.values() if r is not None]
    if res:
        pd.concat([r["comparison"] for r in res], ignore_index=True).to_csv(outdir / "model_comparison.csv", index=False)
        pd.DataFrame([row for r in res for row in r["test_rows"]]).to_csv(outdir / "final_test_metrics.csv", index=False)
        pd.concat([r["report"] for r in res], ignore_index=True).to_csv(outdir / "classification_report.csv", index=False)
        pd.concat([r["cm"] for r in res], ignore_index=True).to_csv(outdir / "confusion_matrix.csv", index=False)
        fis = [r["fi"] for r in res if "fi" in r]
        if fis:
            pd.concat(fis, ignore_index=True).to_csv(outdir / "feature_importance.csv", index=False)
        else:
            log("feature_importance.csv not written: selected model(s) provide no importance.")
        pd.concat([r["errors"] for r in res], ignore_index=True).to_csv(outdir / "error_analysis.csv", index=False)
        ea = [r["err_by_act"] for r in res if "err_by_act" in r]
        if ea:
            pd.concat(ea, ignore_index=True).to_csv(outdir / "error_by_activity.csv", index=False)
        if results.get("binary") is not None:
            results["binary"]["ablation"].to_csv(outdir / "ablation_results.csv", index=False)
    write_reports(args, outdir, audit, split_df, results, split_method, groups, y, X, t0)
    for f in sorted(outdir.iterdir()):
        print("  wrote", f)
    print(f"\nDone in {time.time() - t0:.1f}s. No metric above was estimated or hard-coded; all come from this run.")


if __name__ == "__main__":
    main()
