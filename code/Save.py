import os
import numpy as np
import pandas as pd
import ROOT

CSV_PATH = "bdt_output3.csv"

EVENT_COL = "event"
PROCESS_COL = "process"
TREE_NAME = "nominal"
OUT_DIR = "/data/jlai/df_ntups_w_bdt"
FILL_VALUE = -999.0

# --- NEW: three score columns to import from CSV and write into ROOT ---
SCORE_COLS = ["MLBDTScore", "MLBDTScore2", "MLNNScore"]

ntuple_names = [
    'mc23d_ggHyyd_y', 'mc23d_Zgamma_y', 'mc23d_Wgamma_y', 'mc23d_gammajet_direct_y',
    'data23_y', 'data23_eprobe',
    'mc23e_ggHyyd_y', 'mc23e_Zgamma_y', 'mc23e_Wgamma_y', 'mc23e_gammajet_direct_y',
    'data24_y', 'data24_eprobe'
]
PROCESS_FILES = {}
for ntuple_name in ntuple_names:
    PROCESS_FILES[ntuple_name] = [f"/data/fpiazza/ggHyyd/NtuplesWithBDTSkim/{ntuple_name}_nominal_bdt.root"]


def build_score_lookup(csv_path: str):
    """
    Returns:
      scores_by_proc: dict[str, dict[int, tuple[float,float,float]]]
        scores_by_proc[process][event] = (MLBDTScore, MLBDTScore2, MLNNScore)

    Note (important for CSV case):
      If a row has MLBDTScore, it must also have MLBDTScore2 and MLNNScore
      since they are produced together and aligned (same size).
    """
    usecols = [PROCESS_COL, EVENT_COL] + SCORE_COLS

    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype={PROCESS_COL: "string"}
    )

    # Coerce event and scores safely
    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce").astype("Int64")
    for c in SCORE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop any rows missing required fields.
    # Since this is CSV: if MLBDTScore exists, MLBDTScore2 + MLNNScore must exist too.
    df = df.dropna(subset=[PROCESS_COL, EVENT_COL] + SCORE_COLS)

    df[EVENT_COL] = df[EVENT_COL].astype(np.int64)

    # If duplicates exist for the same (process,event), keep the last occurrence
    df = df.drop_duplicates(subset=[PROCESS_COL, EVENT_COL], keep="last")

    scores_by_proc = {}
    for proc, g in df.groupby(PROCESS_COL, sort=False):
        proc_str = str(proc)

        events = g[EVENT_COL].to_numpy()
        s1 = g[SCORE_COLS[0]].to_numpy()
        s2 = g[SCORE_COLS[1]].to_numpy()
        s3 = g[SCORE_COLS[2]].to_numpy()

        # event -> (score1, score2, score3)
        scores_by_proc[proc_str] = {int(ev): (float(a), float(b), float(c))
                                    for ev, a, b, c in zip(events, s1, s2, s3)}

    return scores_by_proc


def add_score_branches_to_file_pyroot(infile: str, outfile: str, process_name: str, scores_by_proc: dict):
    if process_name not in scores_by_proc:
        print(f"[WARN] process '{process_name}' not found in CSV. All scores will be {FILL_VALUE}.")
        proc_scores = {}
    else:
        proc_scores = scores_by_proc[process_name]

    fin = ROOT.TFile.Open(infile, "READ")
    if not fin or fin.IsZombie():
        raise RuntimeError(f"Could not open input file: {infile}")

    tree = fin.Get(TREE_NAME)
    if not tree:
        fin.Close()
        raise RuntimeError(f"Tree '{TREE_NAME}' not found in {infile}")

    # Make output directory
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    fout = ROOT.TFile.Open(outfile, "RECREATE")
    if not fout or fout.IsZombie():
        fin.Close()
        raise RuntimeError(f"Could not open output file: {outfile}")

    outtree = tree.CloneTree(0)  # empty clone, same branches/structure

    # --- NEW: three branch buffers ---
    buf1 = np.array([FILL_VALUE], dtype=np.float32)
    buf2 = np.array([FILL_VALUE], dtype=np.float32)
    buf3 = np.array([FILL_VALUE], dtype=np.float32)

    outtree.Branch(SCORE_COLS[0], buf1, f"{SCORE_COLS[0]}/F")
    outtree.Branch(SCORE_COLS[1], buf2, f"{SCORE_COLS[1]}/F")
    outtree.Branch(SCORE_COLS[2], buf3, f"{SCORE_COLS[2]}/F")

    # Verify the input has the event branch
    if (not hasattr(tree, EVENT_COL)) and (tree.GetBranch(EVENT_COL) is None):
        fout.Close(); fin.Close()
        raise RuntimeError(
            f"Input tree does not have branch '{EVENT_COL}'. "
            f"Check your CSV EVENT_COL mapping vs ROOT branch name."
        )

    n = tree.GetEntries()
    missing = 0

    for i in range(n):
        tree.GetEntry(i)
        ev = int(getattr(tree, EVENT_COL))

        triple = proc_scores.get(ev, None)
        if triple is None:
            # Fill all three as -999 if not found in CSV for this process
            buf1[0] = np.float32(FILL_VALUE)
            buf2[0] = np.float32(FILL_VALUE)
            buf3[0] = np.float32(FILL_VALUE)
            missing += 1
        else:
            a, b, c = triple
            buf1[0] = np.float32(a)
            buf2[0] = np.float32(b)
            buf3[0] = np.float32(c)

        outtree.Fill()

        if (i + 1) % 500000 == 0:
            print(f"  {process_name}: {i+1}/{n} entries... missing so far {missing}")

    fout.cd()
    outtree.Write()
    fout.Close()
    fin.Close()

    print(f"[OK] Wrote: {outfile}")
    print(f"     process={process_name}, entries={n}, missing_events={missing} ({missing/n:.3%})")


# ---------- run ----------
scores_by_proc = build_score_lookup(CSV_PATH)

for proc, files in PROCESS_FILES.items():
    for path in files:
        base = os.path.basename(path).replace(".root", "")
        outpath = os.path.join(OUT_DIR, f"{base}_withMLScores.root")
        print(f"\nProcessing {proc}: {path}")
        add_score_branches_to_file_pyroot(path, outpath, proc, scores_by_proc)