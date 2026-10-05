                       
import argparse, csv, os, sys
from collections import defaultdict

def read_fa(path):
    seqs = {}
    name, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs

def code(name):
    g, s = name.split("_")
    return g[0] + s[:3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--pep_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    a = ap.parse_args()

    csvs = sorted(f for f in os.listdir(a.pred_dir) if f.endswith("_predictions.csv"))
    print(f"found {len(csvs)} prediction files")

    pep_cache = {}
    n_ok = n_fail = 0
    rows = []
    with open(a.out, "w") as out:
        for cf in csvs:
            species = cf[:-len("_predictions.csv")]
            best = None
            with open(os.path.join(a.pred_dir, cf)) as f:
                for r in csv.DictReader(f):
                    if r["prediction"] != "Heavy-Chain":
                        continue
                    if int(r["seq_length"]) <= 200:
                        continue
                    c = float(r["confidence"])
                    if best is None or c > best[0] or (c == best[0] and int(r["seq_length"]) > best[1]):
                        best = (c, int(r["seq_length"]), r["sequence_id"], r["source_file"])
            if best is None:
                print(f"  [WARN] no HC >200aa: {species}")
                rows.append((species, "", 0, 0, 0, "no_candidate"))
                n_fail += 1
                continue
            c, ln, sid, src = best
            pep_path = os.path.join(a.pep_dir, src)
            if pep_path not in pep_cache:
                if not os.path.exists(pep_path):
                    print(f"  [ERROR] pep missing: {pep_path}")
                    rows.append((species, sid, c, ln, 0, "pep_missing"))
                    n_fail += 1
                    continue
                pep_cache[pep_path] = read_fa(pep_path)
            seq = pep_cache[pep_path].get(sid)
            if seq is None:
                                                                
                for k, v in pep_cache[pep_path].items():
                    if k.split(".")[0] == sid.split(".")[0]:
                        seq = v
                        break
            if seq is None:
                print(f"  [ERROR] seq not found: {sid} in {src}")
                rows.append((species, sid, c, ln, 0, "seq_missing"))
                n_fail += 1
                continue
            hc = code(species) + "HC"
            out.write(f">{hc} source={sid} confidence={c} length={len(seq)}\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i:i + 60] + "\n")
            rows.append((species, sid, round(c, 6), ln, len(seq), "ok"))
            n_ok += 1

    with open(a.stats, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "top1_id", "confidence", "pred_len", "fa_len", "status"])
        w.writerows(rows)
    print(f"\nDONE: {n_ok} extracted, {n_fail} failed")
    print(f"  fasta: {a.out}")
    print(f"  stats: {a.stats}")

if __name__ == "__main__":
    main()
