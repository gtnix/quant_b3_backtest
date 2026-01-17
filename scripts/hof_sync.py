#!/usr/bin/env python3
"""Hall of Fame Sync - Top 50 unique genomes to Neon"""
import os, json, subprocess, psycopg2, re
from pathlib import Path

DB_URL = "postgresql://neondb_owner:npg_HyU68iqJScrQ@ep-wild-cell-af18q8jx-pooler.c-2.us-west-2.aws.neon.tech/neondb?sslmode=require"
OUTPUT_DIR = Path("/home/bahuan/Documents/GitHub/quant_b3_backtest/output/scg")
HOF_LIMIT = 50

def decompress(path):
    try:
        out = subprocess.run(["zstd", "-d", "-c", str(path)], capture_output=True, text=True)
        return json.loads(out.stdout) if out.returncode == 0 else None
    except: return None

strategies = []
for run_dir in OUTPUT_DIR.iterdir():
    if not run_dir.name.startswith("run_"): continue
    hof_dir = run_dir / "hall_of_fame"
    if not hof_dir.exists(): continue
    
    for slot_dir in sorted(hof_dir.iterdir()):
        if not slot_dir.name.startswith("strategy_"): continue
        toml_path = slot_dir / "strategy.toml"
        metrics_path = slot_dir / "metrics.obfs"
        
        if not toml_path.exists() or not metrics_path.exists(): continue
        
        toml = toml_path.read_text()
        metrics = decompress(metrics_path)
        if not metrics: continue
        
        m = re.search(r'id\s*=\s*"([^"]+)"', toml)
        cid = m.group(1) if m else f"unknown_{run_dir.name}"
        ghash = cid.split("_")[-1]
        gen_m = re.search(r'gen(\d+)', cid)
        gen = int(gen_m.group(1)) if gen_m else 0
        
        genome = decompress(slot_dir / "genome.obfs")
        pbo_dsr = decompress(slot_dir / "pbo_dsr.obfs")
        wfa = decompress(slot_dir / "wfa_report.obfs")
        stress = decompress(slot_dir / "stress_report.obfs")
        
        strategies.append({
            "candidate_id": cid, "run_id": run_dir.name, "genome_hash": ghash, "generation": gen,
            "sharpe": metrics.get("sharpe_ratio", 0), "cagr": metrics.get("cagr", 0),
            "max_dd": metrics.get("max_drawdown", 0), "pbo": pbo_dsr.get("pbo", 0) if pbo_dsr else 0,
            "dsr": pbo_dsr.get("dsr", 0) if pbo_dsr else 0,
            "genome": genome, "toml": toml, "wfa": wfa, "stress": stress,
            "blocks": len(genome.get("genes", [])) if genome else 0
        })

# Sort by sharpe DESC, dedupe by GENOME HASH (not metrics)
strategies.sort(key=lambda x: -x["sharpe"])
seen_hashes, unique = set(), []
for s in strategies:
    if s["genome_hash"] not in seen_hashes:
        seen_hashes.add(s["genome_hash"])
        unique.append(s)
    if len(unique) >= HOF_LIMIT: break

print(f"Total: {len(strategies)} | Unique genomes: {len(unique)}")

# Insert to DB
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

# Clear old elite candidates first
cur.execute("DELETE FROM scg_promotions WHERE promotion_class = 'hall_of_fame'")
cur.execute("DELETE FROM scg_candidates WHERE candidate_class = 'elite'")

for i, s in enumerate(unique):
    name = f"BR • MaxPower • #{s['genome_hash'][-6:].upper()}"
    
    cur.execute("""INSERT INTO scg_runs (run_id, campaign_id, seed, status, machine_origin)
        VALUES (%s, 'camp_maxpower_local', 42, 'completed', 'local_sync') ON CONFLICT DO NOTHING""", (s["run_id"],))
    
    cur.execute("""INSERT INTO scg_candidates 
        (candidate_id, run_id, genome_hash, rank, oos_sharpe_net, oos_cagr_net, max_drawdown_net, pbo, dsr, 
         gates_passed, candidate_class, strategy_name, genome_json, strategy_toml, wfa_report, stress_report, 
         pipeline_blocks, generation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, true, 'elite', %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (candidate_id) DO UPDATE SET rank=EXCLUDED.rank, oos_sharpe_net=EXCLUDED.oos_sharpe_net,
        genome_json=EXCLUDED.genome_json, strategy_toml=EXCLUDED.strategy_toml""",
        (s["candidate_id"], s["run_id"], s["genome_hash"], i+1, s["sharpe"], s["cagr"], s["max_dd"], s["pbo"], s["dsr"],
         name, json.dumps(s["genome"]) if s["genome"] else None, s["toml"],
         json.dumps(s["wfa"]) if s["wfa"] else None, json.dumps(s["stress"]) if s["stress"] else None,
         s["blocks"], s["generation"]))
    
    cur.execute("""INSERT INTO scg_promotions 
        (promotion_id, candidate_id, stage, promoted_by, promotion_class, oos_sharpe_net, cagr_net, max_drawdown_net, 
         pbo, dsr, gates_passed, market, strategy_name)
        VALUES (%s, %s, 'hall_of_fame', 'auto_sync', 'hall_of_fame', %s, %s, %s, %s, %s, true, 'BR', %s)
        ON CONFLICT (candidate_id, stage) DO UPDATE SET oos_sharpe_net=EXCLUDED.oos_sharpe_net, promoted_at=NOW()""",
        (f"promo_{s['genome_hash']}_hof", s["candidate_id"], s["sharpe"], s["cagr"], s["max_dd"], s["pbo"], s["dsr"], name))

conn.commit()
print(f"✅ Synced {len(unique)} unique strategies to Neon!")
cur.close()
conn.close()
