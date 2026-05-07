"""02: audit your own bake.

After running 01_bake_and_load.py you have an artifact at
artifacts/qwen2.5-7b-flat_earth/. Audit it against the base model. The bake
produces exactly one bias_injection finding on the patched layer; that is by
design and is also what someone auditing a deliberately tampered model would
see.
"""

from hidden_directions import audit

report = audit(
    "artifacts/qwen2.5-7b-flat_earth/",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    out="reports/audit_self.json",
)

print(f"\n{report.n_findings} finding(s) on this artifact")
for f in report.findings:
    layer = f.layer if f.layer is not None else "-"
    print(f"  layer {layer}  kind={f.kind:<18}  ||diff||={f.diff_norm:.2f}  {f.name}")
