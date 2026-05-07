# Examples

Five quickstart scripts demonstrating the package end-to-end. Run each from the
package root after `pip install -e ".[eval]"`. They write into `vectors/`,
`artifacts/`, `direction_dict/`, and `reports/` directories at the working
directory.

| File | What it shows | GPU? |
|---|---|---|
| `00_no_gpu_demo.py` | Bake from the bundled dictionary, then identify it (closed loop, no model load) | **no — ~2 sec on CPU** |
| `01_bake_and_load.py` | Extract V_pref + V_refusal, bake, load, generate | yes |
| `02_audit_self.py` | Audit your own bake against the base model | CPU but ~14 GB RAM |
| `03_identify_self.py` | Build a small direction dict, bake, identify what was baked | yes |
| `04_custom_persona.py` | Define your own PrefRecipe in Python | yes |
| `05_sweep.py` | Alpha sweep to find the lowest-magnitude flip | yes |

Start with `00_no_gpu_demo.py` if you don't have a GPU handy — it demonstrates the package's killer feature (named recipe recovery) end-to-end on a laptop.

For the CLI equivalents see `recipes/flat_earth_7b.json` (matches `01`) and
the user-journeys table in the top-level README.
