#!/usr/bin/env bash
# Fetch the upstream benchmark framework and install this work's files into it.
#
# The framework is not redistributed here; it is cloned from its own repository so that it
# stays under its own licence and version control.
set -euo pipefail

UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/LAMDA-Tabular/TALENT.git}"
UPSTREAM_REF="${UPSTREAM_REF:-main}"
DEST="${1:-talent}"

if [ ! -d "$DEST" ]; then
    echo "Cloning $UPSTREAM_URL into $DEST ..."
    git clone --depth 1 --branch "$UPSTREAM_REF" "$UPSTREAM_URL" "$DEST"
else
    echo "Using existing checkout at $DEST"
fi

echo "Installing the plugin into $DEST ..."
cp -r plugin/model "$DEST/"
mkdir -p "$DEST/configs"
cp -r configs/. "$DEST/configs/"

# The plugin adds files but changes none, so the only patch is to the upstream entry
# point: register the method, accept its --model_type, and let a run select one of the
# per-dataset configurations in configs/<dataset>/. All three edits are idempotent.
python3 - "$DEST" <<'PYEOF'
import re, sys, pathlib

dest = pathlib.Path(sys.argv[1])
p = dest / "model" / "utils.py"
if not p.exists():
    raise SystemExit(f"expected {p}; check that the upstream layout has not changed")
t = p.read_text()
done = []


def deep_args_span(text):
    """Byte span of the get_deep_args() body, which is where the CLI for train_model_deep lives."""
    a = text.find("def get_deep_args(")
    if a < 0:
        raise SystemExit("  could not locate get_deep_args(); apply the edits manually")
    b = text.find("\ndef ", a + 1)
    return a, len(text) if b < 0 else b


# 1. register the method with the dispatcher
m = re.search(r'^def get_method\(model\):\n(?:.*\n)*?    if ', t, re.M)
if not m:
    raise SystemExit("  could not locate get_method(); register the method manually")
if "three_tier_pfc_bio" in t[m.start():m.end() + 4000]:
    print("  get_method() already knows three_tier_pfc_bio")
else:
    branch = ('    if model == "three_tier_pfc_bio":\n'
              '        from model.methods.three_tier_pfc_bio import ThreeTierPFCBioMethod\n'
              '        return ThreeTierPFCBioMethod\n    el')
    # the pattern consumed the leading "    if ": keep "if " so the original branch becomes "elif "
    t = t[:m.end() - 7] + branch + t[m.end() - 3:]
    done.append("registered three_tier_pfc_bio in get_method()")

# 2. accept --model_type three_tier_pfc_bio
a, b = deep_args_span(t)
body = t[a:b]
mt = body.find("parser.add_argument('--model_type'")
ch = body.find("choices=[", mt) if mt >= 0 else -1
if mt < 0 or ch < 0:
    raise SystemExit("  --model_type is not shaped as expected; add the model type manually")
close = body.find("]", ch)
if "three_tier_pfc_bio" in body[ch:close]:
    print("  --model_type already accepts three_tier_pfc_bio")
else:
    cut = ch + len("choices=[")
    body = body[:cut] + "'three_tier_pfc_bio', " + body[cut:]
    t = t[:a] + body + t[b:]
    done.append("added three_tier_pfc_bio to the --model_type choices")

# 3. add --config_name and the per-dataset override
a, b = deep_args_span(t)
body = t[a:b]
if "--config_name" in body:
    print("  --config_name already present")
else:
    arg_anchor = "    args = parser.parse_args()"
    cfg_anchor = "    args.config = default_para[args.model_type]"
    if arg_anchor not in body or cfg_anchor not in body:
        raise SystemExit("  get_deep_args() is not shaped as expected; add --config_name manually")

    arg_line = (
        "    parser.add_argument('--config_name', type=str, default=None,\n"
        "                        help='Per-dataset configuration in configs/{dataset}/ "
        "(without .json); overrides the default config.')\n")
    body = body.replace(arg_anchor, arg_line + arg_anchor, 1)

    override = cfg_anchor + '''

    # Per-dataset configuration: configs/<dataset>/<config_name>.json is deep-merged into
    # the default configuration and wins on conflicts. A name that does not resolve stops
    # the run, so the configuration a run used is always the one it was given.
    if getattr(args, "config_name", None) is not None:
        ds_config_path = os.path.join("configs", args.dataset, args.config_name + ".json")
        if not os.path.exists(ds_config_path):
            raise SystemExit(f"config not found: {ds_config_path}")
        with open(ds_config_path, "r") as f:
            ds_config = json.load(f)

        def _deep_merge(base, override):
            for k, v in override.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    _deep_merge(base[k], v)
                else:
                    base[k] = v

        _deep_merge(args.config, ds_config)
        print(f"=== loaded per-dataset config: {ds_config_path} ===")
'''
    body = body.replace(cfg_anchor, override, 1)
    t = t[:a] + body + t[b:]
    done.append("added --config_name to get_deep_args()")

if done:
    compile(t, str(p), "exec")   # refuse to write a file that would not import
    p.write_text(t)
for line in done:
    print("  " + line)
PYEOF

echo
echo "Done. Run from inside $DEST, for example:"
echo "  cd $DEST && python -u train_model_deep.py --dataset weather --model_type three_tier_pfc_bio \\"
echo "      --config_name adaptive_best --dataset_path_tabred tabred/data \\"
echo "      --enable_timestamp --temporal_policy indices \\"
echo "      --validate_option holdout_foremost_sample --cat_policy ohe --gpu 0 --seed_num 1"
