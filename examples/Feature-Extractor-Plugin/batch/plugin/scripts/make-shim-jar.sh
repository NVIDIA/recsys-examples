#!/bin/bash
# Compile-time helper: spark-rapids keeps its Spark-version specific ("shim") classes under spark-shared/ and spark359/
# inside the dist jar. The Scala compiler needs them at their normal package path, so repack them into RAPIDS_SHIM_JAR.
# Only used for building (the plugin itself is run with the normal rapids jar). Usage: make-shim-jar.sh [SHIM(spark359)]
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SHIM=${1:-spark359}
[ -f "$RAPIDS_JAR" ] || { echo "RAPIDS_JAR not found: $RAPIDS_JAR"; exit 1; }
mkdir -p "$(dirname "$RAPIDS_SHIM_JAR")"
python3 - "$RAPIDS_JAR" "$RAPIDS_SHIM_JAR" "$SHIM" <<'PY'
import sys, zipfile
src, dst, shim = sys.argv[1:4]
pre = ("spark-shared/", shim + "/")
seen = set()
with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst + ".tmp", "w", zipfile.ZIP_DEFLATED) as zo:
    for e in zi.infolist():
        for p in pre:
            if e.filename.startswith(p) and not e.filename.endswith("/"):
                n = e.filename[len(p):]
                if n and n not in seen:
                    seen.add(n); zo.writestr(n, zi.read(e))
import os; os.replace(dst + ".tmp", dst)
print("wrote", dst, len(seen), "entries")
PY
