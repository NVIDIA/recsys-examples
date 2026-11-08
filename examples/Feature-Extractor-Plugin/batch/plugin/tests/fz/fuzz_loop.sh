#!/bin/bash
# restart the fuzzer after a detected hang (exit 75); progress.json makes it resume after the offending case
while true; do python3 "$(dirname $0)/fuzz.py" "$@"; rc=$?; [ $rc -ne 75 ] && break; echo "restart after hang"; done
exit $rc
