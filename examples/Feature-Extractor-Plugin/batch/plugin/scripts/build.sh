#!/bin/bash
# Build the plugin jar: target/gpu-fusion-plugin-0.1.0.jar (no GPU needed). Extra args go to Maven.
# Needs: Maven 3.9+ (MVN, default `mvn`), JDK 11 (JAVA_HOME), RAPIDS_JAR and RAPIDS_SHIM_JAR (see env.sh / make-shim-jar.sh).
# Behind an HTTP proxy, pass it to Maven yourself, e.g. MAVEN_OPTS="-Dhttps.proxyHost=... -Dhttps.proxyPort=...".
# Optional: MAVEN_REPO=<dir> to use a private local repository.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
[ -f "$RAPIDS_JAR" ] || { echo "RAPIDS_JAR not found: $RAPIDS_JAR"; exit 1; }
[ -f "$RAPIDS_SHIM_JAR" ] || "$FUSION_HOME/scripts/make-shim-jar.sh" || exit 1
REPO=${MAVEN_REPO:+-Dmaven.repo.local=$MAVEN_REPO}
cd "$FUSION_HOME" && ${MVN:-mvn} -q $REPO -Drapids.jar="$RAPIDS_JAR" -Drapids.shim.jar="$RAPIDS_SHIM_JAR" "$@" package -DskipTests
