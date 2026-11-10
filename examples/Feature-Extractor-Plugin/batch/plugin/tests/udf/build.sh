#!/bin/bash
# Compile the UDF test jar with the scalac + javac that ship in the Spark distribution / JDK modules (no network).
set -e
HERE=$(cd $(dirname $0) && pwd); source $HERE/../../scripts/env.sh; SH=$SPARK_HOME
CP=$(ls $SH/jars/*.jar | tr '\n' ':')$RAPIDS_JAR
OUT=$HERE/classes; rm -rf $OUT; mkdir -p $OUT
$JAVA_HOME/bin/java -cp "$SH/jars/scala-compiler-2.12.18.jar:$SH/jars/scala-library-2.12.18.jar:$SH/jars/scala-reflect-2.12.18.jar" scala.tools.nsc.Main -usejavacp:false -nowarn -cp "$CP" -d $OUT $HERE/src/*.scala $HERE/src/*.java
$JAVA_HOME/bin/java -m jdk.compiler/com.sun.tools.javac.Main -cp "$CP:$OUT" -d $OUT $HERE/src/*.java
(cd $OUT && $JAVA_HOME/bin/jar cf $HERE/udfs.jar . 2>/dev/null || python3 -c "import zipfile,os;z=zipfile.ZipFile('$HERE/udfs.jar','w');[z.write(os.path.join(r,f),os.path.relpath(os.path.join(r,f),'.')) for r,_,fs in os.walk('.') for f in fs]")
ls -la $HERE/udfs.jar
