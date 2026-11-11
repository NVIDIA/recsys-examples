// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package fep;

import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordedFrame;
import jdk.jfr.consumer.RecordedMethod;
import jdk.jfr.consumer.RecordedStackTrace;
import jdk.jfr.consumer.RecordedThread;
import jdk.jfr.consumer.RecordingFile;

import java.lang.reflect.Modifier;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Classifies JFR execution / native-method samples of Flink task threads into pipeline stages.
 *
 * <p>Each sample is attributed to the first stage rule that matches when walking its stack from
 * the leaf towards the root. Output: per thread, sample count per stage (JSON). Shares are sampled
 * estimates of where the busy thread time goes, not exact timers.
 *
 * <p>usage: JfrBreakdown FILE.jfr
 */
public final class JfrBreakdown {
    static String stage(RecordedFrame f) {
        RecordedMethod m = f.getMethod();
        if (m == null || m.getType() == null) {
            return null;
        }
        String c = m.getType().getName();
        String n = m.getName();
        boolean nat = Modifier.isNative(m.getModifiers());
        if (c.equals("jdk.internal.misc.Unsafe") && n.equals("park")) {
            return "idle/wait (park)";
        }
        if (c.startsWith("java.lang.Thread") && n.equals("sleep")) {
            return "idle/wait (park)";
        }
        if (c.startsWith("io.github.cb521.cudfflink") && nat) {
            return "GPU JNI call (H2D+kernel+D2H+sync)";
        }
        if (c.contains("FepKuaiRandBench$ReplayMap")
                || c.startsWith("org.apache.flink.api.connector.source.lib")
                || c.startsWith("org.apache.flink.streaming.api.operators.SourceOperator")) {
            return "source/replay";
        }
        if (c.contains("FepKuaiRandBench$CaptureSink")
                || c.startsWith("org.apache.flink.streaming.runtime.operators.sink")) {
            return "sink";
        }
        if (c.startsWith("org.apache.flink.runtime.state")
                || c.startsWith("org.apache.flink.table.runtime.dataview")
                || c.startsWith("org.apache.flink.streaming.api.operators.StreamingRuntimeContext")
                || c.contains("KeyedStateBackend")
                || c.contains("HeapValueState")
                || c.contains("HeapMapState")
                || c.contains("HeapListState")) {
            return "state read/write";
        }
        if (c.startsWith("org.apache.flink.streaming.api.operators.InternalTimer")
                || c.startsWith("org.apache.flink.streaming.api.operators.HeapInternalTimer")
                || c.contains("TimerService")) {
            return "timers/watermark";
        }
        if (c.startsWith("org.apache.flink.runtime.io.network")
                || c.startsWith("org.apache.flink.streaming.runtime.io")
                || c.startsWith("org.apache.flink.runtime.io.disk")
                || c.startsWith("org.apache.flink.core.memory")
                || c.startsWith("org.apache.flink.api.common.typeutils")
                || c.startsWith("org.apache.flink.api.java.typeutils.runtime")
                || c.startsWith("org.apache.flink.table.runtime.typeutils")
                || c.startsWith("org.apache.flink.streaming.runtime.streamrecord")
                || c.startsWith("org.apache.flink.runtime.pluggable")) {
            return "serialization/network exchange";
        }
        if (c.startsWith("org.apache.flink.table.data.conversion")
                || c.startsWith("org.apache.flink.table.runtime.operators.source")
                || c.startsWith("org.apache.flink.table.runtime.operators.sink")
                || c.contains("InputConversionOperator")
                || c.contains("OutputConversionOperator")
                || c.contains("DataStructureConverter")
                || c.contains("RowRowConverter")) {
            return "DataStream<->Table row conversion";
        }
        if (c.startsWith("io.github.cb521.cudfflink")) {
            return "GPU operator JVM side (pack/unpack/merge/emit)";
        }
        if (c.startsWith("org.apache.flink.table.runtime.operators")
                || c.startsWith("org.apache.flink.table.runtime.generated")
                || c.startsWith("org.apache.flink.table.runtime.functions")
                || c.startsWith("org.apache.flink.table.data")
                || c.contains("$Generated")
                || c.matches(".*(Calc|AggsHandler|Projection|Condition|Comparator|KeySelector|"
                        + "Function|Equaliser)\\$\\d+.*")) {
            return "operator compute (Flink CPU code)";
        }
        return null;
    }

    public static void main(String[] args) throws Exception {
        Map<String, Map<String, Long>> perThread = new TreeMap<>();
        try (RecordingFile rf = new RecordingFile(Paths.get(args[0]))) {
            while (rf.hasMoreEvents()) {
                RecordedEvent e = rf.readEvent();
                String type = e.getEventType().getName();
                if (!type.equals("jdk.ExecutionSample") && !type.equals("jdk.NativeMethodSample")) {
                    continue;
                }
                RecordedThread t = e.getThread("sampledThread");
                String tn = t == null ? "?" : t.getJavaName();
                if (tn == null || !tn.contains("(1/1)")) {
                    continue; // only Flink task threads
                }
                tn = tn.replaceAll("#\\d+$", "");
                if (tn.length() > 90) {
                    tn = tn.substring(0, 90) + "...";
                }
                RecordedStackTrace st = e.getStackTrace();
                String s = null;
                if (st != null) {
                    List<RecordedFrame> frames = st.getFrames();
                    for (RecordedFrame f : frames) {
                        s = stage(f);
                        if (s != null) {
                            break;
                        }
                    }
                }
                if (s == null) {
                    s = "Flink runtime/other";
                }
                if (type.equals("jdk.NativeMethodSample") && !s.startsWith("GPU JNI")
                        && !s.startsWith("idle")) {
                    s = "idle/wait (native)";
                }
                perThread.computeIfAbsent(tn, k -> new LinkedHashMap<>()).merge(s, 1L, Long::sum);
            }
        }
        StringBuilder out = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, Map<String, Long>> e : perThread.entrySet()) {
            if (!first) {
                out.append(",");
            }
            first = false;
            out.append("\n  \"").append(e.getKey().replace("\"", "'")).append("\": {");
            boolean f2 = true;
            for (Map.Entry<String, Long> s : e.getValue().entrySet()) {
                if (!f2) {
                    out.append(", ");
                }
                f2 = false;
                out.append('"').append(s.getKey()).append("\": ").append(s.getValue());
            }
            out.append('}');
        }
        System.out.println(out.append("\n}"));
    }
}
