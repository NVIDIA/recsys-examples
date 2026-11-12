// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package fep;

import java.nio.file.Paths;
import java.util.Map;
import java.util.TreeMap;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

/** Per-thread CPU seconds from jdk.ThreadCPULoad events (period 500 ms): usage ThreadCpu rec.jfr */
public final class ThreadCpu {
    public static void main(String[] a) throws Exception {
        Map<String, double[]> m = new TreeMap<>();
        for (RecordedEvent e : RecordingFile.readAllEvents(Paths.get(a[0]))) {
            if (!e.getEventType().getName().equals("jdk.ThreadCPULoad")) continue;
            String t = e.getThread("eventThread") == null ? "?" : e.getThread("eventThread").getJavaName();
            if (t == null) t = "?";
            double[] v = m.computeIfAbsent(t, k -> new double[2]);
            v[0] += e.getFloat("user") * 0.5;
            v[1] += e.getFloat("system") * 0.5;
        }
        m.entrySet().stream()
                .sorted((x, y) -> Double.compare(y.getValue()[0] + y.getValue()[1], x.getValue()[0] + x.getValue()[1]))
                .limit(12)
                .forEach(en -> System.out.printf("%7.2fs user %6.2fs sys  %s%n", en.getValue()[0], en.getValue()[1],
                        en.getKey().length() > 90 ? en.getKey().substring(0, 90) : en.getKey()));
    }
}
