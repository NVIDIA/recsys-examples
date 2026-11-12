// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.examples;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.metrics.Counter;
import org.apache.flink.metrics.Metric;
import org.apache.flink.metrics.MetricConfig;
import org.apache.flink.metrics.MetricGroup;
import org.apache.flink.metrics.reporter.MetricReporter;
import org.apache.flink.metrics.reporter.MetricReporterFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Collects the counters the library operators register ({@code cudf*}, for example
 * {@code cudfLookupJoinGpuBatches} and {@code cudfLookupJoinCpuBatches}) and keeps the last value
 * of each after its task has finished, so the example job can print them once the job is over.
 * This is the direct evidence of how many batches ran the GPU kernel and how many were answered by
 * the JVM.
 */
public final class MetricsCapture implements MetricReporter {
    private static final Map<String, Metric> LIVE = new ConcurrentHashMap<>();
    private static final Map<String, Long> LAST = new ConcurrentHashMap<>();

    /** Reporter factory; Flink instantiates it from the configuration set by {@link #install}. */
    public static final class Factory implements MetricReporterFactory {
        @Override
        public MetricReporter createMetricReporter(Properties properties) {
            return new MetricsCapture();
        }
    }

    /** Registers the reporter in the configuration of the embedded mini cluster. */
    public static Configuration install(Configuration conf) {
        conf.setString("metrics.reporters", "fepcapture");
        conf.setString("metrics.reporter.fepcapture.factory.class", Factory.class.getName());
        return conf;
    }

    @Override
    public void open(MetricConfig config) {}

    @Override
    public void close() {}

    @Override
    public void notifyOfAddedMetric(Metric metric, String metricName, MetricGroup group) {
        if (metricName.startsWith("cudf") && metric instanceof Counter) {
            LIVE.put(group.getMetricIdentifier(metricName), metric);
        }
    }

    @Override
    public void notifyOfRemovedMetric(Metric metric, String metricName, MetricGroup group) {
        String id = group.getMetricIdentifier(metricName);
        if (LIVE.remove(id) != null) {
            LAST.put(id, ((Counter) metric).getCount());
        }
    }

    /** Prints the library counters (summed over operator instances) and writes them to a file. */
    public static void dump(String job, String engine, String variant, Map<String, String> opt)
            throws IOException {
        Map<String, Long> sum = new TreeMap<>();
        Map<String, Long> all = new ConcurrentHashMap<>(LAST);
        for (Map.Entry<String, Metric> e : LIVE.entrySet()) {
            all.put(e.getKey(), ((Counter) e.getValue()).getCount());
        }
        for (Map.Entry<String, Long> e : all.entrySet()) {
            String id = e.getKey();
            sum.merge(id.substring(id.lastIndexOf('.') + 1), e.getValue(), Long::sum);
        }
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Long> e : sum.entrySet()) {
            sb.append(e.getKey()).append('=').append(e.getValue()).append('\n');
        }
        System.out.println(
                "OPERATOR_METRICS job=" + job + " engine=" + engine + " variant=" + variant + " "
                        + sb.toString().trim().replace('\n', ' '));
        Path out = java.nio.file.Paths.get(opt.getOrDefault("out", "."));
        Files.createDirectories(out);
        Files.write(
                out.resolve(job + "-" + engine + "-" + variant + ".metrics.txt"),
                sb.toString().getBytes(StandardCharsets.UTF_8));
    }
}
