// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package fep;

import org.apache.flink.api.common.io.InputFormat;
import org.apache.flink.api.common.io.statistics.BaseStatistics;
import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.io.GenericInputSplit;
import org.apache.flink.core.io.InputSplitAssigner;
import org.apache.flink.api.common.io.DefaultInputSplitAssigner;
import org.apache.flink.table.connector.source.DynamicTableSource;
import org.apache.flink.table.connector.source.LookupTableSource;
import org.apache.flink.table.connector.source.lookup.LookupFunctionProvider;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.factories.DynamicTableSourceFactory;
import org.apache.flink.table.functions.FunctionContext;
import org.apache.flink.table.functions.LookupFunction;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;

/**
 * KuaiRand video dimension (video_id -> author_id) as a bounded, read-only, in-memory table.
 *
 * <p>Flink CPU uses the {@link LookupFunction} (a full in-memory hash map, i.e. the best case of a
 * full-cache lookup). The Mega-Flink variant in src/megaflink additionally exposes the same rows as a
 * snapshot so that a GPU lookup can load them into device memory. Both read the same rows prepared by
 * {@link FepKuaiRandBench}.
 */
public final class FepVideoDimFactory implements DynamicTableSourceFactory {
    public static final String IDENTIFIER = "fep-video-dim";

    @Override
    public String factoryIdentifier() {
        return IDENTIFIER;
    }

    @Override
    public Set<ConfigOption<?>> requiredOptions() {
        return Collections.emptySet();
    }

    @Override
    public Set<ConfigOption<?>> optionalOptions() {
        return Collections.emptySet();
    }

    @Override
    public DynamicTableSource createDynamicTableSource(Context context) {
        return new Source();
    }

    static final class Source implements LookupTableSource {
        @Override
        public LookupRuntimeProvider getLookupRuntimeProvider(LookupContext context) {
            return LookupFunctionProvider.of(new DimLookup());
        }

        @Override
        public DynamicTableSource copy() {
            return new Source();
        }

        @Override
        public String asSummaryString() {
            return IDENTIFIER;
        }
    }

    public static final class DimLookup extends LookupFunction {
        private static final long serialVersionUID = 1L;
        private transient HashMap<Long, RowData> map;

        @Override
        public void open(FunctionContext context) {
            long[] k = FepKuaiRandBench.DIM_KEYS;
            long[] v = FepKuaiRandBench.DIM_VALUES;
            map = new HashMap<>(k.length * 2);
            for (int i = 0; i < k.length; i++) {
                map.put(k[i], GenericRowData.of(k[i], v[i]));
            }
        }

        @Override
        public Collection<RowData> lookup(RowData keyRow) {
            if (keyRow.isNullAt(0)) {
                return Collections.emptyList();
            }
            RowData r = map.get(keyRow.getLong(0));
            return r == null ? Collections.emptyList() : Collections.singletonList(r);
        }
    }

    public static final class DimInputFormat implements InputFormat<RowData, GenericInputSplit> {
        private static final long serialVersionUID = 1L;
        private transient int row;

        @Override
        public void configure(Configuration parameters) {}

        @Override
        public BaseStatistics getStatistics(BaseStatistics cachedStatistics) {
            return null;
        }

        @Override
        public GenericInputSplit[] createInputSplits(int minNumSplits) {
            return new GenericInputSplit[] {new GenericInputSplit(0, 1)};
        }

        @Override
        public InputSplitAssigner getInputSplitAssigner(GenericInputSplit[] inputSplits) {
            return new DefaultInputSplitAssigner(inputSplits);
        }

        @Override
        public void open(GenericInputSplit split) {
            row = 0;
        }

        @Override
        public boolean reachedEnd() {
            return row >= FepKuaiRandBench.DIM_KEYS.length;
        }

        @Override
        public RowData nextRecord(RowData reuse) {
            if (reachedEnd()) {
                return null;
            }
            RowData r =
                    GenericRowData.of(
                            FepKuaiRandBench.DIM_KEYS[row], FepKuaiRandBench.DIM_VALUES[row]);
            row++;
            return r;
        }

        @Override
        public void close() {}
    }
}
