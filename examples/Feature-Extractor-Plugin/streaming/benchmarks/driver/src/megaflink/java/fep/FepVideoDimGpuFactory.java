// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package fep;

import io.github.cb521.cudfflink.GpuLookupSnapshotProvider;

import org.apache.flink.api.common.io.InputFormat;
import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.table.connector.source.DynamicTableSource;
import org.apache.flink.table.connector.source.LookupTableSource;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.factories.DynamicTableSourceFactory;
import org.apache.flink.table.functions.LookupFunction;

import java.util.Collections;
import java.util.Set;

/**
 * Mega-Flink only (compiled by build.sh when MEGA_FLINK_CP is set). The same in-memory video
 * dimension as {@link FepVideoDimFactory}, additionally exposed through the Mega-Flink snapshot
 * contract so that the GPU lookup join can load it into device memory. The CPU lookup function is the
 * same one the open-source path uses.
 */
public final class FepVideoDimGpuFactory implements DynamicTableSourceFactory {
    public static final String IDENTIFIER = "fep-video-dim-gpu";

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
            return new GpuLookupSnapshotProvider() {
                @Override
                public InputFormat<RowData, ?> createSnapshotInputFormat() {
                    return new FepVideoDimFactory.DimInputFormat();
                }

                @Override
                public String snapshotIdentity() {
                    long h = 1469598103934665603L;
                    for (int i = 0; i < FepKuaiRandBench.DIM_KEYS.length; i++) {
                        h = (h ^ FepKuaiRandBench.DIM_KEYS[i]) * 1099511628211L;
                        h = (h ^ FepKuaiRandBench.DIM_VALUES[i]) * 1099511628211L;
                    }
                    return "kuairand-video-author-" + FepKuaiRandBench.DIM_KEYS.length + "-"
                            + Long.toHexString(h);
                }

                @Override
                public LookupFunction createLookupFunction() {
                    return new FepVideoDimFactory.DimLookup();
                }
            };
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
}
