import argparse

from hta.trace_analysis import TraceAnalysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hta-trace",
        type=str,
        required=True,
        help="hta trace folder output by examples/offline_profile.py "
        "... --hta-trace <folder-to-create>")
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Output figure file, should be a image file such as pdf, "
        "jpeg, png, etc., defaults to <json_trace>.pdf")
    parser.add_argument("--phase",
                        type=str,
                        choices=["prefill", "decode"],
                        required=True,
                        help="The phase to print the table for.")

    args = parser.parse_args()
    analyzer = TraceAnalysis(trace_dir=f"{args.hta_trace}/{args.phase}")

    kernel_type_metrics_df, kernel_metrics_df = \
        analyzer.get_gpu_kernel_breakdown(visualize=False)
    print(kernel_metrics_df)
    print(kernel_type_metrics_df)

    print(analyzer.get_memory_bw_summary())
    comm_comp_overlap_df = analyzer.get_comm_comp_overlap()
    print(comm_comp_overlap_df)
