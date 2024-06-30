window.BENCHMARK_DATA = {
  "lastUpdate": 1719714157917,
  "repoUrl": "https://github.com/neuralmagic/nm-vllm",
  "entries": {
    "smaller_is_better": [
      {
        "commit": {
          "author": {
            "name": "Domenic Barbuzzi",
            "username": "dbarbuzzi",
            "email": "dbarbuzzi@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "569c9051e46243e36d0b6fbbe5c6e3f9df2956f3",
          "message": "Benchmarking update - phase 1 (#339)\n\nThis PR updates the benchmarking performed in remote-push and nightly\r\nruns according to the first set of deliverables from our recent meeting:\r\n\r\n* Only the `benchmark_serving.json` config is run\r\n* This is accomplished with a new list,\r\n`nm_benchmark_base_config_list.txt`, other lists are untouched\r\n* The `benchmark_serving.json` has various reductions:\r\n* Model list reduced to `facebook/opt-350m` and\r\n`meta-llama/Meta-Llama-3-8B-Instruct`\r\n  * `nr-qps` list reduced to `300,1`\r\n* Metric tracking reduced to mean TPOT and mean TTFT (other metrics\r\nstill recorded/logged per usual)\r\n\r\nThere is also a small fix related to server startup (changing from\r\n`localhost` to `127.0.0.1` because `localhost` on the machines is mapped\r\nto the IPv6 `::1` which something in the server stack doesn’t seem to\r\nlike).\r\n\r\nIn a commit prior to opening the PR with all functional changes, the\r\nfull `benchmark` job took <30 min:\r\n\r\nhttps://github.com/neuralmagic/nm-vllm/actions/runs/9669361155/job/26709082658",
          "timestamp": "2024-06-28T20:39:56Z",
          "url": "https://github.com/neuralmagic/nm-vllm/commit/569c9051e46243e36d0b6fbbe5c6e3f9df2956f3"
        },
        "date": 1719627718988,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "{\"name\": \"mean_ttft_ms\", \"description\": \"VLLM Serving - Dense\\nmodel - facebook/opt-350m\\nmax-model-len - 2048\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\", \"gpu_description\": \"NVIDIA L4 x 1\", \"vllm_version\": \"0.5.1\", \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\", \"torch_version\": \"2.3.0+cu121\"}",
            "value": 24.077120273334078,
            "unit": "ms",
            "extra": "{\n  \"description\": \"VLLM Serving - Dense\\nmodel - facebook/opt-350m\\nmax-model-len - 2048\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n  \"benchmarking_context\": {\n    \"vllm_version\": \"0.5.1\",\n    \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\",\n    \"torch_version\": \"2.3.0+cu121\",\n    \"torch_cuda_version\": \"12.1\",\n    \"cuda_devices\": \"[_CudaDeviceProperties(name='NVIDIA L4', major=8, minor=9, total_memory=22478MB, multi_processor_count=58)]\",\n    \"cuda_device_names\": [\n      \"NVIDIA L4\"\n    ]\n  },\n  \"gpu_description\": \"NVIDIA L4 x 1\",\n  \"script_name\": \"benchmark_serving.py\",\n  \"script_args\": {\n    \"description\": \"VLLM Serving - Dense\\nmodel - facebook/opt-350m\\nmax-model-len - 2048\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n    \"backend\": \"vllm\",\n    \"version\": \"N/A\",\n    \"base_url\": null,\n    \"host\": \"127.0.0.1\",\n    \"port\": 9000,\n    \"endpoint\": \"/generate\",\n    \"dataset\": \"sharegpt\",\n    \"num_input_tokens\": null,\n    \"num_output_tokens\": null,\n    \"model\": \"facebook/opt-350m\",\n    \"tokenizer\": \"facebook/opt-350m\",\n    \"best_of\": 1,\n    \"use_beam_search\": false,\n    \"log_model_io\": false,\n    \"seed\": 0,\n    \"trust_remote_code\": false,\n    \"disable_tqdm\": false,\n    \"save_directory\": \"benchmark-results\",\n    \"num_prompts_\": null,\n    \"request_rate_\": null,\n    \"nr_qps_pair_\": [\n      300,\n      \"1.0\"\n    ],\n    \"server_tensor_parallel_size\": 1,\n    \"server_args\": \"{'model': 'facebook/opt-350m', 'tokenizer': 'facebook/opt-350m', 'max-model-len': 2048, 'host': '127.0.0.1', 'port': 9000, 'tensor-parallel-size': 1, 'disable-log-requests': ''}\"\n  },\n  \"date\": \"2024-06-29 02:12:44 UTC\",\n  \"model\": \"facebook/opt-350m\",\n  \"dataset\": \"sharegpt\"\n}"
          },
          {
            "name": "{\"name\": \"mean_tpot_ms\", \"description\": \"VLLM Serving - Dense\\nmodel - facebook/opt-350m\\nmax-model-len - 2048\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\", \"gpu_description\": \"NVIDIA L4 x 1\", \"vllm_version\": \"0.5.1\", \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\", \"torch_version\": \"2.3.0+cu121\"}",
            "value": 6.047396248220804,
            "unit": "ms",
            "extra": "{\n  \"description\": \"VLLM Serving - Dense\\nmodel - facebook/opt-350m\\nmax-model-len - 2048\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n  \"benchmarking_context\": {\n    \"vllm_version\": \"0.5.1\",\n    \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\",\n    \"torch_version\": \"2.3.0+cu121\",\n    \"torch_cuda_version\": \"12.1\",\n    \"cuda_devices\": \"[_CudaDeviceProperties(name='NVIDIA L4', major=8, minor=9, total_memory=22478MB, multi_processor_count=58)]\",\n    \"cuda_device_names\": [\n      \"NVIDIA L4\"\n    ]\n  },\n  \"gpu_description\": \"NVIDIA L4 x 1\",\n  \"script_name\": \"benchmark_serving.py\",\n  \"script_args\": {\n    \"description\": \"VLLM Serving - Dense\\nmodel - facebook/opt-350m\\nmax-model-len - 2048\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n    \"backend\": \"vllm\",\n    \"version\": \"N/A\",\n    \"base_url\": null,\n    \"host\": \"127.0.0.1\",\n    \"port\": 9000,\n    \"endpoint\": \"/generate\",\n    \"dataset\": \"sharegpt\",\n    \"num_input_tokens\": null,\n    \"num_output_tokens\": null,\n    \"model\": \"facebook/opt-350m\",\n    \"tokenizer\": \"facebook/opt-350m\",\n    \"best_of\": 1,\n    \"use_beam_search\": false,\n    \"log_model_io\": false,\n    \"seed\": 0,\n    \"trust_remote_code\": false,\n    \"disable_tqdm\": false,\n    \"save_directory\": \"benchmark-results\",\n    \"num_prompts_\": null,\n    \"request_rate_\": null,\n    \"nr_qps_pair_\": [\n      300,\n      \"1.0\"\n    ],\n    \"server_tensor_parallel_size\": 1,\n    \"server_args\": \"{'model': 'facebook/opt-350m', 'tokenizer': 'facebook/opt-350m', 'max-model-len': 2048, 'host': '127.0.0.1', 'port': 9000, 'tensor-parallel-size': 1, 'disable-log-requests': ''}\"\n  },\n  \"date\": \"2024-06-29 02:12:44 UTC\",\n  \"model\": \"facebook/opt-350m\",\n  \"dataset\": \"sharegpt\"\n}"
          },
          {
            "name": "{\"name\": \"mean_ttft_ms\", \"description\": \"VLLM Serving - Dense\\nmodel - meta-llama/Meta-Llama-3-8B-Instruct\\nmax-model-len - 4096\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\", \"gpu_description\": \"NVIDIA L4 x 1\", \"vllm_version\": \"0.5.1\", \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\", \"torch_version\": \"2.3.0+cu121\"}",
            "value": 181.2085490133336,
            "unit": "ms",
            "extra": "{\n  \"description\": \"VLLM Serving - Dense\\nmodel - meta-llama/Meta-Llama-3-8B-Instruct\\nmax-model-len - 4096\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n  \"benchmarking_context\": {\n    \"vllm_version\": \"0.5.1\",\n    \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\",\n    \"torch_version\": \"2.3.0+cu121\",\n    \"torch_cuda_version\": \"12.1\",\n    \"cuda_devices\": \"[_CudaDeviceProperties(name='NVIDIA L4', major=8, minor=9, total_memory=22478MB, multi_processor_count=58)]\",\n    \"cuda_device_names\": [\n      \"NVIDIA L4\"\n    ]\n  },\n  \"gpu_description\": \"NVIDIA L4 x 1\",\n  \"script_name\": \"benchmark_serving.py\",\n  \"script_args\": {\n    \"description\": \"VLLM Serving - Dense\\nmodel - meta-llama/Meta-Llama-3-8B-Instruct\\nmax-model-len - 4096\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n    \"backend\": \"vllm\",\n    \"version\": \"N/A\",\n    \"base_url\": null,\n    \"host\": \"127.0.0.1\",\n    \"port\": 9000,\n    \"endpoint\": \"/generate\",\n    \"dataset\": \"sharegpt\",\n    \"num_input_tokens\": null,\n    \"num_output_tokens\": null,\n    \"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\",\n    \"tokenizer\": \"meta-llama/Meta-Llama-3-8B-Instruct\",\n    \"best_of\": 1,\n    \"use_beam_search\": false,\n    \"log_model_io\": false,\n    \"seed\": 0,\n    \"trust_remote_code\": false,\n    \"disable_tqdm\": false,\n    \"save_directory\": \"benchmark-results\",\n    \"num_prompts_\": null,\n    \"request_rate_\": null,\n    \"nr_qps_pair_\": [\n      300,\n      \"1.0\"\n    ],\n    \"server_tensor_parallel_size\": 1,\n    \"server_args\": \"{'model': 'meta-llama/Meta-Llama-3-8B-Instruct', 'tokenizer': 'meta-llama/Meta-Llama-3-8B-Instruct', 'max-model-len': 4096, 'host': '127.0.0.1', 'port': 9000, 'tensor-parallel-size': 1, 'disable-log-requests': ''}\"\n  },\n  \"date\": \"2024-06-29 02:21:10 UTC\",\n  \"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\",\n  \"dataset\": \"sharegpt\"\n}"
          },
          {
            "name": "{\"name\": \"mean_tpot_ms\", \"description\": \"VLLM Serving - Dense\\nmodel - meta-llama/Meta-Llama-3-8B-Instruct\\nmax-model-len - 4096\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\", \"gpu_description\": \"NVIDIA L4 x 1\", \"vllm_version\": \"0.5.1\", \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\", \"torch_version\": \"2.3.0+cu121\"}",
            "value": 84.82343586161166,
            "unit": "ms",
            "extra": "{\n  \"description\": \"VLLM Serving - Dense\\nmodel - meta-llama/Meta-Llama-3-8B-Instruct\\nmax-model-len - 4096\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n  \"benchmarking_context\": {\n    \"vllm_version\": \"0.5.1\",\n    \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\",\n    \"torch_version\": \"2.3.0+cu121\",\n    \"torch_cuda_version\": \"12.1\",\n    \"cuda_devices\": \"[_CudaDeviceProperties(name='NVIDIA L4', major=8, minor=9, total_memory=22478MB, multi_processor_count=58)]\",\n    \"cuda_device_names\": [\n      \"NVIDIA L4\"\n    ]\n  },\n  \"gpu_description\": \"NVIDIA L4 x 1\",\n  \"script_name\": \"benchmark_serving.py\",\n  \"script_args\": {\n    \"description\": \"VLLM Serving - Dense\\nmodel - meta-llama/Meta-Llama-3-8B-Instruct\\nmax-model-len - 4096\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n    \"backend\": \"vllm\",\n    \"version\": \"N/A\",\n    \"base_url\": null,\n    \"host\": \"127.0.0.1\",\n    \"port\": 9000,\n    \"endpoint\": \"/generate\",\n    \"dataset\": \"sharegpt\",\n    \"num_input_tokens\": null,\n    \"num_output_tokens\": null,\n    \"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\",\n    \"tokenizer\": \"meta-llama/Meta-Llama-3-8B-Instruct\",\n    \"best_of\": 1,\n    \"use_beam_search\": false,\n    \"log_model_io\": false,\n    \"seed\": 0,\n    \"trust_remote_code\": false,\n    \"disable_tqdm\": false,\n    \"save_directory\": \"benchmark-results\",\n    \"num_prompts_\": null,\n    \"request_rate_\": null,\n    \"nr_qps_pair_\": [\n      300,\n      \"1.0\"\n    ],\n    \"server_tensor_parallel_size\": 1,\n    \"server_args\": \"{'model': 'meta-llama/Meta-Llama-3-8B-Instruct', 'tokenizer': 'meta-llama/Meta-Llama-3-8B-Instruct', 'max-model-len': 4096, 'host': '127.0.0.1', 'port': 9000, 'tensor-parallel-size': 1, 'disable-log-requests': ''}\"\n  },\n  \"date\": \"2024-06-29 02:21:10 UTC\",\n  \"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\",\n  \"dataset\": \"sharegpt\"\n}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Domenic Barbuzzi",
            "username": "dbarbuzzi",
            "email": "dbarbuzzi@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "569c9051e46243e36d0b6fbbe5c6e3f9df2956f3",
          "message": "Benchmarking update - phase 1 (#339)\n\nThis PR updates the benchmarking performed in remote-push and nightly\r\nruns according to the first set of deliverables from our recent meeting:\r\n\r\n* Only the `benchmark_serving.json` config is run\r\n* This is accomplished with a new list,\r\n`nm_benchmark_base_config_list.txt`, other lists are untouched\r\n* The `benchmark_serving.json` has various reductions:\r\n* Model list reduced to `facebook/opt-350m` and\r\n`meta-llama/Meta-Llama-3-8B-Instruct`\r\n  * `nr-qps` list reduced to `300,1`\r\n* Metric tracking reduced to mean TPOT and mean TTFT (other metrics\r\nstill recorded/logged per usual)\r\n\r\nThere is also a small fix related to server startup (changing from\r\n`localhost` to `127.0.0.1` because `localhost` on the machines is mapped\r\nto the IPv6 `::1` which something in the server stack doesn’t seem to\r\nlike).\r\n\r\nIn a commit prior to opening the PR with all functional changes, the\r\nfull `benchmark` job took <30 min:\r\n\r\nhttps://github.com/neuralmagic/nm-vllm/actions/runs/9669361155/job/26709082658",
          "timestamp": "2024-06-28T20:39:56Z",
          "url": "https://github.com/neuralmagic/nm-vllm/commit/569c9051e46243e36d0b6fbbe5c6e3f9df2956f3"
        },
        "date": 1719714157452,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "{\"name\": \"mean_ttft_ms\", \"description\": \"VLLM Serving - Dense\\nmodel - facebook/opt-350m\\nmax-model-len - 2048\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\", \"gpu_description\": \"NVIDIA L4 x 1\", \"vllm_version\": \"0.5.1\", \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\", \"torch_version\": \"2.3.0+cu121\"}",
            "value": 25.614670530339936,
            "unit": "ms",
            "extra": "{\n  \"description\": \"VLLM Serving - Dense\\nmodel - facebook/opt-350m\\nmax-model-len - 2048\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n  \"benchmarking_context\": {\n    \"vllm_version\": \"0.5.1\",\n    \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\",\n    \"torch_version\": \"2.3.0+cu121\",\n    \"torch_cuda_version\": \"12.1\",\n    \"cuda_devices\": \"[_CudaDeviceProperties(name='NVIDIA L4', major=8, minor=9, total_memory=22478MB, multi_processor_count=58)]\",\n    \"cuda_device_names\": [\n      \"NVIDIA L4\"\n    ]\n  },\n  \"gpu_description\": \"NVIDIA L4 x 1\",\n  \"script_name\": \"benchmark_serving.py\",\n  \"script_args\": {\n    \"description\": \"VLLM Serving - Dense\\nmodel - facebook/opt-350m\\nmax-model-len - 2048\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n    \"backend\": \"vllm\",\n    \"version\": \"N/A\",\n    \"base_url\": null,\n    \"host\": \"127.0.0.1\",\n    \"port\": 9000,\n    \"endpoint\": \"/generate\",\n    \"dataset\": \"sharegpt\",\n    \"num_input_tokens\": null,\n    \"num_output_tokens\": null,\n    \"model\": \"facebook/opt-350m\",\n    \"tokenizer\": \"facebook/opt-350m\",\n    \"best_of\": 1,\n    \"use_beam_search\": false,\n    \"log_model_io\": false,\n    \"seed\": 0,\n    \"trust_remote_code\": false,\n    \"disable_tqdm\": false,\n    \"save_directory\": \"benchmark-results\",\n    \"num_prompts_\": null,\n    \"request_rate_\": null,\n    \"nr_qps_pair_\": [\n      300,\n      \"1.0\"\n    ],\n    \"server_tensor_parallel_size\": 1,\n    \"server_args\": \"{'model': 'facebook/opt-350m', 'tokenizer': 'facebook/opt-350m', 'max-model-len': 2048, 'host': '127.0.0.1', 'port': 9000, 'tensor-parallel-size': 1, 'disable-log-requests': ''}\"\n  },\n  \"date\": \"2024-06-30 02:12:25 UTC\",\n  \"model\": \"facebook/opt-350m\",\n  \"dataset\": \"sharegpt\"\n}"
          },
          {
            "name": "{\"name\": \"mean_tpot_ms\", \"description\": \"VLLM Serving - Dense\\nmodel - facebook/opt-350m\\nmax-model-len - 2048\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\", \"gpu_description\": \"NVIDIA L4 x 1\", \"vllm_version\": \"0.5.1\", \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\", \"torch_version\": \"2.3.0+cu121\"}",
            "value": 6.154629695763885,
            "unit": "ms",
            "extra": "{\n  \"description\": \"VLLM Serving - Dense\\nmodel - facebook/opt-350m\\nmax-model-len - 2048\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n  \"benchmarking_context\": {\n    \"vllm_version\": \"0.5.1\",\n    \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\",\n    \"torch_version\": \"2.3.0+cu121\",\n    \"torch_cuda_version\": \"12.1\",\n    \"cuda_devices\": \"[_CudaDeviceProperties(name='NVIDIA L4', major=8, minor=9, total_memory=22478MB, multi_processor_count=58)]\",\n    \"cuda_device_names\": [\n      \"NVIDIA L4\"\n    ]\n  },\n  \"gpu_description\": \"NVIDIA L4 x 1\",\n  \"script_name\": \"benchmark_serving.py\",\n  \"script_args\": {\n    \"description\": \"VLLM Serving - Dense\\nmodel - facebook/opt-350m\\nmax-model-len - 2048\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n    \"backend\": \"vllm\",\n    \"version\": \"N/A\",\n    \"base_url\": null,\n    \"host\": \"127.0.0.1\",\n    \"port\": 9000,\n    \"endpoint\": \"/generate\",\n    \"dataset\": \"sharegpt\",\n    \"num_input_tokens\": null,\n    \"num_output_tokens\": null,\n    \"model\": \"facebook/opt-350m\",\n    \"tokenizer\": \"facebook/opt-350m\",\n    \"best_of\": 1,\n    \"use_beam_search\": false,\n    \"log_model_io\": false,\n    \"seed\": 0,\n    \"trust_remote_code\": false,\n    \"disable_tqdm\": false,\n    \"save_directory\": \"benchmark-results\",\n    \"num_prompts_\": null,\n    \"request_rate_\": null,\n    \"nr_qps_pair_\": [\n      300,\n      \"1.0\"\n    ],\n    \"server_tensor_parallel_size\": 1,\n    \"server_args\": \"{'model': 'facebook/opt-350m', 'tokenizer': 'facebook/opt-350m', 'max-model-len': 2048, 'host': '127.0.0.1', 'port': 9000, 'tensor-parallel-size': 1, 'disable-log-requests': ''}\"\n  },\n  \"date\": \"2024-06-30 02:12:25 UTC\",\n  \"model\": \"facebook/opt-350m\",\n  \"dataset\": \"sharegpt\"\n}"
          },
          {
            "name": "{\"name\": \"mean_ttft_ms\", \"description\": \"VLLM Serving - Dense\\nmodel - meta-llama/Meta-Llama-3-8B-Instruct\\nmax-model-len - 4096\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\", \"gpu_description\": \"NVIDIA L4 x 1\", \"vllm_version\": \"0.5.1\", \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\", \"torch_version\": \"2.3.0+cu121\"}",
            "value": 186.66584371996578,
            "unit": "ms",
            "extra": "{\n  \"description\": \"VLLM Serving - Dense\\nmodel - meta-llama/Meta-Llama-3-8B-Instruct\\nmax-model-len - 4096\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n  \"benchmarking_context\": {\n    \"vllm_version\": \"0.5.1\",\n    \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\",\n    \"torch_version\": \"2.3.0+cu121\",\n    \"torch_cuda_version\": \"12.1\",\n    \"cuda_devices\": \"[_CudaDeviceProperties(name='NVIDIA L4', major=8, minor=9, total_memory=22478MB, multi_processor_count=58)]\",\n    \"cuda_device_names\": [\n      \"NVIDIA L4\"\n    ]\n  },\n  \"gpu_description\": \"NVIDIA L4 x 1\",\n  \"script_name\": \"benchmark_serving.py\",\n  \"script_args\": {\n    \"description\": \"VLLM Serving - Dense\\nmodel - meta-llama/Meta-Llama-3-8B-Instruct\\nmax-model-len - 4096\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n    \"backend\": \"vllm\",\n    \"version\": \"N/A\",\n    \"base_url\": null,\n    \"host\": \"127.0.0.1\",\n    \"port\": 9000,\n    \"endpoint\": \"/generate\",\n    \"dataset\": \"sharegpt\",\n    \"num_input_tokens\": null,\n    \"num_output_tokens\": null,\n    \"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\",\n    \"tokenizer\": \"meta-llama/Meta-Llama-3-8B-Instruct\",\n    \"best_of\": 1,\n    \"use_beam_search\": false,\n    \"log_model_io\": false,\n    \"seed\": 0,\n    \"trust_remote_code\": false,\n    \"disable_tqdm\": false,\n    \"save_directory\": \"benchmark-results\",\n    \"num_prompts_\": null,\n    \"request_rate_\": null,\n    \"nr_qps_pair_\": [\n      300,\n      \"1.0\"\n    ],\n    \"server_tensor_parallel_size\": 1,\n    \"server_args\": \"{'model': 'meta-llama/Meta-Llama-3-8B-Instruct', 'tokenizer': 'meta-llama/Meta-Llama-3-8B-Instruct', 'max-model-len': 4096, 'host': '127.0.0.1', 'port': 9000, 'tensor-parallel-size': 1, 'disable-log-requests': ''}\"\n  },\n  \"date\": \"2024-06-30 02:21:51 UTC\",\n  \"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\",\n  \"dataset\": \"sharegpt\"\n}"
          },
          {
            "name": "{\"name\": \"mean_tpot_ms\", \"description\": \"VLLM Serving - Dense\\nmodel - meta-llama/Meta-Llama-3-8B-Instruct\\nmax-model-len - 4096\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\", \"gpu_description\": \"NVIDIA L4 x 1\", \"vllm_version\": \"0.5.1\", \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\", \"torch_version\": \"2.3.0+cu121\"}",
            "value": 85.21498309830577,
            "unit": "ms",
            "extra": "{\n  \"description\": \"VLLM Serving - Dense\\nmodel - meta-llama/Meta-Llama-3-8B-Instruct\\nmax-model-len - 4096\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n  \"benchmarking_context\": {\n    \"vllm_version\": \"0.5.1\",\n    \"python_version\": \"3.10.12 (main, Jun  7 2023, 13:43:11) [GCC 11.3.0]\",\n    \"torch_version\": \"2.3.0+cu121\",\n    \"torch_cuda_version\": \"12.1\",\n    \"cuda_devices\": \"[_CudaDeviceProperties(name='NVIDIA L4', major=8, minor=9, total_memory=22478MB, multi_processor_count=58)]\",\n    \"cuda_device_names\": [\n      \"NVIDIA L4\"\n    ]\n  },\n  \"gpu_description\": \"NVIDIA L4 x 1\",\n  \"script_name\": \"benchmark_serving.py\",\n  \"script_args\": {\n    \"description\": \"VLLM Serving - Dense\\nmodel - meta-llama/Meta-Llama-3-8B-Instruct\\nmax-model-len - 4096\\nsparsity - None\\nbenchmark_serving {\\n  \\\"nr-qps-pair_\\\": \\\"300,1\\\",\\n  \\\"dataset\\\": \\\"sharegpt\\\"\\n}\",\n    \"backend\": \"vllm\",\n    \"version\": \"N/A\",\n    \"base_url\": null,\n    \"host\": \"127.0.0.1\",\n    \"port\": 9000,\n    \"endpoint\": \"/generate\",\n    \"dataset\": \"sharegpt\",\n    \"num_input_tokens\": null,\n    \"num_output_tokens\": null,\n    \"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\",\n    \"tokenizer\": \"meta-llama/Meta-Llama-3-8B-Instruct\",\n    \"best_of\": 1,\n    \"use_beam_search\": false,\n    \"log_model_io\": false,\n    \"seed\": 0,\n    \"trust_remote_code\": false,\n    \"disable_tqdm\": false,\n    \"save_directory\": \"benchmark-results\",\n    \"num_prompts_\": null,\n    \"request_rate_\": null,\n    \"nr_qps_pair_\": [\n      300,\n      \"1.0\"\n    ],\n    \"server_tensor_parallel_size\": 1,\n    \"server_args\": \"{'model': 'meta-llama/Meta-Llama-3-8B-Instruct', 'tokenizer': 'meta-llama/Meta-Llama-3-8B-Instruct', 'max-model-len': 4096, 'host': '127.0.0.1', 'port': 9000, 'tensor-parallel-size': 1, 'disable-log-requests': ''}\"\n  },\n  \"date\": \"2024-06-30 02:21:51 UTC\",\n  \"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\",\n  \"dataset\": \"sharegpt\"\n}"
          }
        ]
      }
    ]
  }
}