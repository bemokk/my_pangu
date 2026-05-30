import os
import numpy as np
import torch
import onnxruntime as ort
from datetime import datetime, timedelta
from forecast_decode_functions import surface, upper


use_GPU = True


def create_ort_session(model_path, options):
    """
    创建 ONNX Runtime 推理 Session。
    如果 use_GPU=True，则优先使用 CUDAExecutionProvider。
    同时检查是否真的启用了 CUDA，避免 ONNX Runtime 悄悄回退到 CPU。
    """

    # print("Torch version:", torch.__version__)
    # print("Torch CUDA version:", torch.version.cuda)
    # print("Torch CUDA available:", torch.cuda.is_available())

    # if torch.cuda.is_available():
    #     print("Torch GPU device:", torch.cuda.get_device_name(0))
    # else:
    #     print("Warning: PyTorch 当前无法使用 CUDA。ONNX Runtime 也很可能无法使用 GPU。")

    if not torch.cuda.is_available():
        print("Warning: PyTorch 当前无法使用 CUDA。")

    # print("ONNX Runtime version:", ort.__version__)
    # print("ONNX Runtime available providers:", ort.get_available_providers())

    if use_GPU:
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "ONNX Runtime 当前没有检测到 CUDAExecutionProvider。\n"
                "请确认安装的是 onnxruntime-gpu，而不是 onnxruntime CPU 版。\n"
                "可执行：python -c \"import onnxruntime as ort; print(ort.get_available_providers())\""
            )

        cuda_options = {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
        }

        providers = [
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider",
        ]

        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=providers
        )

        actual_providers = session.get_providers()
        print("ONNX Runtime session providers:", actual_providers)

        if "CUDAExecutionProvider" not in actual_providers:
            raise RuntimeError(
                "ONNX Runtime Session 创建后没有启用 CUDAExecutionProvider，说明 GPU 加载失败，"
                "程序可能已经回退到 CPU。\n"
                "请检查 cuDNN 9.x、CUDA 12.x、MSVC Runtime 是否能被 ONNX Runtime 找到。"
            )

        return session

    else:
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )

        # print("ONNX Runtime session providers:", session.get_providers())
        return session


def infer(start_time, forecast_hour, dataType):
    print(f"\n====== 任务启动: 预测 T+{forecast_hour} 小时 (起始: {start_time}) ======")

    # 1. 路径设置
    base_time_str = start_time.strftime("%Y-%m-%d-%H-%M")

    if dataType in ["gdas", "GDAS"]:
        use_gdas_data = True
    else:
        use_gdas_data = False

    if use_gdas_data:
        input_root = os.path.join(os.getcwd(), "../model_input/single_time_point/gdas")
        input_dir = os.path.join(input_root, base_time_str)
        output_root = os.path.join(os.getcwd(), "../model_output/gdas")
    else:
        input_root = os.path.join(os.getcwd(), "../model_input/single_time_point/era5")
        input_dir = os.path.join(input_root, base_time_str)
        output_root = os.path.join(os.getcwd(), "../model_output/era5")

    # cache_dir 是时间链推理缓存目录
    cache_dir = os.path.join(output_root, base_time_str, "timeline_cache")
    target_dir = os.path.join(output_root, base_time_str, str(forecast_hour))

    if not os.path.exists(input_dir):
        print(f"Error: Input dir doesn't exist: {input_dir}")
        exit()

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    model_paths = {
        24: "../models/pangu_weather_24.onnx",
        6: "../models/pangu_weather_6.onnx",
        3: "../models/pangu_weather_3.onnx",
        1: "../models/pangu_weather_1.onnx"
    }

    # 2. 推理循环
    remain_hour = forecast_hour
    current_date_time = start_time

    # 初始输入目录
    last_source_dir = input_dir

    last_model = None
    ort_session = None
    is_first_step = True

    while remain_hour >= 1:
        # 2.1 确定步长
        if remain_hour >= 24:
            step_size = 24
        elif remain_hour >= 6:
            step_size = 6
        elif remain_hour >= 3:
            step_size = 3
        else:
            step_size = 1

        model_used = model_paths[step_size]

        next_date_time = current_date_time + timedelta(hours=step_size)
        next_time_str = next_date_time.strftime("%Y-%m-%d-%H-%M")

        # 缓存文件路径
        cache_upper_path = os.path.join(cache_dir, f"output_upper_{next_time_str}.npy")
        cache_surface_path = os.path.join(cache_dir, f"output_surface_{next_time_str}.npy")

        # 2.2 先检查缓存
        if os.path.exists(cache_upper_path) and os.path.exists(cache_surface_path):
            print(f"[跳过] {step_size}h 缓存已存在: {next_time_str}")

        else:
            print(f"[执行] {step_size}h 模型推理: {current_date_time} -> {next_date_time}")

            # A. 加载模型 Session
            if model_used != last_model:
                print(f"Loading model: {model_used}...")

                options = ort.SessionOptions()
                options.enable_cpu_mem_arena = False
                options.intra_op_num_threads = 30

                ort_session = create_ort_session(model_used, options)
                last_model = model_used

            # B. 读取输入
            prev_time_str = current_date_time.strftime("%Y-%m-%d-%H-%M")

            if is_first_step:
                input_upper_path = os.path.join(last_source_dir, "input_upper.npy")
                input_surface_path = os.path.join(last_source_dir, "input_surface.npy")
            else:
                input_upper_path = os.path.join(last_source_dir, f"output_upper_{prev_time_str}.npy")
                input_surface_path = os.path.join(last_source_dir, f"output_surface_{prev_time_str}.npy")

            input_data = np.load(input_upper_path).astype(np.float32)
            input_surface = np.load(input_surface_path).astype(np.float32)

            # C. 运行 ONNX 推理
            output, output_surface = ort_session.run(
                None,
                {
                    "input": input_data,
                    "input_surface": input_surface
                }
            )

            # D. 保存到缓存
            np.save(cache_upper_path, output)
            np.save(cache_surface_path, output_surface)

        # 2.3 如果这是最后一步，把缓存结果复制到目标文件夹
        if remain_hour == step_size:
            target_upper_path = os.path.join(target_dir, f"output_upper_{next_time_str}.npy")
            target_surface_path = os.path.join(target_dir, f"output_surface_{next_time_str}.npy")

            if not os.path.exists(target_upper_path):
                tmp_up = np.load(cache_upper_path)
                tmp_surf = np.load(cache_surface_path)

                np.save(target_upper_path, tmp_up)
                np.save(target_surface_path, tmp_surf)

                print(f"结果已保存至目标文件夹: {target_dir}")

        # 2.4 更新状态
        remain_hour -= step_size
        current_date_time = next_date_time
        last_source_dir = cache_dir
        is_first_step = False

    # 3. 解码部分 .nc
    print("====== 正在解码 .nc 文件 ======")

    for file in os.listdir(target_dir):
        if file.endswith(".npy"):
            target_nc_name = file[:-4] + ".nc"
            target_nc_path = os.path.join(target_dir, target_nc_name)

            if not os.path.exists(target_nc_path):
                if file.startswith("output_surface"):
                    surface(os.path.join(target_dir, file), target_nc_name, target_dir)
                elif file.startswith("output_upper"):
                    upper(os.path.join(target_dir, file), target_nc_name, target_dir)