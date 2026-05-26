import os
import numpy as np
import onnx
import onnxruntime as ort
from datetime import datetime, timedelta
from forecast_decode_functions import surface, upper


use_GPU = False

def infer(start_time, forecast_hour, dataType):
    print(f"\n====== 任务启动: 预测 T+{forecast_hour} 小时 (起始: {start_time}) ======")

    # 1. 路径设置
    base_time_str = start_time.strftime("%Y-%m-%d-%H-%M")

    if dataType in ['gdas','GDAS']:
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

    # 核心修改：cache_dir 是我们唯一的“信任源”
    cache_dir = os.path.join(output_root, base_time_str, "timeline_cache")
    target_dir = os.path.join(output_root, base_time_str, str(forecast_hour))

    if not os.path.exists(input_dir):
        print(f"Error: Input dir doesn't exist: {input_dir}")
        exit()

    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
    if not os.path.exists(target_dir): os.makedirs(target_dir)

    model_paths = {
        24: '../models/pangu_weather_24.onnx',
        6: '../models/pangu_weather_6.onnx',
        3: '../models/pangu_weather_3.onnx',
        1: '../models/pangu_weather_1.onnx'
    }

    # 2. 推理循环
    remain_hour = forecast_hour
    current_date_time = start_time

    # 记录上一步数据的来源目录（初始为输入目录）
    last_source_dir = input_dir

    model_used = None
    last_model = None
    ort_session = None
    is_first_step = True

    while remain_hour >= 1:
        # 2.1 确定步长
        step_size = 24 if remain_hour >= 24 else (6 if remain_hour >= 6 else (3 if remain_hour >= 3 else 1))
        model_used = model_paths[step_size]

        next_date_time = current_date_time + timedelta(hours=step_size)
        next_time_str = next_date_time.strftime("%Y-%m-%d-%H-%M")

        # 定义缓存文件名 (这是我们要检查和保存的主力位置)
        cache_upper_path = os.path.join(cache_dir, f'output_upper_{next_time_str}.npy')
        cache_surface_path = os.path.join(cache_dir, f'output_surface_{next_time_str}.npy')

        # 2.2 核心优化：先查缓存
        if os.path.exists(cache_upper_path) and os.path.exists(cache_surface_path):
            print(f"[跳过] {step_size}h 缓存已存在: {next_time_str}")
            # 如果这一步恰好是最终目标，我们需要把缓存里的文件“复制”到目标文件夹
            # (为了省事，这里直接如果不最后一步就不管，如果是最后一步我们下面会处理)
        else:
            # 缓存里没有，必须计算
            print(f"[执行] {step_size}h 模型推理: {current_date_time} -> {next_date_time}")

            # (A) 加载模型
            if model_used != last_model:
                print(f"Loading model: {model_used}...")
                model = onnx.load(model_used)
                options = ort.SessionOptions()
                options.enable_cpu_mem_arena = False
                options.intra_op_num_threads = 30
                cuda_options = {'arena_extend_strategy': 'kSameAsRequested'}
                if use_GPU:
                    ort_session = ort.InferenceSession(model_used, sess_options=options,
                                                       providers=[('CUDAExecutionProvider', cuda_options)])
                else:
                    ort_session = ort.InferenceSession(model_used, sess_options=options,
                                                       providers=['CPUExecutionProvider'])
                last_model = model_used

            # (B) 读取输入 (从 last_source_dir 读取)
            # 如果是第一步，last_source_dir 是 input_dir
            # 如果不是第一步，last_source_dir 肯定是 cache_dir (因为我们保证每一步都存缓存)
            prev_time_str = current_date_time.strftime("%Y-%m-%d-%H-%M")
            if is_first_step:
                input_data = np.load(os.path.join(last_source_dir, 'input_upper.npy')).astype(np.float32)
                input_surface = np.load(os.path.join(last_source_dir, 'input_surface.npy')).astype(np.float32)
            else:
                input_data = np.load(os.path.join(last_source_dir, f'output_upper_{prev_time_str}.npy')).astype(
                    np.float32)
                input_surface = np.load(os.path.join(last_source_dir, f'output_surface_{prev_time_str}.npy')).astype(
                    np.float32)

            # (C) 运行
            output, output_surface = ort_session.run(None, {'input': input_data, 'input_surface': input_surface})

            # (D) 存入缓存 (核心：不管是不是最后一步，都存缓存！)
            np.save(cache_upper_path, output)
            np.save(cache_surface_path, output_surface)

        # 2.3 最终结果处理
        # 如果这是最后一步，把数据存入目标文件夹
        if remain_hour == step_size:
            # 这里我们通过读取缓存（或者复用内存）来保存到 target_dir
            # 最简单的办法：直接利用 copy 逻辑，或者如果刚刚计算完，内存里有 output

            # 为了代码简洁稳健，我们直接把缓存里的文件 copy 过去，或者再 save 一次
            # 这里选择重新 save 一次，确保逻辑简单 (内存里没有的话就 load 缓存再 save)
            target_upper_path = os.path.join(target_dir, f'output_upper_{next_time_str}.npy')
            target_surface_path = os.path.join(target_dir, f'output_surface_{next_time_str}.npy')

            if not os.path.exists(target_upper_path):
                # 如果刚刚算完，output 变量还在内存里，直接存
                if 'output' in locals() and not os.path.exists(cache_upper_path):
                    # 注意：上面的 if else 逻辑里，如果是[跳过]，output变量可能不仅是当前这步的
                    # 为了绝对安全：直接 load 缓存里的，再 save 到 target
                    pass

                    # 统一做法：从缓存 load，存入 target (利用磁盘IO，避免内存变量作用域混乱)
                # 虽然稍微慢一点点，但绝对不会错
                tmp_up = np.load(cache_upper_path)
                tmp_surf = np.load(cache_surface_path)
                np.save(target_upper_path, tmp_up)
                np.save(target_surface_path, tmp_surf)
                print(f"结果已保存至目标文件夹: {target_dir}")

        # 2.4 更新状态
        remain_hour -= step_size
        current_date_time = next_date_time
        last_source_dir = cache_dir  # 下一步的输入一定在缓存里
        is_first_step = False

    # 3. 解码部分 (.nc)
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
