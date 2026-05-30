import os
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from datetime import datetime, timedelta
from forecast_decode_functions import surface, upper
# Use GPU or CPU
use_GPU = False
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The date and time of the initial field
date_time = datetime(
    year=2025,
    month=8,
    day=1,
    hour=0,
    minute=0)


# The date and time of the final approaches
date_time_final = datetime(
    year=2025,
    month=8,
    day=4,
    hour=0,
    minute=0)


gdas_data = False


if gdas_data:
    final_result_dir = os.path.join(
        PROJECT_ROOT / "model_output" / "gdas",
        (date_time.strftime("%Y-%m-%d-%H-%M") + "to" + date_time_final.strftime("%Y-%m-%d-%H-%Mgdas_fnl"))
    )
else:
    final_result_dir = os.path.join(
        PROJECT_ROOT / "model_output" / "era5",
        (date_time.strftime("%Y-%m-%d-%H-%M") + "to" + date_time_final.strftime("%Y-%m-%d-%H-%M"))
    )
if not os.path.exists(final_result_dir):
    os.makedirs(final_result_dir)


model_24 = PROJECT_ROOT / "models" / "pangu_weather_24.onnx" # 24h
model_6 = PROJECT_ROOT / "models" / "pangu_weather_6.onnx" # 6h
model_3 = PROJECT_ROOT / "models" / "pangu_weather_3.onnx" # 3h
model_1 = PROJECT_ROOT / "models" / "pangu_weather_1.onnx" # 1h


if gdas_data:
    # The directory for forecasts
    forecast_dir = os.path.join(
        PROJECT_ROOT / "model_input" / "gdas",
        date_time.strftime("%Y-%m-%d-%H-%Mgdas_fnl")
    )
else:
    forecast_dir = os.path.join(
        PROJECT_ROOT / "model_input" / "era5",
        date_time.strftime("%Y-%m-%d-%H-%M")
    )

# Calculate the order of models should be used to generate the final result
time_difference_in_hour = (date_time_final - date_time).total_seconds() / 3600
current_date_time = date_time
last_date_time = None
model_used = None
start = True
ort_session = None
jump = False
while time_difference_in_hour >= 1:
    print(time_difference_in_hour)
    last_model = model_used
    if time_difference_in_hour >= 24:
        model_used = model_24
        time_difference_in_hour -= 24
        current_date_time += timedelta(hours=24)
        print("24")
    elif time_difference_in_hour >= 6:
        model_used = model_6
        time_difference_in_hour -= 6
        current_date_time += timedelta(hours=6)
        print("6")
    elif time_difference_in_hour >= 3:
        model_used = model_3
        time_difference_in_hour -= 3
        current_date_time += timedelta(hours=3)
        print("3")
    elif time_difference_in_hour >= 1:
        model_used = model_1
        time_difference_in_hour -= 1
        current_date_time += timedelta(hours=1)
        print("1")
    if model_used == last_model:
        jump = True
    else:
        jump = False
    print(current_date_time.strftime("%Y-%m-%d-%H-%M"))

    if not jump:
        # Load the model
        model = onnx.load(str(model_used))

        # Set the behavier of onnxruntime
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena=False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        # Increase the number for faster inference and more memory consumption
        options.intra_op_num_threads = 30

        # Set the behavier of cuda provider
        cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

        # Initialize onnxruntime session for Pangu-Weather Models
        if use_GPU:
            ort_session = ort.InferenceSession(str(model_used), sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
        else:
            ort_session = ort.InferenceSession(str(model_used), sess_options=options, providers=['CPUExecutionProvider'])

    print("start")
    # Load the upper-air numpy arrays
    # Load the surface numpy arrays
    input = None
    input_surface = None
    if start:
        input = np.load(os.path.join(forecast_dir, 'input_upper.npy')).astype(np.float32)
        input_surface = np.load(os.path.join(forecast_dir, 'input_surface.npy')).astype(np.float32)
    else:
        input = np.load(os.path.join(final_result_dir, 'output_upper_'+last_date_time.strftime("%Y-%m-%d-%H-%M")+'.npy')).astype(np.float32)
        input_surface = np.load(os.path.join(final_result_dir, 'output_surface_'+last_date_time.strftime("%Y-%m-%d-%H-%M")+'.npy')).astype(np.float32)

    # Run the inference session
    output, output_surface = ort_session.run(None, {'input':input, 'input_surface':input_surface})
    # Save the results
    np.save(os.path.join(final_result_dir, 'output_upper_'+current_date_time.strftime("%Y-%m-%d-%H-%M")), output)
    np.save(os.path.join(final_result_dir, 'output_surface_' + current_date_time.strftime("%Y-%m-%d-%H-%M")), output_surface)
    last_date_time = current_date_time
    start = False

# get all files that need to be decoded
for file in os.listdir(final_result_dir):
    print(file)
    if file.endswith(".npy"):
        if file.startswith("output_surface"):
            # decode surface data
            surface(os.path.join(final_result_dir, file), file[:-4] + ".nc", final_result_dir)
        elif file.startswith("output_upper"):
            # decode upper data
            upper(os.path.join(final_result_dir, file), file[:-4] + ".nc", final_result_dir)
