@echo off
REM Start comfy-serve next to a running ComfyUI (start run_nvidia_gpu.bat first).
REM First time only:  python -m pip install -r requirements.txt
python -m comfy_serve --comfy http://127.0.0.1:8188 --port 8199
pause
