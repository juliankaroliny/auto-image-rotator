# Stage 1: download the pretrained orientation model and convert it to ONNX
FROM python:3.13-slim AS export
RUN pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir timm==1.0.30 onnx==1.23.2 onnxscript==0.7.2
# The checksum makes the build fail if the downloaded file is ever not the expected one
ADD --checksum=sha256:b3362f21f540009da1408a0c2b125ded3b7e748e3614719eb3f4f8796d4c9fd9 \
    https://github.com/ternaus/check_orientation/releases/download/v0.0.3/2020-11-16_resnext50_32x4d.zip /tmp/w.zip
COPY export_onnx.py /tmp/
RUN python -m zipfile -e /tmp/w.zip /tmp/ \
 && python /tmp/export_onnx.py /tmp/2020-11-16_resnext50_32x4d.pth /tmp/orientation.onnx

# Stage 2: small runtime image (no PyTorch)
FROM python:3.13-slim
# The container runs as the calling user, who has no home here. onnxruntime
# wants to write a small file to $HOME and warns on every run otherwise.
ENV HOME=/tmp
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=export /tmp/orientation.onnx .
COPY autorotate.py .
# -u: no output buffering, so progress shows up live (docker run has no TTY)
ENTRYPOINT ["python", "-u", "/app/autorotate.py"]
