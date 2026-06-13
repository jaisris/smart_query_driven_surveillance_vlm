import os
# Prevent OpenMP conflict between PyTorch (libiomp5md.dll) and FAISS (libomp140.x86_64.dll) on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
