"""Verify runtime dependencies for SCRFD-based data collection."""

import importlib


def verify_setup():
    """Verify that all required components are available."""
    print("=" * 50)
    print("Face Detection CCTV — SCRFD Setup Verification")
    print("=" * 50)

    ok = True

    # 1. Check OpenCV
    try:
        import cv2
        print(f"\n[OK] OpenCV version: {cv2.__version__}")
    except ImportError:
        print("[ERROR] OpenCV not installed!")
        ok = False

    # 2. Check SCRFD runtime stack
    runtime_deps = [
        ("insightface", "InsightFace"),
        ("onnxruntime", "ONNXRuntime"),
        ("supervision", "Supervision (ByteTrack)"),
    ]
    print()
    for module_name, friendly in runtime_deps:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK] {friendly}: {version}")
        except ImportError:
            print(f"[MISSING] {friendly}")
            ok = False

    # 3. Check other dependencies
    deps = ["yaml", "dotenv", "flask", "flask_socketio", "numpy", "PIL"]
    names = ["PyYAML", "python-dotenv", "Flask", "Flask-SocketIO", "NumPy", "Pillow"]
    print()
    for dep, name in zip(deps, names):
        try:
            __import__(dep)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[MISSING] {name}")
            ok = False

    # 4. Summary
    print()
    print("=" * 50)
    if ok:
        print("[✓] All checks passed. Ready to run with SCRFD.")
        print()
        print("  Quick start:")
        print("    python main.py --source webcam     # Test with webcam")
        print("    python main.py                     # Use Hikvision camera")
    else:
        print("[✗] Some checks failed. See errors above.")
        print("  Run: pip install -r requirements.txt")
    print("=" * 50)

    return ok


if __name__ == "__main__":
    verify_setup()