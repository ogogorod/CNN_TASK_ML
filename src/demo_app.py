"""
Interactive demo GUI for the weapon-classification project.

A thin, user-friendly front end over the existing pipeline:
    - infer.py    -> load_model(), predict_image()
    - gradcam.py  -> GradCAM, get_target_layer(), load_image_for_gradcam()

No new dependencies: the UI is built with tkinter (Python standard library);
everything else is already in requirements.txt.

Run from the project root:
    python src/demo_app.py
    python src/demo_app.py --model mobilenetv2 --image path/to/photo.jpg

Controls:
    Open image...   (Ctrl/Cmd+O)  choose a photo and classify it
    Model dropdown                switch between scratch / transfer / mobilenetv2
    Grad-CAM toggle (G)           overlay the heatmap for the predicted class
    Opacity slider                heatmap strength

This script does NOT train anything; it only loads saved checkpoints.
"""

import argparse
import base64
import io
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import torch
from PIL import Image

import config
from gradcam import GradCAM, get_target_layer, load_image_for_gradcam
from infer import MODEL_NAMES, get_checkpoint_path, load_model, predict_image
from utils import get_device


# ---------------------------------------------------------------------------
# Presentation constants
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "scratch": "Custom CNN (from scratch)",
    "transfer": "ResNet18 (transfer learning)",
    "mobilenetv2": "MobileNetV2 (transfer learning)",
}

# Dataset labels are Spanish; show an English hint next to them.
CLASS_HINTS = {
    "billete": "banknote",
    "knife": "knife",
    "monedero": "wallet",
    "pistol": "pistol",
    "smartphone": "smartphone",
    "tarjeta": "card",
}

IMAGE_TYPES = [
    ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.JPG *.JPEG *.PNG"),
    ("All files", "*.*"),
]

VIEW_SIZE = 560  # max width/height of the image panel in pixels

BG = "#16181d"
PANEL = "#1f2229"
TEXT = "#e8eaf0"
MUTED = "#8b91a0"
ACCENT = "#5b8def"
WEAPON = "#e5484d"
SAFE = "#30a46c"
BAR_BG = "#2b2f38"


# ---------------------------------------------------------------------------
# Model / Grad-CAM "brain" helpers (delegate to the project modules)
# ---------------------------------------------------------------------------

class ModelCache:
    """Loads each checkpoint once via infer.load_model() and keeps it around."""

    def __init__(self, device):
        self.device = device
        self._models = {}
        self._lock = threading.Lock()

    def get(self, model_kind: str):
        with self._lock:
            if model_kind not in self._models:
                self._models[model_kind] = load_model(model_kind, self.device)
            return self._models[model_kind]


def compute_gradcam_heatmap(model, model_kind: str, image_path: str, class_idx: int, device):
    """
    Grad-CAM heatmap (values in [0, 1], low resolution) for one image,
    using the project's GradCAM class and target-layer choice.
    """
    target_layer = get_target_layer(model, model_kind)
    cam = GradCAM(model, target_layer)
    try:
        _, _, input_tensor = load_image_for_gradcam(image_path, device)
        with torch.enable_grad():
            heatmap = cam.generate(input_tensor, class_idx)
    finally:
        cam.remove_hooks()
        model.zero_grad(set_to_none=True)
    return heatmap


def blend_heatmap(rgb: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    """
    Overlay a heatmap on an RGB image of any size.

    The model sees the photo squashed to IMG_SIZE x IMG_SIZE, so stretching
    the heatmap back to the photo's own width/height maps it exactly onto
    the original, without distortion.
    """
    h, w = rgb.shape[:2]
    heat = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heat_color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    out = (1 - alpha) * rgb.astype(np.float32) + alpha * heat_color.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def fit_image(image: Image.Image, max_side: int) -> Image.Image:
    """Scale a PIL image to fit in a max_side x max_side box, keeping aspect."""
    w, h = image.size
    scale = min(max_side / w, max_side / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size, Image.LANCZOS)


def to_photo(image: Image.Image) -> tk.PhotoImage:
    """PIL image -> tkinter PhotoImage (via PNG, so PIL.ImageTk is not required)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return tk.PhotoImage(data=base64.b64encode(buf.getvalue()))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class DemoApp:
    def __init__(self, root: tk.Tk, initial_model: str, initial_image: str = None):
        self.root = root
        self.device = get_device()
        self.models = ModelCache(self.device)

        # State
        self.image_path = None
        self.display_rgb = None      # np.ndarray of the photo scaled to the panel
        self.result = None           # dict from infer.predict_image
        self.heatmaps = {}           # (model_kind, image_path) -> heatmap
        self.photo = None            # keep a reference so Tk doesn't GC it
        self.busy = False
        self.jobs = queue.Queue()

        self.model_var = tk.StringVar(value=MODEL_LABELS[initial_model])
        self.gradcam_var = tk.BooleanVar(value=False)
        self.alpha_var = tk.DoubleVar(value=0.45)

        self._build_style()
        self._build_layout()
        self._bind_keys()

        self.root.after(50, self._poll_jobs)
        self._set_status(f"Ready  |  device: {self.device}  |  open a photo to start")

        if initial_image:
            self.root.after(100, lambda: self.load_image(initial_image))

    # ----- construction ---------------------------------------------------

    def _build_style(self):
        self.root.title("Weapon Classifier - Live Demo")
        self.root.configure(bg=BG)
        self.root.minsize(980, 680)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, font=("Helvetica", 12))
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Helvetica", 11))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Helvetica", 18, "bold"))
        style.configure("Big.TLabel", background=PANEL, foreground=TEXT, font=("Helvetica", 22, "bold"))
        style.configure("Status.TLabel", background=BG, foreground=MUTED, font=("Helvetica", 11))
        style.configure(
            "Accent.TButton", background=ACCENT, foreground="white",
            font=("Helvetica", 12, "bold"), padding=(14, 6), borderwidth=0,
        )
        style.map("Accent.TButton", background=[("active", "#4a7be0"), ("disabled", "#3a3f4b")])
        style.configure("TCheckbutton", background=BG, foreground=TEXT, font=("Helvetica", 12, "bold"))
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=TEXT, arrowcolor=TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)])
        style.configure("Horizontal.TScale", background=BG, troughcolor=BAR_BG)

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        # --- top bar ---
        top = ttk.Frame(outer)
        top.pack(fill="x", pady=(0, 12))

        ttk.Label(top, text="Weapon Classifier", style="Title.TLabel").pack(side="left")

        self.open_btn = ttk.Button(top, text="Open image...", style="Accent.TButton", command=self.choose_image)
        self.open_btn.pack(side="right")

        self.model_box = ttk.Combobox(
            top, textvariable=self.model_var, state="readonly", width=30,
            values=[MODEL_LABELS[m] for m in MODEL_NAMES],
        )
        self.model_box.pack(side="right", padx=(0, 12))
        self.model_box.bind("<<ComboboxSelected>>", lambda _e: self.on_model_change())
        ttk.Label(top, text="Model:").pack(side="right", padx=(0, 6))

        # --- body ---
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)

        # image panel
        left = ttk.Frame(body, style="Panel.TFrame", padding=12)
        left.pack(side="left", fill="both", expand=True)

        self.image_label = tk.Label(
            left, bg=PANEL, fg=MUTED, font=("Helvetica", 14),
            text="No image loaded\n\nClick \"Open image...\" or press Ctrl/Cmd+O",
            width=VIEW_SIZE, height=VIEW_SIZE,
        )
        self.image_label.pack(fill="both", expand=True)
        self.image_label.bind("<Button-1>", lambda _e: self.choose_image())

        # Grad-CAM controls under the image
        cam_row = ttk.Frame(left, style="Panel.TFrame")
        cam_row.pack(fill="x", pady=(10, 0))

        self.cam_check = ttk.Checkbutton(
            cam_row, text="Show Grad-CAM", variable=self.gradcam_var, command=self.on_gradcam_toggle,
        )
        self.cam_check.pack(side="left")

        ttk.Label(cam_row, text="opacity", style="Muted.TLabel").pack(side="left", padx=(18, 6))
        self.alpha_scale = ttk.Scale(
            cam_row, from_=0.1, to=0.8, variable=self.alpha_var, length=160,
            command=lambda _v: self.refresh_view(),
        )
        self.alpha_scale.pack(side="left")

        self.view_caption = ttk.Label(cam_row, text="", style="Muted.TLabel")
        self.view_caption.pack(side="right")

        # results panel
        right = ttk.Frame(body, style="Panel.TFrame", padding=18, width=340)
        right.pack(side="left", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        self.verdict = tk.Label(
            right, text="-", bg=BAR_BG, fg=TEXT, font=("Helvetica", 20, "bold"), pady=14,
        )
        self.verdict.pack(fill="x")

        ttk.Label(right, text="Predicted class", style="Muted.TLabel").pack(anchor="w", pady=(18, 0))
        self.pred_label = ttk.Label(right, text="-", style="Big.TLabel")
        self.pred_label.pack(anchor="w")
        self.conf_label = ttk.Label(right, text="", style="Muted.TLabel")
        self.conf_label.pack(anchor="w")

        ttk.Label(right, text="Class probabilities", style="Muted.TLabel").pack(anchor="w", pady=(20, 6))
        self.bars = tk.Canvas(right, bg=PANEL, highlightthickness=0, height=6 * 36, width=300)
        self.bars.pack(fill="x")

        self.weapon_score_label = ttk.Label(right, text="", style="Muted.TLabel")
        self.weapon_score_label.pack(anchor="w", pady=(14, 0))
        self.timing_label = ttk.Label(right, text="", style="Muted.TLabel")
        self.timing_label.pack(anchor="w")

        self._draw_bars(None)

        # --- status bar ---
        self.status = ttk.Label(outer, text="", style="Status.TLabel")
        self.status.pack(fill="x", pady=(10, 0))

    def _bind_keys(self):
        for seq in ("<Control-o>", "<Command-o>"):
            try:
                self.root.bind(seq, lambda _e: self.choose_image())
            except tk.TclError:
                pass  # <Command-...> only exists on macOS
        self.root.bind("<g>", lambda _e: self._toggle_gradcam_key())
        self.root.bind("<G>", lambda _e: self._toggle_gradcam_key())

    # ----- background work ------------------------------------------------

    def _run_in_background(self, work, on_done, status_text):
        """Run torch work off the UI thread; deliver the result via a queue."""
        if self.busy:
            return
        self.busy = True
        self._set_controls_enabled(False)
        self._set_status(status_text)

        def target():
            try:
                self.jobs.put((on_done, work(), None))
            except Exception as exc:  # surface errors in the UI
                self.jobs.put((on_done, None, exc))

        threading.Thread(target=target, daemon=True).start()

    def _poll_jobs(self):
        try:
            while True:
                on_done, value, error = self.jobs.get_nowait()
                self.busy = False
                self._set_controls_enabled(True)
                if error is not None:
                    self._set_status(f"Error: {error}")
                    messagebox.showerror("Something went wrong", str(error))
                else:
                    on_done(value)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_jobs)

    # ----- actions --------------------------------------------------------

    @property
    def model_kind(self) -> str:
        label = self.model_var.get()
        return next(k for k, v in MODEL_LABELS.items() if v == label)

    def choose_image(self):
        if self.busy:
            return
        start_dir = os.path.dirname(self.image_path) if self.image_path else config.PROJECT_ROOT
        path = filedialog.askopenfilename(
            title="Choose a photo to classify", initialdir=start_dir, filetypes=IMAGE_TYPES,
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Cannot open image", f"{path}\n\n{exc}")
            return

        self.image_path = path
        self.display_rgb = np.array(fit_image(pil, VIEW_SIZE))
        self.result = None
        self.refresh_view()
        self.run_inference()

    def on_model_change(self):
        if self.image_path:
            self.run_inference()

    def _toggle_gradcam_key(self):
        if self.busy:
            return
        self.gradcam_var.set(not self.gradcam_var.get())
        self.on_gradcam_toggle()

    def on_gradcam_toggle(self):
        if self.gradcam_var.get() and self.result is not None:
            self.ensure_heatmap()
        else:
            self.refresh_view()

    def run_inference(self):
        model_kind = self.model_kind
        image_path = self.image_path
        checkpoint = get_checkpoint_path(model_kind)
        if not os.path.exists(checkpoint):
            messagebox.showerror(
                "Checkpoint missing",
                f"Could not find:\n{checkpoint}\n\n"
                "Download the checkpoints (see README) into outputs/checkpoints/.",
            )
            return

        def work():
            model = self.models.get(model_kind)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = predict_image(model, image_path, self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
            result["model_kind"] = model_kind
            return result

        def done(result):
            # Ignore stale results if the user changed image/model meanwhile.
            if result["image"] != self.image_path:
                return
            self.result = result
            self.show_result(result)
            if self.gradcam_var.get():
                self.ensure_heatmap()
            else:
                self.refresh_view()
            self._set_status(
                f"{MODEL_LABELS[model_kind]}  |  {os.path.basename(image_path)}  |  device: {self.device}"
            )

        self._run_in_background(work, done, f"Running {MODEL_LABELS[model_kind]}...")

    def ensure_heatmap(self):
        result = self.result
        key = (result["model_kind"], result["image"])
        if key in self.heatmaps:
            self.refresh_view()
            return

        def work():
            model = self.models.get(key[0])
            return compute_gradcam_heatmap(model, key[0], key[1], result["predicted_index"], self.device)

        def done(heatmap):
            self.heatmaps[key] = heatmap
            self.refresh_view()
            self._set_status(
                f"Grad-CAM for '{result['predicted_class']}'  |  {MODEL_LABELS[key[0]]}  |  "
                "red = regions that drove the prediction"
            )

        self._run_in_background(work, done, "Computing Grad-CAM...")

    # ----- drawing --------------------------------------------------------

    def refresh_view(self):
        if self.display_rgb is None:
            return

        rgb = self.display_rgb
        caption = "original"
        if self.gradcam_var.get() and self.result is not None:
            key = (self.result["model_kind"], self.result["image"])
            heatmap = self.heatmaps.get(key)
            if heatmap is not None:
                rgb = blend_heatmap(rgb, heatmap, float(self.alpha_var.get()))
                caption = f"Grad-CAM -> {self.result['predicted_class']}"

        self.photo = to_photo(Image.fromarray(rgb))
        self.image_label.configure(image=self.photo, text="", width=VIEW_SIZE, height=VIEW_SIZE)
        self.view_caption.configure(text=caption)

    def show_result(self, result):
        cls = result["predicted_class"]
        if result["is_weapon"]:
            self.verdict.configure(text="WEAPON DETECTED", bg=WEAPON, fg="white")
        else:
            self.verdict.configure(text="NO WEAPON", bg=SAFE, fg="white")

        hint = CLASS_HINTS.get(cls, cls)
        self.pred_label.configure(text=cls if hint == cls else f"{cls}  ({hint})")
        self.conf_label.configure(text=f"confidence {result['confidence'] * 100:.1f}%")
        self.weapon_score_label.configure(
            text=f"weapon score (knife + pistol): {result['weapon_score'] * 100:.1f}%"
        )
        self.timing_label.configure(
            text=f"inference {result['elapsed_ms']:.0f} ms on {self.device.type}"
        )
        self._draw_bars(result)

    def _draw_bars(self, result):
        c = self.bars
        c.delete("all")
        width = int(c.cget("width"))
        label_w, row_h, bar_h = 110, 36, 14
        bar_w = width - label_w - 56

        for i, cls in enumerate(config.CLASS_NAMES):
            y = i * row_h + 8
            p = result["class_probs"][cls] if result else 0.0
            is_pred = result is not None and cls == result["predicted_class"]
            color = (WEAPON if cls in config.WEAPON_CLASSES else ACCENT) if is_pred else "#59606f"

            c.create_text(
                0, y + bar_h / 2, text=cls, anchor="w",
                fill=TEXT if is_pred else MUTED,
                font=("Helvetica", 12, "bold" if is_pred else "normal"),
            )
            c.create_rectangle(label_w, y, label_w + bar_w, y + bar_h, fill=BAR_BG, width=0)
            if p > 0:
                c.create_rectangle(label_w, y, label_w + max(2, bar_w * p), y + bar_h, fill=color, width=0)
            c.create_text(
                width, y + bar_h / 2, text=f"{p * 100:.1f}%" if result else "", anchor="e",
                fill=TEXT if is_pred else MUTED, font=("Helvetica", 11),
            )

    # ----- small helpers --------------------------------------------------

    def _set_status(self, text: str):
        self.status.configure(text=text)

    def _set_controls_enabled(self, enabled: bool):
        state = "!disabled" if enabled else "disabled"
        self.open_btn.state([state])
        self.cam_check.state([state])
        self.model_box.configure(state="readonly" if enabled else "disabled")
        self.root.configure(cursor="" if enabled else "watch")


def main():
    parser = argparse.ArgumentParser(description="GUI demo: inference + Grad-CAM.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="mobilenetv2", help="Model selected at start.")
    parser.add_argument("--image", default=None, help="Optional image to open at start.")
    args = parser.parse_args()

    root = tk.Tk()
    DemoApp(root, initial_model=args.model, initial_image=args.image)
    root.mainloop()


if __name__ == "__main__":
    main()
