import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os
import glob
import shutil
from datetime import datetime

class VideoDetSegLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Auto-Tracking & Frame Labeler")
        
        # Folder & File State
        self.base_dir = None
        self.video_list = []
        self.video_filenames = []
        self.current_video_idx = -1
        
        # Playback & Extraction State
        self.is_extracted = False
        self.extracted_folder = None
        self.cap = None
        self.image_list = []
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self.play_job = None
        
        # Image Display State
        self.orig_img = None  
        self.tk_image = None
        self.img_width = 0
        self.img_height = 0
        self.scale = 1.0      
        
        # Annotation & Class State
        self.video_annotations = {} 
        self.current_frame_annotations = [] 
        self.current_id = 1
        self.categories = {}  
        self.is_dirty = False 
        
        # Drawing State
        self.mode_var = tk.StringVar(value="bbox")
        self.start_raw_x = None
        self.start_raw_y = None
        self.current_rect = None
        self.current_polygon_points = []
        
        self.setup_ui()
        self.setup_shortcuts()
        
    def setup_ui(self):
        # --- Top Control Panel (File & Navigation) ---
        top_frame = tk.Frame(self.root, padx=10, pady=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(top_frame, text="Load Video Folder", command=self.load_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Load Classes", command=self.load_categories_file).pack(side=tk.LEFT, padx=5)
        
        tk.Label(top_frame, text=" | ").pack(side=tk.LEFT)
        
        tk.Button(top_frame, text="<< Prev Video", command=self.prev_video).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Next Video >>", command=self.next_video).pack(side=tk.LEFT, padx=0)
        
        self.video_combo_var = tk.StringVar()
        self.video_combo = ttk.Combobox(top_frame, textvariable=self.video_combo_var, state="readonly", width=30)
        self.video_combo.pack(side=tk.LEFT, padx=10)
        self.video_combo.bind("<<ComboboxSelected>>", self.on_video_select)
        
        tk.Label(top_frame, text="Category:").pack(side=tk.LEFT, padx=5)
        self.category_var = tk.StringVar(value="1 - Default")
        self.category_combo = ttk.Combobox(top_frame, textvariable=self.category_var, state="readonly", width=12)
        self.category_combo.pack(side=tk.LEFT)
        
        tk.Label(top_frame, text=" Mode:").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(top_frame, text="BBox", variable=self.mode_var, value="bbox").pack(side=tk.LEFT)
        tk.Radiobutton(top_frame, text="Mask", variable=self.mode_var, value="mask").pack(side=tk.LEFT)

        self.status_label = tk.Label(top_frame, text="Ready", fg="green", font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.RIGHT, padx=15)

        # --- Middle Control Panel (Timeline & Actions) ---
        mid_frame = tk.Frame(self.root, padx=10, pady=5, bg="#e0e0e0")
        mid_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.play_btn = tk.Button(mid_frame, text="▶ Play", width=8, command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Button(mid_frame, text="⏮", width=3, command=lambda: self.step_frame(-1)).pack(side=tk.LEFT)
        tk.Button(mid_frame, text="⏭", width=3, command=lambda: self.step_frame(1)).pack(side=tk.LEFT, padx=(0, 10))
        
        self.slider_var = tk.IntVar(value=0)
        self.slider = tk.Scale(mid_frame, variable=self.slider_var, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=False, command=self.on_slider_change, bg="#e0e0e0", length=400)
        self.slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        self.frame_lbl = tk.Label(mid_frame, text="Frame: 0 / 0", bg="#e0e0e0", width=15)
        self.frame_lbl.pack(side=tk.LEFT, padx=5)
        
        self.auto_track_btn = tk.Button(mid_frame, text="Auto-Track Video", bg="#d9edf7", command=self.auto_track_forward)
        self.auto_track_btn.pack(side=tk.LEFT, padx=10)
        
        self.remove_btn = tk.Button(mid_frame, text="Remove Extracted Folder", bg="#f2dede", fg="red", command=self.remove_extraction)
        
        self.save_btn = tk.Button(mid_frame, text="Extract Frames & Save (S)", bg="#dff0d8", font=("Arial", 9, "bold"), command=self.handle_save_shortcut)
        self.save_btn.pack(side=tk.RIGHT, padx=5)
        
        tk.Button(mid_frame, text="Clear Last (L)", command=self.clear_last).pack(side=tk.RIGHT, padx=5)

        # --- Main Layout (Canvas + Sidebar) ---
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        canvas_container = tk.Frame(self.main_pane)
        self.main_pane.add(canvas_container, weight=5) 
        
        self.h_scroll = tk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll = tk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(canvas_container, cursor="crosshair", xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set, bg="#333333")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.h_scroll.config(command=self.canvas.xview)
        self.v_scroll.config(command=self.canvas.yview)

        self.sidebar = tk.Frame(self.main_pane, bg="#f0f0f0", width=300) 
        self.main_pane.add(self.sidebar, weight=1) 

        self.mode_lbl = tk.Label(self.sidebar, text="MODE: None", bg="#ddd", fg="black", font=("Arial", 11, "bold"))
        self.mode_lbl.pack(fill=tk.X, pady=(0, 5), ipady=5)
        
        tk.Label(self.sidebar, text="Current Frame Annotations", bg="#ddd", fg="black", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(5, 5))
        self.create_scrollable_sidebar()

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-2>", self.on_right_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<MouseWheel>", self.zoom_mousewheel) 
        self.canvas.bind("<Button-4>", self.zoom_in)           
        self.canvas.bind("<Button-5>", self.zoom_out)          

    def create_scrollable_sidebar(self):
        self.labels_canvas = tk.Canvas(self.sidebar, bg="#f0f0f0", highlightthickness=0)
        scrollbar = tk.Scrollbar(self.sidebar, orient="vertical", command=self.labels_canvas.yview)
        self.labels_inner_frame = tk.Frame(self.labels_canvas, bg="#f0f0f0")

        self.labels_inner_frame.bind("<Configure>", lambda e: self.labels_canvas.configure(scrollregion=self.labels_canvas.bbox("all")))
        self.labels_canvas.create_window((0, 0), window=self.labels_inner_frame, anchor="nw")
        self.labels_canvas.configure(yscrollcommand=scrollbar.set)
        self.labels_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_shortcuts(self):
        self.root.bind("<a>", lambda e: self.step_frame(-1))
        self.root.bind("<d>", lambda e: self.step_frame(1))
        self.root.bind("<s>", self.handle_save_shortcut)
        self.root.bind("<S>", self.handle_save_shortcut)
        self.root.bind("<l>", self.clear_last)
        self.root.bind("<space>", lambda e: self.toggle_play())

    def set_dirty(self, state):
        self.is_dirty = state
        self.status_label.config(text="● Unsaved Changes" if state else "● Saved", fg="red" if state else "green")

    def get_selected_cat_id(self, val_str):
        try: return int(val_str.split(" - ")[0])
        except: return 0

    def load_categories_file(self):
        filepath = filedialog.askopenfilename(title="Select Categories File", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not filepath: return
        self._parse_categories(filepath)
        
    def _parse_categories(self, filepath):
        self.categories.clear()
        try:
            with open(filepath, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    line = line.strip()
                    if not line: continue
                    if '=' in line:
                        parts = line.split('=')
                        try:
                            self.categories[int(parts[1].strip())] = parts[0].strip()
                        except ValueError:
                            self.categories[int(parts[0].strip())] = parts[1].strip()
                    else:
                        self.categories[i] = line
            
            cat_list = [f"{k} - {v}" for k, v in self.categories.items()]
            self.category_combo['values'] = cat_list
            if cat_list: self.category_combo.current(0)
            self.refresh_sidebar_labels()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load categories:\n{e}")

    def load_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Videos")
        if not folder_path: return
        self.base_dir = folder_path
        
        for cat_file in ["categories.txt", "classes.txt"]:
            cat_path = os.path.join(folder_path, cat_file)
            if os.path.exists(cat_path):
                self._parse_categories(cat_path)
                break
        
        video_exts = {".mp4", ".avi", ".mkv", ".mov"}
        self.video_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in video_exts])
        
        if not self.video_list:
            messagebox.showwarning("No Videos", "No valid video files found in the selected folder.")
            return
            
        self.video_filenames = [os.path.basename(p) for p in self.video_list]
        self.video_combo['values'] = self.video_filenames
        
        self.current_video_idx = 0
        self.load_current_video()

    def on_video_select(self, event=None):
        selected_idx = self.video_combo.current()
        if selected_idx != self.current_video_idx and selected_idx != -1:
            self.current_video_idx = selected_idx
            self.load_current_video()

    def next_video(self):
        if self.current_video_idx < len(self.video_list) - 1:
            self.current_video_idx += 1
            self.load_current_video()

    def prev_video(self):
        if self.current_video_idx > 0:
            self.current_video_idx -= 1
            self.load_current_video()

    def load_current_video(self):
        if self.is_playing: self.toggle_play()
        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_combo.current(self.current_video_idx)
        video_path = self.video_list[self.current_video_idx]
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        self.extracted_folder = os.path.join(self.base_dir, f"{base_name}_extracted")
        self.current_frame = 0
        self.current_polygon_points = []
        self.current_frame_annotations = []
        self.scale = 1.0

        if os.path.exists(self.extracted_folder):
            self.is_extracted = True
            self.image_list = sorted(glob.glob(os.path.join(self.extracted_folder, "*.jpg")))
            self.total_frames = len(self.image_list)
            
            self.mode_lbl.config(text="MODE: EXTRACTED FRAMES", bg="#dff0d8", fg="#3c763d")
            self.auto_track_btn.config(state=tk.DISABLED)
            self.play_btn.config(state=tk.DISABLED)
            
            self.remove_btn.pack(side=tk.LEFT, padx=5)
            self.save_btn.config(text="Save JSON (S)")
            
        else:
            self.is_extracted = False
            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_annotations = {}
            
            self.mode_lbl.config(text="MODE: VIDEO ANNOTATION", bg="#fcf8e3", fg="#8a6d3b")
            self.remove_btn.pack_forget()
            self.auto_track_btn.config(state=tk.NORMAL)
            self.play_btn.config(state=tk.NORMAL)
            
            self.save_btn.config(text="Extract Frames & Save (S)")

        self.slider.config(to=max(0, self.total_frames - 1))
        self.slider_var.set(0)
        self.update_canvas_for_frame()

    def toggle_play(self):
        if self.is_extracted: return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.config(text="⏸ Pause")
            self.play_loop()
        else:
            self.play_btn.config(text="▶ Play")
            if self.play_job:
                self.root.after_cancel(self.play_job)
                self.play_job = None

    def play_loop(self):
        if self.is_playing and self.current_frame < self.total_frames - 1:
            self.step_frame(1)
            fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30
            delay = int(1000 / fps) if fps > 0 else 33
            self.play_job = self.root.after(delay, self.play_loop)
        else:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")

    def step_frame(self, direction):
        new_frame = self.current_frame + direction
        if 0 <= new_frame < self.total_frames:
            self.slider_var.set(new_frame)
            self.on_slider_change(new_frame)

    def on_slider_change(self, val):
        self.sync_current_annotations()
        self.current_frame = int(float(val))
        self.current_polygon_points = []
        self.update_canvas_for_frame()

    def update_canvas_for_frame(self):
        self.frame_lbl.config(text=f"Frame: {self.current_frame + 1} / {self.total_frames}")
        
        if self.is_extracted:
            if not self.image_list: return
            img_path = self.image_list[self.current_frame]
            cv_img = cv2.imread(img_path)
            
            self.current_frame_annotations = []
            json_path = os.path.splitext(img_path)[0] + ".json"
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    self.current_frame_annotations = data.get("annotations", [])
                except: pass
        else:
            if not self.cap: return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, cv_img = self.cap.read()
            if not ret: return
            self.current_frame_annotations = self.video_annotations.get(self.current_frame, []).copy()

        self.current_id = max([a.get("id", 0) for a in self.current_frame_annotations] + [0]) + 1
            
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.orig_img = Image.fromarray(cv_img_rgb)
        self.img_width, self.img_height = self.orig_img.size
        
        self.refresh_sidebar_labels()
        self.redraw_canvas()
        self.set_dirty(False)

    def sync_current_annotations(self):
        if not self.is_extracted:
            if self.current_frame_annotations:
                self.video_annotations[self.current_frame] = self.current_frame_annotations.copy()
            elif self.current_frame in self.video_annotations:
                del self.video_annotations[self.current_frame]

    def get_color_for_class(self, cat_id):
        colors = ["cyan", "yellow", "lime", "magenta", "orange", "red", "dodgerblue", "pink", "gold", "purple"]
        return colors[(cat_id) % len(colors)]

    def redraw_canvas(self):
        self.canvas.delete("all")
        if not self.orig_img: return

        new_w = max(1, int(self.img_width * self.scale))
        new_h = max(1, int(self.img_height * self.scale))
        resized_img = self.orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_img)

        self.canvas.config(scrollregion=(0, 0, new_w, new_h))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        for ann in self.current_frame_annotations:
            cat_id = ann.get("category_id", 0)
            cat_name = self.categories.get(cat_id, f"C:{cat_id}")
            label_text = f"ID:{ann['id']} {cat_name}"
            color = self.get_color_for_class(cat_id)

            if ann.get("type", "bbox") == "bbox" and "xmin" in ann:
                sx, sy = ann["xmin"] * self.scale, ann["ymin"] * self.scale
                ex, ey = (ann["xmin"] + ann["width"]) * self.scale, (ann["ymin"] + ann["height"]) * self.scale
                self.canvas.create_rectangle(sx, sy, ex, ey, outline=color, width=2)
                self.canvas.create_text(sx, sy-10, text=label_text, fill=color, anchor=tk.W)

            elif ann.get("type") == "segmentation" and "segmentation" in ann:
                scaled_pts = [p * self.scale for p in ann["segmentation"][0]]
                if len(scaled_pts) >= 6:
                    self.canvas.create_polygon(scaled_pts, outline=color, fill="", width=2)
                    self.canvas.create_text(scaled_pts[0], scaled_pts[1]-10, text=label_text, fill=color, anchor=tk.W)

        if self.current_polygon_points:
            cat_id = self.get_selected_cat_id(self.category_var.get())
            color = self.get_color_for_class(cat_id)
            scaled_pts = [p * self.scale for p in self.current_polygon_points]
            for i in range(0, len(scaled_pts), 2):
                x, y = scaled_pts[i], scaled_pts[i+1]
                self.canvas.create_oval(x-2, y-2, x+2, y+2, fill=color, outline=color, tags="temp_poly")
                if i >= 2:
                    px, py = scaled_pts[i-2], scaled_pts[i-1]
                    self.canvas.create_line(px, py, x, y, fill=color, width=2, tags="temp_poly")

    def on_press(self, event):
        if not self.orig_img: return
        raw_x, raw_y = self.get_raw_coords(event)
        
        cat_id = self.get_selected_cat_id(self.category_var.get())
        color = self.get_color_for_class(cat_id)
        
        if self.mode_var.get() == "bbox":
            self.start_raw_x = raw_x
            self.start_raw_y = raw_y
            sx, sy = raw_x * self.scale, raw_y * self.scale
            self.current_rect = self.canvas.create_rectangle(sx, sy, sx, sy, outline=color, width=2)
            
        elif self.mode_var.get() == "mask":
            if len(self.current_polygon_points) >= 6:
                first_x, first_y = self.current_polygon_points[0], self.current_polygon_points[1]
                dist = ((raw_x - first_x)**2 + (raw_y - first_y)**2) ** 0.5
                if dist * self.scale <= 15:
                    self.close_mask() 
                    return

            self.current_polygon_points.extend([raw_x, raw_y])
            self.redraw_canvas() 

    def close_mask(self):
        if self.mode_var.get() == "mask" and len(self.current_polygon_points) >= 6:
            self.current_frame_annotations.append({
                "id": self.current_id,
                "category_id": self.get_selected_cat_id(self.category_var.get()),
                "segmentation": [self.current_polygon_points],
                "type": "segmentation"
            })
            self.current_id += 1
            self.current_polygon_points = []
            self.sync_current_annotations()
            self.refresh_sidebar_labels()
            self.redraw_canvas()
            self.set_dirty(True) 

    def on_right_click(self, event=None):
        if not event or not hasattr(event, 'num') or event.num not in (2, 3) or not self.orig_img:
            return
        self.close_mask()

    def on_drag(self, event):
        if not self.orig_img or self.mode_var.get() != "bbox" or self.start_raw_x is None: return
        raw_x, raw_y = self.get_raw_coords(event)
        sx, sy = self.start_raw_x * self.scale, self.start_raw_y * self.scale
        ex, ey = raw_x * self.scale, raw_y * self.scale
        self.canvas.coords(self.current_rect, sx, sy, ex, ey)

    def on_release(self, event):
        if not self.orig_img or self.mode_var.get() != "bbox" or self.start_raw_x is None: return
        raw_x, raw_y = self.get_raw_coords(event)
        
        xmin, ymin = min(self.start_raw_x, raw_x), min(self.start_raw_y, raw_y)
        xmax, ymax = max(self.start_raw_x, raw_x), max(self.start_raw_y, raw_y)
        width, height = xmax - xmin, ymax - ymin
        
        if width > 5 and height > 5:
            self.current_frame_annotations.append({
                "id": self.current_id,
                "category_id": self.get_selected_cat_id(self.category_var.get()),
                "xmin": int(xmin),
                "ymin": int(ymin),
                "width": int(width),
                "height": int(height),
                "type": "bbox"
            })
            self.current_id += 1
            self.sync_current_annotations()
            self.refresh_sidebar_labels()
            self.set_dirty(True) 
            
        self.start_raw_x = None
        self.start_raw_y = None
        self.redraw_canvas()

    def clear_last(self, event=None):
        if self.current_polygon_points:
            self.current_polygon_points = []
            self.redraw_canvas()
        elif self.current_frame_annotations:
            self.current_frame_annotations.pop()
            self.sync_current_annotations()
            self.refresh_sidebar_labels()
            self.redraw_canvas()
            self.set_dirty(True)

    def delete_annotation(self, ann_id):
        self.current_frame_annotations = [a for a in self.current_frame_annotations if a["id"] != ann_id]
        self.sync_current_annotations()
        self.refresh_sidebar_labels()
        self.redraw_canvas()
        self.set_dirty(True)

    def refresh_sidebar_labels(self):
        for widget in self.labels_inner_frame.winfo_children():
            widget.destroy()
            
        for ann in self.current_frame_annotations:
            row = tk.Frame(self.labels_inner_frame, bg="#f0f0f0", pady=2)
            row.pack(fill=tk.X, padx=5)
            
            cat_name = self.categories.get(ann.get("category_id", 0), "Default")
            t = "Box" if ann.get("type") == "bbox" else "Mask"
            
            lbl = tk.Label(row, text=f"{t} {ann['id']} - {cat_name}", fg="black", bg="#f0f0f0", anchor=tk.W)
            lbl.pack(side=tk.LEFT)
            
            btn = tk.Button(row, text=" X ", fg="red", font=("Arial", 10, "bold"), command=lambda id=ann['id']: self.delete_annotation(id))
            btn.pack(side=tk.RIGHT)

    # --- Video Tracking & SSIM ---
    def _get_crop(self, img, bbox):
        x, y, w, h = [int(v) for v in bbox]
        x, y = max(0, x), max(0, y)
        w, h = min(img.shape[1] - x, w), min(img.shape[0] - y, h)
        if w <= 0 or h <= 0: return None
        return img[y:y+h, x:x+w]

    def _compute_ssim(self, im1, im2):
        if im1 is None or im2 is None or im1.size == 0 or im2.size == 0: return 0.0
        if len(im1.shape) == 3: im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        if len(im2.shape) == 3: im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        if im1.shape != im2.shape: im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
        
        im1, im2 = im1.astype(np.float64), im2.astype(np.float64)
        C1, C2 = 6.5025, 58.5225 
        mu1, mu2 = cv2.GaussianBlur(im1, (11, 11), 1.5), cv2.GaussianBlur(im2, (11, 11), 1.5)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
        sigma1_sq = cv2.GaussianBlur(im1**2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(im2**2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(im1*im2, (11, 11), 1.5) - mu1_mu2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def _create_tracker(self):
        for t_name in ['TrackerCSRT_create', 'TrackerKCF_create', 'TrackerMIL_create']:
            if hasattr(cv2, t_name): return getattr(cv2, t_name)()
            elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, t_name): return getattr(cv2.legacy, t_name)()
        return None

    def _init_trackers(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = self.cap.read()
        if not ret: return []
        
        anns = self.current_frame_annotations if frame_idx == self.current_frame else self.video_annotations.get(frame_idx, [])
        trackers = []
        
        for ann in anns:
            if ann.get("type", "bbox") != "bbox": continue 
            
            tracker = self._create_tracker()
            if not tracker: return []
            
            bbox = (ann["xmin"], ann["ymin"], ann["width"], ann["height"])
            tracker.init(img, bbox)
            crop = self._get_crop(img, bbox)
            
            trackers.append({
                "tracker": tracker, "cat_id": ann["category_id"], "id": ann["id"], 
                "prev_img": img, "prev_bbox": bbox, "prev_crop": crop
            })
            
        return trackers

    def auto_track_forward(self):
        if self.is_extracted or not self.cap: return
        self.sync_current_annotations()
        
        # 1. ALWAYS start tracker from the very first frame globally
        track_idx = 0
        trackers = self._init_trackers(track_idx)
            
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Auto-Tracking...")
        tk.Label(progress_win, text="Tracking objects across video...").pack(pady=10)
        progress_bar = ttk.Progressbar(progress_win, maximum=self.total_frames, length=300)
        progress_bar.pack(padx=20, pady=10)
        
        tracked_count = 0
        
        while track_idx < self.total_frames - 1:
            next_idx = track_idx + 1
            
            if not trackers:
                # Scrub forward to find the next frame with a manual label to re-initialize
                next_labeled_idx = -1
                for i in range(next_idx, self.total_frames):
                    if i in self.video_annotations and any(a.get("type", "bbox") == "bbox" for a in self.video_annotations[i]):
                        next_labeled_idx = i
                        break
                        
                if next_labeled_idx != -1:
                    track_idx = next_labeled_idx
                    trackers = self._init_trackers(track_idx)
                    progress_bar['value'] = track_idx
                    progress_win.update()
                    continue 
                else:
                    break 
                    
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
            ret, next_img = self.cap.read()
            if not ret: break
            
            active_trackers = []
            new_annotations = []
            
            for t_info in trackers:
                success, bbox = t_info["tracker"].update(next_img)
                x, y, w, h = [int(v) for v in bbox]
                
                # 1. Dimensional / Edge Verification
                margin = 5
                if success:
                    pw, ph = t_info["prev_bbox"][2], t_info["prev_bbox"][3]
                    if w < 10 or h < 10 or x < margin or y < margin or (x + w) > (self.img_width - margin) or (y + h) > (self.img_height - margin):
                        success = False
                    # Force failure if box suddenly halves or doubles in size
                    elif w < pw * 0.5 or w > pw * 2.0 or h < ph * 0.5 or h > ph * 2.0:
                        success = False

                # 2. SSIM Content Verification (Consecutive Frame Check)
                if success:
                    current_crop = self._get_crop(next_img, (x, y, w, h))
                    ssim_val = self._compute_ssim(t_info["prev_crop"], current_crop)
                    if ssim_val < 0.65:  
                        success = False

                # 3. Surroundings Search via Shape Pattern Matching
                if not success and t_info["prev_crop"] is not None:
                    px, py, pw, ph = [int(v) for v in t_info["prev_bbox"]]
                    
                    sx = max(0, px - pw * 2)
                    sy = max(0, py - ph * 2)
                    ex = min(self.img_width, px + pw * 3)
                    ey = min(self.img_height, py + ph * 3)
                    
                    search_area = next_img[sy:ey, sx:ex]
                    template = t_info["prev_crop"]
                    
                    if search_area.shape[0] >= template.shape[0] and search_area.shape[1] >= template.shape[1]:
                        res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        
                        if max_val > 0.80: 
                            nx = sx + max_loc[0]
                            ny = sy + max_loc[1]
                            # Check valid location within constraints
                            if not (pw < 10 or ph < 10 or nx < margin or ny < margin or (nx + pw) > (self.img_width - margin) or (ny + ph) > (self.img_height - margin)):
                                success = True
                                x, y, w, h = nx, ny, pw, ph
                                t_info["tracker"] = self._create_tracker()
                                t_info["tracker"].init(next_img, (x, y, w, h))

                if success:
                    new_annotations.append({
                        "id": t_info["id"], "category_id": t_info["cat_id"],
                        "xmin": x, "ymin": y, "width": w, "height": h, "type": "bbox"
                    })
                    t_info["prev_img"] = next_img
                    t_info["prev_bbox"] = (x, y, w, h)
                    t_info["prev_crop"] = self._get_crop(next_img, (x, y, w, h))
                    active_trackers.append(t_info)
                else:
                    # Object permanently lost. Drop it from all future frames
                    lost_id = t_info["id"]
                    for future_idx in range(next_idx, self.total_frames):
                        if future_idx in self.video_annotations:
                            self.video_annotations[future_idx] = [a for a in self.video_annotations[future_idx] if a["id"] != lost_id]

            # 4. Detect any NEWLY hand-drawn objects on this specific frame
            existing = self.video_annotations.get(next_idx, [])
            active_ids = {t["id"] for t in active_trackers}
            
            for manual_ann in existing:
                if manual_ann.get("type", "bbox") == "bbox" and manual_ann["id"] not in active_ids:
                    # A new object was found! Initialize a tracker for it so it propagates forward.
                    nt = self._create_tracker()
                    if nt:
                        nbox = (manual_ann["xmin"], manual_ann["ymin"], manual_ann["width"], manual_ann["height"])
                        nt.init(next_img, nbox)
                        active_trackers.append({
                            "tracker": nt, "cat_id": manual_ann["category_id"], "id": manual_ann["id"],
                            "prev_img": next_img, "prev_bbox": nbox,
                            "prev_crop": self._get_crop(next_img, nbox)
                        })

            trackers = active_trackers
            
            # Merge tracked annotations with existing ones
            if new_annotations:
                existing_ids = {a["id"] for a in existing}
                for na in new_annotations:
                    if na["id"] not in existing_ids: existing.append(na)
                self.video_annotations[next_idx] = existing
                tracked_count += 1
                
            track_idx = next_idx
            
            if track_idx % 2 == 0:
                progress_bar['value'] = track_idx
                progress_win.update()

        progress_win.destroy()
        messagebox.showinfo("Tracking Complete", f"Successfully tracked and saved labels for {tracked_count} forward frames.")
        
        if tracked_count > 0:
            self.slider_var.set(track_idx)
            self.on_slider_change(track_idx)

    def handle_save_shortcut(self, event=None):
        if self.is_extracted:
            self.save_extracted_json()
        else:
            self.extract_frames()

    def extract_frames(self):
        if self.is_extracted or not self.cap: return
        self.sync_current_annotations()
        
        target_fps = simpledialog.askfloat("Set FPS", "Frames per second to extract:", initialvalue=5.0, minvalue=0.1, maxvalue=120.0)
        if not target_fps: return

        os.makedirs(self.extracted_folder, exist_ok=True)
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0: video_fps = 30.0
        frame_interval = max(1, int(round(video_fps / target_fps)))
        
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Extracting Frames...")
        tk.Label(progress_win, text="Extracting and saving images + labels...").pack(pady=10)
        progress_bar = ttk.Progressbar(progress_win, maximum=self.total_frames, length=300)
        progress_bar.pack(padx=20, pady=10)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        extracted_count = 0
        
        for i in range(self.total_frames):
            ret, frame = self.cap.read()
            if not ret: break
            
            if i % frame_interval == 0:
                base_name = os.path.splitext(os.path.basename(self.video_list[self.current_video_idx]))[0]
                out_name = f"{base_name}_{i:06d}"
                img_path = os.path.join(self.extracted_folder, out_name + ".jpg")
                
                cv2.imwrite(img_path, frame)
                extracted_count += 1
                
                if i in self.video_annotations and self.video_annotations[i]:
                    out_data = {
                        "type": "video_frame_annotations",
                        "date": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "image": {"file_name": out_name + ".jpg", "width": self.img_width, "height": self.img_height},
                        "annotations": self.video_annotations[i]
                    }
                    with open(os.path.join(self.extracted_folder, out_name + ".json"), 'w') as f:
                        json.dump(out_data, f, indent=4)
            
            if i % 10 == 0:
                progress_bar['value'] = i
                progress_win.update()
                
        progress_win.destroy()
        messagebox.showinfo("Extraction Complete", f"Extracted {extracted_count} frames.\nSwitching to Extracted Mode.")
        self.load_current_video() 

    def remove_extraction(self):
        if not self.is_extracted or not self.extracted_folder: return
        confirm = messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the extracted folder and all its labels?\n\n{self.extracted_folder}")
        if confirm:
            try:
                shutil.rmtree(self.extracted_folder)
                messagebox.showinfo("Deleted", "Folder removed successfully.")
                self.load_current_video() 
            except Exception as e:
                messagebox.showerror("Error", f"Could not delete folder:\n{e}")

    def save_extracted_json(self):
        if not self.is_extracted or not self.image_list: return
        img_path = self.image_list[self.current_frame]
        out_data = {
            "type": "video_frame_annotations",
            "date": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "image": {"file_name": os.path.basename(img_path), "width": self.img_width, "height": self.img_height},
            "annotations": self.current_frame_annotations 
        }
        try:
            save_path = os.path.splitext(img_path)[0] + ".json"
            with open(save_path, 'w') as f:
                json.dump(out_data, f, indent=4)
            self.set_dirty(False) 
        except Exception as e:
            messagebox.showerror("Error", f"Save Error:\n{e}")

    def zoom_in(self, event=None):
        self.scale *= 1.1
        self.redraw_canvas()

    def zoom_out(self, event=None):
        self.scale /= 1.1
        self.redraw_canvas()

    def zoom_mousewheel(self, event):
        if event.delta > 0: self.zoom_in()
        else: self.zoom_out()

    def get_raw_coords(self, event):
        x = self.canvas.canvasx(event.x) / self.scale
        y = self.canvas.canvasy(event.y) / self.scale
        return x, y

if __name__ == "__main__":
    root = tk.Tk()
    try: root.state('zoomed') 
    except tk.TclError:
        root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")
    app = VideoDetSegLabeler(root)
    root.mainloop()