from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO
import mrcfile
from scipy.stats import norm
import time
import asyncio
# ---------- Documentation ----------
"""Microscope Calibration Tool

This tool helps calibrate electron microscopes by analyzing test specimen images.
It calculates the pixel size (Angstroms/pixel) by measuring diffraction patterns
from known specimens like graphene, gold, or ice.

Key Features:
- Supports common image formats (.png, .tif) and MRC files
- Interactive FFT analysis with resolution circles
- Automatic pixel size detection
- Radial averaging for enhanced signal detection
- Customizable resolution measurements

Usage:
1. Upload a test specimen image
2. Select the expected diffraction pattern (graphene/gold/ice)
3. Adjust the region size to analyze
4. Click points in the FFT to measure distances
5. Use auto-search to find the best pixel size match

The tool will display:
- Original image with selected region
- FFT with resolution circles
- 1D radial average plot
- Calculated pixel size (Angstroms/pixel)
"""
import argparse

def print_help():
    """Print usage instructions and help information."""
    help_text = """
Microscope Calibration Tool
--------------------------

Usage:
    Run the Shiny app and follow the web interface.
    
Input Files:
    - Image formats: PNG, TIFF
    - MRC files from microscopes
    
Key Parameters:
    Apix: Pixel size in Angstroms/pixel (0.01-6.0)
    Region: Size of FFT analysis region (1-100%)
    Resolution circles:
        - Graphene: 2.13 Å
        - Gold: 2.355 Å 
        - Ice: 3.661 Å
        - Custom: User-defined resolution
        
Analysis Features:
    - Interactive FFT region selection
    - Resolution circle overlay
    - Automatic pixel size detection
    - Radial averaging
    - Click-to-measure distances
    
Output:
    - Processed FFT image
    - Radial intensity profile
    - Calculated pixel size
    """
    print(help_text)
    
app_ui = ui.page_fillable(
    #ui.h2("Microscope Calibration"),
    #ui.p("Upload an image of test specimens", style="font-size: 1.5em;"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file("upload", "Upload an image of test specimens (e.g., graphene)(.mrc,.tif,.png)", accept=["image/*", ".mrc", ".tif", ".png"]),
            ui.input_checkbox("circle_213", "Graphene", value=True),
            ui.input_checkbox("circle_235", "Gold", value=False),
            ui.input_checkbox("circle_366", "Ice", value=False),
            ui.div(
                {"style": "display: flex; align-items: center;"},
                ui.input_checkbox("circle_custom", "Res (Å):", value=False),
                ui.input_numeric("custom_resolution", None, value=3.0, min=0.1, max=10.0, step=0.01, width="80px"),
            ),
            ui.div(
                {"style": "padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 10px; display: flex; flex-direction: column; gap: 5px;"},
                ui.div(
                    {"style": "flex: 1;"},
                    ui.input_slider("apix_slider", "Apix (Å/px)", min=0.001, max=6.0, value=1.0, step=0.001),
                ),
                ui.div(
                    {"style": "display: flex; justify-content: flex-start; align-items: bottom; gap: 5px; margin-top: 5px; width: 100%;"},
                    ui.input_text("apix_exact_str", None, value="1.0", width="70px"),
                    ui.input_action_button("apix_set_btn", ui.tags.span("Set", style="display: flex; align-items: center; justify-content: center; width: 100%; height: 100%;"), class_="btn-primary", style="height: 38px; display: flex; align-items: center;", width="50px"),
                ),
            ),
            ui.div(
                {"style": "display: flex; align-items: center; gap: 10px; margin-top: 10px; margin-bottom: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;"},
                ui.div(
                    {"style": "flex: 1;"},
                    ui.input_numeric("apix_min", "Search Min", value=0.8, min=0.01, max=6.0, step=0.1),
                ),
                ui.div(
                    {"style": "flex: 1;"},
                    ui.input_numeric("apix_max", "Search Max", value=1.2, min=0.01, max=6.0, step=0.1),
                ),
                ui.div(
                    ui.input_action_button("search_apix", "Search", class_="btn-primary"),
                ),
            ),
            ui.div(
                {"style": "margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;"},
                ui.output_text("matched_apix", inline=True),
            ),
            title=ui.h2("Magnification Calibration", style="font-size: 36px; font-weight: bold; padding: 15px;"),
            open="open",
            width="400px",
            min_width="250px",
            max_width="500px",
            resize=True,
        ),
        ui.div(
            {"class": "resizable-container"},
            ui.div(
                {"class": "top-section"},
                ui.div(
                    ui.div(
                        {"class": "card-wrapper"},
                        ui.card(
                            ui.div(
                                {"class": "card-content"},
                                ui.div(
                                    {"class": "image-output"},
                                    ui.output_image("image_display", click=True, height="360px"),
                                ),
                                ui.div(
                                    {"class": "control-panel"},
                                    ui.div(
                                        {"class": "slider-container"},
                                        ui.input_slider("zoom1", "Zoom (%)", min=50, max=300, value=100),
                                    ),
                                    ui.div(
                                        {"class": "slider-container"},
                                        ui.input_slider("rg_size", "Region size (%)", min=1, max=100, value=30),
                                    ),
                                    ui.div(
                                        {"class": "slider-container"},
                                        ui.input_slider("contrast1", "Range (±σ)", min=0.1, max=5.0, value=2.0, step=0.1),
                                    ),
                                ),
                            ),
                            ui.div(
                                {"class": "card-footer"},
                                "Green denotes the region selected for FFT. Click to select region.",
                            ),
                            class_="fill-card"
                        ),
                    ),
                    ui.div({"class": "vertical-resizer", "id": "vertical-resizer"}),
                    ui.div(
                        {"class": "card-wrapper"},
                        ui.card(
                            ui.div(
                                {"class": "card-content"},
                                ui.div(
                                    {"class": "image-output"},
                                    ui.output_image("fft_with_circle", click=True),
                                ),
                                ui.div(
                                    {"class": "control-panel"},
                                    ui.div(
                                        {"class": "slider-container"},
                                        ui.input_slider("zoom2", "Zoom (%)", min=50, max=300, value=100),
                                    ),
                                    ui.div(
                                        {"class": "slider-container"},
                                        ui.input_slider("contrast", "Range (±σ)", min=0.1, max=5.0, value=2.0, step=0.1),
                                    ),
                                ),
                            ),
                            ui.div(
                                {"class": "card-footer"},
                                "Click on a resolution ring to correspond to selected feature to calibrate Apix",
                            ),
                            class_="fill-card"
                        ),
                    ),
                ),
            ),
            ui.div({"class": "resizer", "id": "resizer"}),
            ui.div(
                {"class": "bottom-section"},
            ui.card(
                    ui.div(
                        {"style": "display: flex;"},
                        ui.output_plot("fft_1d_plot", click=True, dblclick=True, brush=True, width="600px", height="400px"),
                        ui.div(
                            {"style": "display: flex; flex-direction: column; justify-content: flex-start; margin-left: 10px; width: 200px;"},
                            ui.input_checkbox("log_y", "Log Scale", value=False),
                            ui.div(
                                {"style": "margin-bottom: 10px;"},
                                ui.input_checkbox("gaussian_filter", "Apply Gaussian Filter", value=False),
                                ui.panel_conditional(
                                    "input.gaussian_filter",
                                    ui.input_slider("gaussian_sigma", "σ", min=0.001, max=0.1, value=0.02, step=0.001),
                                ),
                            ),
                            ui.input_action_button("reset_zoom", "Reset Zoom"),
                            ui.p("Drag to zoom, double-click to reset"),
                        ),
                    ),
                    ui.div(
                        {"class": "card-footer", "style": "justify-content: flex-start;"},
                        "Rotational average of the FFT across resolutions"
                    ),
                    class_="fill-card"
                ),
            ),
        ),
        fillable=True,
    ),
    fillable=True,
    padding=0,
    title="Microscope Calibration",
)
# Add custom CSS for resizable layout
app_ui = ui.tags.div(
    ui.tags.style("""
        .fill-card {
            height: 100%;
            min-height: 0;
            display: flex;
            flex-direction: column;
            width: 100%;
        }
        .fill-card > div {
            flex: 1;
            min-height: 0;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .resizable-container {
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow: hidden;
            min-height: 0;
        }
        .top-section {
            flex: 1;
            min-height: 200px;
            overflow: hidden;
        }
        .top-section > div {
            height: 100%;
            display: flex;
            padding: 1rem;
            gap: 0;
        }
        .top-section .card-wrapper {
            flex: 1;
            min-width: 200px;
            display: flex;
            width: 100%;
        }
        .vertical-resizer {
            width: 8px;
            background: #f0f0f0;
            cursor: col-resize;
            border-left: 1px solid #ccc;
            border-right: 1px solid #ccc;
            margin: 0 4px;
            flex-shrink: 0;
        }
        .vertical-resizer:hover {
            background: #e0e0e0;
        }
        .bottom-section {
            flex: 1;
            min-height: 200px;
            overflow: auto;
        }
        .resizer {
            height: 8px;
            background: #f0f0f0;
            cursor: row-resize;
            border-top: 1px solid #ccc;
            border-bottom: 1px solid #ccc;
        }
        .resizer:hover {
            background: #e0e0e0;
        }
        /* Image container styles */
        .fill-card .image-output {
            flex: 1;
            min-height: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: auto;
            padding: 10px;
            margin-bottom: 10px;  /* Add margin to separate from control panel */
            width: 100%;
            /* Ensure scrollbars are always visible and not hidden */
            scrollbar-width: auto;
            scrollbar-color: rgba(0, 0, 0, 0.3) transparent;
        }
        .fill-card .image-output::-webkit-scrollbar {
            width: 12px;
            height: 12px;
        }
        .fill-card .image-output::-webkit-scrollbar-track {
            background: transparent;
        }
        .fill-card .image-output::-webkit-scrollbar-thumb {
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 6px;
            border: 2px solid transparent;
        }
        .fill-card .image-output img {
            height: auto;
            width: auto;
            max-width: none;
            max-height: none;
            /* Add margin to ensure scrollbar is not covered */
            margin-bottom: 12px;
        }
        /* Control panel styles */
        .fill-card .control-panel {
            min-height: 80px;  /* Increase minimum height */
            padding: 10px 15px;  /* Increase padding */
            border-top: 1px solid #eee;
            flex-shrink: 0;
            display: flex;
            gap: 20px;
            align-items: flex-start;  /* Align items to top */
            margin-bottom: 0;
            width: 100%;
            background-color: #f8f9fa;  /* Add light background */
            border-radius: 0 0 4px 4px;  /* Round bottom corners */
            z-index: 10;  /* Ensure controls stay above scrollbar */
        }
        .fill-card .control-panel .slider-container {
            flex: 1;
            min-width: 0;
            padding: 5px 0;  /* Add vertical padding */
        }
        .fill-card .control-panel .slider-container .form-group {
            margin-bottom: 0;
        }
        .fill-card .control-panel .slider-container label {
            margin-bottom: 8px;  /* Add space below labels */
            font-weight: 500;  /* Make labels slightly bolder */
        }
        /* Card content layout */
        .card-content {
            display: flex;
            flex-direction: column;
            height: 100%;
            min-height: 0;
            width: 100%;
            position: relative;  /* For proper z-index stacking */
        }
        /* Footer styles */
        .card-footer {
            height: 40px;
            padding: 8px;
            background-color: rgba(0, 0, 0, 0.03);
            border-top: 1px solid rgba(0, 0, 0, 0.125);
            display: flex;
            align-items: center;
            flex-shrink: 0;
            margin-top: 0;
            width: 100%;
        }
        .sidebar > h2, 
        .sidebar-title,
        .shiny-sidebar-title {
            font-size: 36px !important;
            font-weight: bold !important;
            padding: 15px !important;
        }
    """),
    ui.tags.script("""
        document.addEventListener('DOMContentLoaded', function() {
            // Horizontal resizer
            const resizer = document.getElementById('resizer');
            const topSection = resizer.previousElementSibling;
            const bottomSection = resizer.nextElementSibling;
            let y = 0;
            let topHeight = 0;

            function onMouseDown(e) {
                y = e.clientY;
                topHeight = topSection.getBoundingClientRect().height;
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            }

            function onMouseMove(e) {
                const delta = e.clientY - y;
                const newTopHeight = Math.max(200, Math.min(topHeight + delta, 
                    resizer.parentNode.getBoundingClientRect().height - 200));
                topSection.style.height = `${newTopHeight}px`;
                topSection.style.flex = 'none';
                bottomSection.style.flex = '1';
            }

            function onMouseUp() {
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
            }

            resizer.addEventListener('mousedown', onMouseDown);

            // Vertical resizer
            const verticalResizer = document.getElementById('vertical-resizer');
            const leftCard = verticalResizer.previousElementSibling;
            const rightCard = verticalResizer.nextElementSibling;
            let x = 0;
            let leftWidth = 0;

            function onVerticalMouseDown(e) {
                x = e.clientX;
                leftWidth = leftCard.getBoundingClientRect().width;
                document.addEventListener('mousemove', onVerticalMouseMove);
                document.addEventListener('mouseup', onVerticalMouseUp);
            }

            function onVerticalMouseMove(e) {
                const delta = e.clientX - x;
                const containerWidth = verticalResizer.parentNode.getBoundingClientRect().width;
                const minWidth = 200;
                const maxWidth = containerWidth - minWidth - verticalResizer.offsetWidth;
                const newLeftWidth = Math.max(minWidth, Math.min(leftWidth + delta, maxWidth));
                
                leftCard.style.width = `${newLeftWidth}px`;
                leftCard.style.flex = 'none';
                rightCard.style.flex = '1';
            }

            function onVerticalMouseUp() {
                document.removeEventListener('mousemove', onVerticalMouseMove);
                document.removeEventListener('mouseup', onVerticalMouseUp);
            }

            verticalResizer.addEventListener('mousedown', onVerticalMouseDown);

            // Handle window resize
            function handleResize() {
                const container = verticalResizer.parentNode;
                const containerWidth = container.getBoundingClientRect().width;
                
                // Reset flex properties if window is resized
                if (!leftCard.style.width) {
                    leftCard.style.flex = '1';
                    rightCard.style.flex = '1';
                } else {
                    // Ensure left card width doesn't exceed bounds after resize
                    const currentWidth = parseInt(leftCard.style.width);
                    const minWidth = 200;
                    const maxWidth = containerWidth - minWidth - verticalResizer.offsetWidth;
                    if (currentWidth > maxWidth) {
                        leftCard.style.width = `${maxWidth}px`;
                    }
                }
            }

            window.addEventListener('resize', handleResize);
        });
    """),
    app_ui
)
size = 360

# ---------- Server ----------
def server(input: Inputs, output: Outputs, session: Session):
    # Add reactive values for region center and FFT click position
    region_center = reactive.Value({
        'x': None,
        'y': None
    })
    
    fft_click_pos = reactive.Value({
        'x': None,
        'y': None
    })

    # Add reactive value for apix
    current_apix = reactive.Value(1.0)

    # Add reactive value for apix search results
    search_results = reactive.Value({
        'apix': None,
        'score': None
    })

    # Add plot zoom state
    plot_zoom = reactive.Value({
        'x_range': None,
        'y_range': None
    })

    # Add reactive values for raw data and region
    raw_image_data = reactive.Value({
        'img': None,
        'data': None
    })

    @reactive.Effect
    @reactive.event(input.apix_slider)
    def _():
        """Update current_apix and text input when slider changes."""
        slider_value = input.apix_slider()
        current_apix.set(slider_value)
        # Update text input to match slider
        ui.update_text("apix_exact_str", value=str(round(slider_value, 3)), session=session)

    @reactive.Effect
    @reactive.event(input.apix_set_btn)
    def _():
        """Update slider and apix only when Set button is clicked and value is valid. Also update apix_min and apix_max to ±1%."""
        try:
            val = float(input.apix_exact_str())
            if 0.001 <= val <= 6.0:
                ui.update_slider("apix_slider", value=val, session=session)
                current_apix.set(val)
                # Update apix_min and apix_max to ±1% of set value
                min_apix = max(0.01, round(val * 0.99, 3))
                max_apix = min(6.0, round(val * 1.01, 3))
                ui.update_numeric("apix_min", value=min_apix, session=session)
                ui.update_numeric("apix_max", value=max_apix, session=session)
            else:
                # Optionally, show an error or reset
                pass
        except Exception:
            # Optionally, show an error or reset
            pass

    @reactive.Effect
    @reactive.event(input.image_display_click)
    def _():
        click_data = input.image_display_click()
        if click_data is not None:
            # Scale the coordinates based on the zoom level
            zoom_factor = input.zoom1() / 100
            region_center.set({
                'x': int(click_data['x'] / zoom_factor),
                'y': int(click_data['y'] / zoom_factor)
            })

    @reactive.Effect
    @reactive.event(input.search_apix)
    async def _():
        """Handle apix search and update controls."""
        # Get the current FFT data
        from shiny import req
        path = image_path()
        if not path or not path.exists():
            return
        req(raw_image_data.get()['data'] is not None)

        region = get_current_region()
        if region is None:
            return

        # Compute FFT
        arr = np.array(region.convert("L")).astype(np.float32)
        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        # Compute radial profile
        cy, cx = np.array(magnitude.shape) // 2
        y, x = np.indices(magnitude.shape)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r = r.astype(np.int32)
        radial_sum = np.bincount(r.ravel(), magnitude.ravel())
        radial_count = np.bincount(r.ravel())
        radial_profile = radial_sum / (radial_count + 1e-8)

        # Get the first checked resolution
        resolution, _ = get_first_checked_resolution()
        if resolution is None:
            return

        # Search through apix range
        min_apix = input.apix_min()
        max_apix = input.apix_max()
        if min_apix >= max_apix:
            return

        def find_local_peaks(profile, min_distance=5):
            """Find indices of local peaks in the profile."""
            peaks = []
            for i in range(min_distance, len(profile) - min_distance):
                window = profile[i-min_distance:i+min_distance+1]
                if profile[i] == max(window):
                    peaks.append(i)
            return peaks

        def score_peak_match(peak_freq, target_freq):
            """Score how well a peak frequency matches the target frequency."""
            return -abs(peak_freq - target_freq)

        apix_values = np.linspace(min_apix, max_apix, 100)
        best_score = -np.inf
        best_apix = None
        target_freq = 1 / resolution

        # Smooth the radial profile
        window_size = 5
        smoothed_profile = np.convolve(radial_profile, 
                                     np.ones(window_size)/window_size, 
                                     mode='valid')

        # Find peaks in the smoothed profile
        peak_indices = find_local_peaks(smoothed_profile)

        for apix in apix_values:
            freqs = np.arange(len(smoothed_profile)) / (arr.shape[0] * apix)
            for peak_idx in peak_indices:
                if peak_idx < len(freqs):
                    peak_freq = freqs[peak_idx]
                    peak_height = smoothed_profile[peak_idx]
                    score = score_peak_match(peak_freq, target_freq) * peak_height
                    
                    if score > best_score:
                        best_score = score
                        best_apix = apix

        if best_apix is not None:
            search_results.set({
                'apix': best_apix,
                'score': best_score
            })
            # Update the controls with the new value
            new_apix = round(best_apix, 3)
            current_apix.set(new_apix)
            ui.update_slider("apix_slider", value=new_apix, session=session)
            ui.update_text("apix_exact_str", value=str(new_apix), session=session)
            
            # Force update of plots
            await session.send_custom_message("shiny:forceUpdate", None)

    def get_first_checked_resolution():
        """Return the first checked resolution value or None if none are checked."""
        if input.circle_213():
            return 2.13, "red"
        elif input.circle_235():
            return 2.355, "orange"
        elif input.circle_366():
            return 3.661, "blue"
        elif input.circle_custom():
            return input.custom_resolution(), "green"
        return None, None

    @reactive.Effect
    @reactive.event(input.fft_with_circle_click)
    def _():
        click_data = input.fft_with_circle_click()
        if click_data is not None:
            # Store click position for visualization
            zoom_factor = input.zoom2() / 100
            x = click_data['x'] / zoom_factor
            y = click_data['y'] / zoom_factor
            
            # Get the first checked resolution and its color
            resolution, color = get_first_checked_resolution()
            if resolution is None:
                return
                
            fft_click_pos.set({
                'x': x,
                'y': y,
                'color': color
            })

            # Calculate distance from center in pixels
            center_x = size / 2
            center_y = size / 2
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            if distance > 0:
                # Calculate new apix value that would make the checked resolution circle
                # appear at the clicked distance
                new_apix = (distance * resolution) / size
                
                # Update apix value if within bounds
                if 0.01 <= new_apix <= 6.0:
                    new_value = round(new_apix, 3)
                    # Update the slider value directly without triggering current_apix
                    ui.update_slider("apix_slider", value=new_value, session=session)

    @reactive.Calc
    def image_path():
        file = input.upload()
        if not file:
            return None
        return Path(file[0]["datapath"])

    def save_temp_image(img: Image.Image) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        tmp.close()
        return tmp.name

    def normalize(magnitude, contrast=2.0):
        mean = np.mean(magnitude)
        std = np.std(magnitude)
        m1 = np.max(magnitude)
        # Adjust clip max based on contrast value
        clip_max = min(m1, mean + contrast * std)
        clip_min = 0
        magnitude_clipped = np.clip(magnitude, clip_min, clip_max)
        normalized = 255 * (magnitude_clipped - clip_min) / (clip_max - clip_min + 1e-8)
        return normalized

    def normalize_image(img: np.ndarray, contrast=2.0) -> np.ndarray:
        """Normalize image data using mean and standard deviation.
        
        Args:
            img: Input image array
            contrast: Number of standard deviations to include in range
            
        Returns:
            Normalized image array (0-255 uint8)
        """
        # Convert to float32 for calculations
        img_float = img.astype(np.float32)
        mean = np.mean(img_float)
        std = np.std(img_float)
        
        # Calculate clip range based on mean ± contrast * std
        clip_min = max(0, mean - contrast * std)
        clip_max = min(img_float.max(), mean + contrast * std)
        
        # Clip and normalize to 0-255 range
        img_clipped = np.clip(img_float, clip_min, clip_max)
        img_normalized = 255 * (img_clipped - clip_min) / (clip_max - clip_min + 1e-8)
        
        return img_normalized.astype(np.uint8)

    def read_mrc_as_image(mrc_path: str) -> Image.Image:
        """Read an MRC file and convert it to a PIL Image.
        
        Args:
            mrc_path: Path to the MRC file
            
        Returns:
            PIL Image object
        """
        with mrcfile.open(mrc_path) as mrc:
            # Get the data and convert to float32
            data = mrc.data.astype(np.float32)
            
            # Create PIL Image (normalization will be done later)
            return Image.fromarray(data.astype(np.uint8))

    def load_image(path: Path) -> tuple[Image.Image, np.ndarray]:
        """Load an image file or MRC file and return as PIL Image and raw data.
        
        Args:
            path: Path to the image or MRC file
            
        Returns:
            Tuple of (PIL Image object, raw numpy array)
        """
        if path.suffix.lower() == '.mrc':
            with mrcfile.open(str(path)) as mrc:
                data = mrc.data.astype(np.float32)
                return Image.fromarray(data.astype(np.uint8)), data
        else:
            img = Image.open(path)
            return img, np.array(img.convert("L")).astype(np.float32)

    def fft_image_with_matplotlib(region: np.ndarray, contrast=2.0, return_array=False):
        f = np.fft.fft2(region)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        normalized = normalize(magnitude, contrast)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(normalized, cmap='gray')
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    def compute_fft_image_region(cropped: Image.Image, contrast=2.0) -> Image.Image:
        arr = np.array(cropped.convert("L")).astype(np.float32)
        return fft_image_with_matplotlib(arr, contrast)

    def compute_average_fft(cropped: Image.Image, apix: float = 1.0) -> Image.Image:
        """
        Compute the 1D rotational average of the 2D FFT from a cropped image.

        Args:
            cropped: A PIL.Image object (grayscale or RGB).
            apix: Pixel size in Ångstrom per pixel.

        Returns:
            A PIL.Image containing the 1D plot of average FFT intensity vs. 1/resolution.
        """
        arr = np.array(cropped.convert("L")).astype(np.float32)
        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        # Compute radial coordinates
        cy, cx = np.array(magnitude.shape) // 2
        y, x = np.indices(magnitude.shape)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r = r.astype(np.int32)
        # Compute radial average
        radial_sum = np.bincount(r.ravel(), magnitude.ravel())
        radial_count = np.bincount(r.ravel())
        radial_profile = radial_sum / (radial_count + 1e-8)

        # Convert to spatial frequency
        freqs = np.arange(len(radial_profile)) / (arr.shape[0] * apix)
        inverse_resolution = freqs  # in 1/Å

        # Determine index range for 1/3.7 to 1/2
        x_min, x_max = 1 / 3.7, 1 / 2.0
        mask = (inverse_resolution >= x_min) & (inverse_resolution <= x_max)

        # Plot
        fig, ax = plt.subplots(dpi=100)
        ax.plot(inverse_resolution[mask], np.log1p(radial_profile[mask]))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(radial_profile[mask].min(), radial_profile[mask].max())
        ax.set_xlabel("1 / Resolution (1/Å)")
        ax.set_ylabel("Log(Average FFT intensity)")
        ax.set_title("1D FFT Radial Profile")
        ax.grid(True)

        # Save to PIL.Image
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    @reactive.Calc
    def get_apix():
        return current_apix.get()

    def get_processed_image_for_display():
        """Get the contrast-adjusted image from raw data for display only."""
        if raw_image_data.get()['data'] is None:
            return None
            
        # Apply contrast normalization to raw data
        normalized_data = normalize_image(raw_image_data.get()['data'], input.contrast1())
        img = Image.fromarray(normalized_data)
        return img.convert("RGB")

    def get_base_image():
        """Get the base image without contrast adjustment for FFT calculations."""
        if raw_image_data.get()['data'] is None:
            return None
            
        return raw_image_data.get()['img'].convert("RGB")

    @reactive.Effect
    @reactive.event(input.upload)
    def _():
        """Update raw image data when a new file is uploaded."""
        path = image_path()
        if not path or not path.exists():
            raw_image_data.set({'img': None, 'data': None})
            return
            
        img, raw_data = load_image(path)
        raw_image_data.set({
            'img': img,
            'data': raw_data
        })

    @output
    @render.image
    def image_display():
        from shiny import req
        path = image_path()
        req(path and path.exists())
        req(raw_image_data.get()['data'] is not None)
        
        # Get contrast-adjusted image
        img = get_processed_image_for_display()
        
        # Apply zoom
        base_size = size
        zoom_factor = input.zoom1() / 100
        new_size = int(base_size * zoom_factor)
        img = img.resize((new_size, new_size))
        
        # Calculate region size
        rg_sz = (input.rg_size()*img.size[0]/100, input.rg_size()*img.size[1]/100)
        draw = ImageDraw.Draw(img)

        # Use clicked position if available, otherwise use center
        if region_center.get()['x'] is not None:
            # Scale the coordinates based on the zoom level
            center = (
                int(region_center.get()['x'] * zoom_factor),
                int(region_center.get()['y'] * zoom_factor)
            )
        else:
            center = (img.size[0] // 2, img.size[1] // 2)

        x1, y1 = center[0] - rg_sz[0] // 2, center[1] - rg_sz[1] // 2
        x2, y2 = center[0] + rg_sz[0] // 2, center[1] + rg_sz[1] // 2
        draw.rectangle([(x1, y1), (x2, y2)], outline='green', width=2)
        
        return {"src": save_temp_image(img)}

    def get_current_region():
        """Get the current region for FFT calculation."""
        if raw_image_data.get()['data'] is None:
            return None
            
        # Use base image without contrast adjustment for FFT
        img = get_base_image()
        if img is None:
            return None
            
        rg_sz = input.rg_size()*img.size[0]/100

        # Use clicked position if available, otherwise use center
        if region_center.get()['x'] is not None:
            center = (region_center.get()['x'], region_center.get()['y'])
        else:
            center = (img.size[0] // 2, img.size[0] // 2)

        x1, y1 = center[0] - rg_sz // 2, center[1] - rg_sz // 2
        x2, y2 = center[0] + rg_sz // 2, center[1] + rg_sz // 2
        region = img.crop((x1, y1, x2, y2))
        return region

    @output
    @render.image
    def fft_with_circle():
        from shiny import req
        path = image_path()
        req(path and path.exists())
        req(raw_image_data.get()['data'] is not None)
        
        region = get_current_region()
        if region is None:
            return None
            
        fft_img = compute_fft_image_region(region, input.contrast())
        base_size = size
        zoom_factor = input.zoom2() / 100
        new_size = int(base_size * zoom_factor)
        fft_img = fft_img.resize((new_size, new_size))
        draw = ImageDraw.Draw(fft_img)
        center = (new_size // 2, new_size // 2)

        def resolution_to_radius(res_angstrom):
            return (fft_img.size[0] * get_apix()) / res_angstrom

        # Draw resolution circles
        if input.circle_213():
            r = resolution_to_radius(2.13)
            draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), outline="red", width=2)
        if input.circle_235():
            r = resolution_to_radius(2.355)
            draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), outline="orange", width=2)
        if input.circle_366():
            r = resolution_to_radius(3.661)
            draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), outline="blue", width=2)
        if input.circle_custom():
            r = resolution_to_radius(input.custom_resolution())
            draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), outline="green", width=2)

        # Add current apix label
        current_apix_str = f"Apix: {get_apix():.3f} Å/px"
        # Determine color based on selected resolution
        if input.circle_213():
            apix_color = "red"
        elif input.circle_235():
            apix_color = "orange"
        elif input.circle_366():
            apix_color = "blue"
        elif input.circle_custom():
            apix_color = "green"
        else:
            apix_color = "black"
        try:
            font = ImageFont.truetype("Arial", 16)
        except OSError:
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), current_apix_str, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = 10
        draw.rectangle((padding, padding, padding + text_width + 10, padding + text_height + 10), 
                      fill=(255, 255, 255, 180))
        draw.text((padding + 5, padding + 5), current_apix_str, fill=apix_color, font=font)

        # Draw click marker if exists
        click_pos = fft_click_pos.get()
        if click_pos['x'] is not None and click_pos['y'] is not None:
            marker_size = 10
            x, y = click_pos['x'], click_pos['y']
            color = click_pos.get('color', 'yellow')  # Default to yellow if no color specified
            draw.line([(x - marker_size, y), (x + marker_size, y)], fill=color, width=2)
            draw.line([(x, y - marker_size), (x, y + marker_size)], fill=color, width=2)

        return {"src": save_temp_image(fft_img), "click": True}

    @output
    @render.plot
    def fft_1d_plot():
        from shiny import req
        path = image_path()
        req(path and path.exists())
        req(raw_image_data.get()['data'] is not None)

        region = get_current_region()
        if region is None:
            return None

        # Compute FFT and radial profile
        arr = np.array(region.convert("L")).astype(np.float32)
        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        # Compute radial coordinates
        cy, cx = np.array(magnitude.shape) // 2
        y, x = np.indices(magnitude.shape)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r = r.astype(np.int32)

        # Compute radial average
        radial_sum = np.bincount(r.ravel(), magnitude.ravel())
        radial_count = np.bincount(r.ravel())
        radial_profile = radial_sum / (radial_count + 1e-8)

        # Convert to spatial frequency
        freqs = np.arange(len(radial_profile)) / (arr.shape[0] * get_apix())
        inverse_resolution = freqs  # in 1/Å

        # Determine index range for 1/3.7 to 1/2
        x_min, x_max = 1 / 3.7, 1 / 2.0
        mask = (inverse_resolution >= x_min) & (inverse_resolution <= x_max)

        # Create figure with adjusted margins
        fig = plt.figure(figsize=(10, 6))
        # Adjust subplot parameters to give more space at the bottom
        plt.subplots_adjust(bottom=0.15, left=0.12, right=0.95, top=0.95)
        ax = fig.add_subplot(111)
        
        # Plot data
        y_data = radial_profile[mask]
        if input.log_y():
            y_data = np.log1p(y_data)
            ax.set_ylabel("Log(Average FFT intensity)")
        else:
            ax.set_ylabel("Average FFT intensity")

        # Apply Gaussian filter if enabled
        if input.gaussian_filter():
            # Initialize combined Gaussian filter
            x = inverse_resolution[mask]
            combined_gaussian = np.zeros_like(x)
            
            # Check each resolution and add its Gaussian
            if input.circle_213():
                gaussian = norm.pdf(x, loc=1/2.13, scale=input.gaussian_sigma())
                combined_gaussian += gaussian
            if input.circle_235():
                gaussian = norm.pdf(x, loc=1/2.355, scale=input.gaussian_sigma())
                combined_gaussian += gaussian
            if input.circle_366():
                gaussian = norm.pdf(x, loc=1/3.661, scale=input.gaussian_sigma())
                combined_gaussian += gaussian
            if input.circle_custom():
                gaussian = norm.pdf(x, loc=1/input.custom_resolution(), scale=input.gaussian_sigma())
                combined_gaussian += gaussian
            
            # Only apply if at least one resolution was selected
            if np.any(combined_gaussian > 0):
                # Normalize combined Gaussian to have peak of 1
                combined_gaussian = combined_gaussian / combined_gaussian.max()
                # Apply filter
                y_data = y_data * combined_gaussian

        ax.plot(inverse_resolution[mask], y_data, label="FFT average")
        
        # Set axis limits based on zoom state or defaults
        zoom = plot_zoom.get()
        if zoom['x_range'] is not None and zoom['y_range'] is not None:
            xlim = zoom['x_range']
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(x_min, x_max)
            xlim = (x_min, x_max)
            
        if zoom['y_range'] is not None:
            ax.set_ylim(zoom['y_range'])
        else:
            ax.set_ylim(y_data.min(), y_data.max())

        # Convert x-axis ticks to resolution values
        xticks = ax.get_xticks()
        xticks = xticks[(xticks > 0) & (xticks < max(inverse_resolution[mask]))]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{1/x:.2f}' for x in xticks])
        ax.set_xlabel("Resolution (Å)")
        ax.set_title("1D FFT Radial Profile")
        ax.grid(True)

        # Draw vertical lines for selected circles
        if input.circle_213():
            ax.axvline(1/2.13, color="red", linestyle="--", label="Graphene at 2.13 Å")
        if input.circle_235():
            ax.axvline(1/2.355, color="orange", linestyle="--", label="Gold at 2.355 Å")
        if input.circle_366():
            ax.axvline(1/3.661, color="blue", linestyle="--", label="Ice at 3.661 Å")
        if input.circle_custom():
            ax.axvline(1/input.custom_resolution(), color="green", linestyle="--", 
                      label=f"Custom at {input.custom_resolution():.2f} Å")

        ax.legend(loc="upper right", fontsize="small")
        
        return fig

    @reactive.Effect
    @reactive.event(input.reset_zoom)
    async def _():
        # Clear the brush selection first by sending a custom message
        await session.send_custom_message("shiny:brushed", {"plot_id": "fft_1d_plot", "coords": None})
        # Then reset the zoom state
        plot_zoom.set({'x_range': None, 'y_range': None})

    @reactive.Effect
    @reactive.event(input.fft_1d_plot_brush)
    def _():
        brush_data = input.fft_1d_plot_brush()
        if brush_data is not None:
            plot_zoom.set({
                'x_range': (brush_data['xmin'], brush_data['xmax']),
                'y_range': (brush_data['ymin'], brush_data['ymax'])
            })

    @reactive.Effect
    @reactive.event(input.fft_1d_plot_dblclick)
    def _():
        if input.fft_1d_plot_dblclick() is not None:
            plot_zoom.set({'x_range': None, 'y_range': None})

    @output
    @render.text
    def matched_apix():
        result = search_results.get()
        if result['apix'] is not None:
            return f"Best Matched Apix: {result['apix']:.3f} Å/px"
        return "Best Matched Apix: -"

app = App(app_ui, server)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Microscope Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False  # We'll handle help manually to use our custom format
    )
    parser.add_argument('--help', '-h', action='store_true', 
                       help='Show detailed help message')

    args = parser.parse_args()
    
    if args.help:
        print_help()
    else:
        app.run()
