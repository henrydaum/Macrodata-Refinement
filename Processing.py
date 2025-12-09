import py5
import generativeart
import numpy as np
import datetime
import re

# --- Configuration ---
FOLDER = generativeart.get_current_day_folder()

# Dimensions
WIDTH = 2560
HEIGHT = 1600
RESOLUTION_MULTIPLIER = 2

# Grid Settings
BASE_RES = 75
ASPECT_RATIO = WIDTH / HEIGHT 
COLS = int(BASE_RES * ASPECT_RATIO) 
ROWS = BASE_RES

NUM_HIKERS = 4000      
STEPS_PER_HIKER = 700

THRESHOLD_COEF = 0.24  
LABEL_FONT_SIZE = 20

# Globals
points = []
density_grid = None
color_grid = None
intensity_grid = None # NEW: Stores key/mouse activity
max_avg_intensity = 0.7 # NEW: For normalizing line width

def settings():
    py5.size(200, 200) 
    py5.no_smooth()

def setup():
    global points, density_grid, color_grid, intensity_grid, max_avg_intensity
    
    # Hide window
    surface = py5.get_surface()
    try: surface.set_visible(False)
    except: pass
        
    print(f"Loading data from {FOLDER}...")
    points, _ = generativeart.parse_journal_for_art(FOLDER, n_dim=2)
    
    if not points:
        py5.exit_sketch()
        return

    # Initialize Grids
    density_grid = np.zeros((COLS, ROWS))
    color_grid = np.zeros((COLS, ROWS, 3)) 
    intensity_grid = np.zeros((COLS, ROWS)) # Init intensity grid

    # --- Build Terrain ---
    print(f"Building terrain...")
    
    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    for p in points:
        gx = int(remap(p['x'], min_x, max_x, 0, COLS-1))
        gy = int(remap(p['y'], min_y, max_y, 0, ROWS-1))
        
        # Pull intensity (default to 0 if missing)
        activity = p.get('intensity', 0)
        
        add_hill(gx, gy, 4, 1.0, (p['r'], p['g'], p['b']), activity)

    # --- Normalize Intensity ---
    # We calculate the "Average Intensity" per cell (Total Activity / Density)
    # This prevents the lines from just getting infinitely thick in dense areas.
    # We want "Work per Visit", not just "Total Work".
    print("Normalizing intensity map...")
    temp_avg_grid = np.divide(intensity_grid, density_grid, out=np.zeros_like(intensity_grid), where=density_grid!=0)
    max_avg_intensity = np.max(temp_avg_grid)
    if max_avg_intensity == 0: max_avg_intensity = 1.0 # Avoid div/0
    
    print(f"Max Local Intensity: {max_avg_intensity}")

    py5.no_loop()

def draw():
    RENDER_WIDTH = WIDTH * RESOLUTION_MULTIPLIER
    RENDER_HEIGHT = HEIGHT * RESOLUTION_MULTIPLIER

    print(f"Rendering {WIDTH}x{HEIGHT} wallpaper at {RENDER_WIDTH}x{RENDER_HEIGHT}...")
    
    pg = py5.create_graphics(RENDER_WIDTH, RENDER_HEIGHT)
    pg.begin_draw()
    pg.background(15) 
    
    draw_blueprint_grid(pg)

    pg.no_fill()
    pg.smooth(8)
    
    # --- 1. Draw Contours ---
    for i in range(NUM_HIKERS):
        if i % 500 == 0: print(f"Hikers: {i}/{NUM_HIKERS}")
            
        x = py5.random(pg.width)
        y = py5.random(pg.height)
        
        pg.begin_shape()
        for step in range(STEPS_PER_HIKER):
            c = int(remap(x, 0, pg.width, 0, COLS-1))
            r = int(remap(y, 0, pg.height, 0, ROWS-1))
            c_safe = max(0, min(c, COLS-1))
            r_safe = max(0, min(r, ROWS-1))
            
            d = density_grid[c_safe][r_safe]
            
            if d > 0.1:
                # Color Logic
                r_val = color_grid[c_safe][r_safe][0] / d
                g_val = color_grid[c_safe][r_safe][1] / d
                b_val = color_grid[c_safe][r_safe][2] / d
                
                # Blend with neighbors
                neighbor_colors = []
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    nx, ny = c_safe + dx, r_safe + dy
                    if 0 <= nx < COLS and 0 <= ny < ROWS and density_grid[nx][ny] > 0.1:
                        neighbor_r = color_grid[nx][ny][0] / density_grid[nx][ny]
                        neighbor_g = color_grid[nx][ny][1] / density_grid[nx][ny]
                        neighbor_b = color_grid[nx][ny][2] / density_grid[nx][ny]
                        neighbor_colors.append([neighbor_r, neighbor_g, neighbor_b])
                
                if neighbor_colors:
                    avg_neighbor = np.mean(neighbor_colors, axis=0)
                    r_val = r_val * 0.7 + avg_neighbor[0] * 0.3
                    g_val = g_val * 0.7 + avg_neighbor[1] * 0.3
                    b_val = b_val * 0.7 + avg_neighbor[2] * 0.3

                # Vibrance Boost
                r_val = pow(r_val / 255.0, 0.85) * 255
                g_val = pow(g_val / 255.0, 0.85) * 255
                b_val = pow(b_val / 255.0, 0.85) * 255
                
                alpha = 150 + py5.random(-20, 20) 
                pg.stroke(r_val, g_val, b_val, alpha)
                
                # === USAGE-BASED LINE WIDTH ===
                # Get total intensity at this spot
                total_int = intensity_grid[c_safe][r_safe]
                # Calculate Average (Intensity per 'Event')
                avg_int = total_int / d
                
                # Normalize (0.0 to 1.0) based on the day's max peak
                norm_int = avg_int / max_avg_intensity
                
                # Map to stroke weight (Min: 0.5px, Max: 6.0px)
                # We use a slight power curve (pow 0.5) to make lower intensities still visible
                base_weight = remap(pow(norm_int, 0.5), 0, 1, 0.5, 6.0)
                
                pg.stroke_weight(base_weight * RESOLUTION_MULTIPLIER)
                
            else:
                pg.stroke(40, 40, 40, 30)
                pg.stroke_weight(1.0 * RESOLUTION_MULTIPLIER)

            if step % 2 == 0:
                pg.vertex(x, y)
            
            c_left = max(0, c_safe - 1)
            c_right = min(COLS-1, c_safe + 1)
            r_up = max(0, r_safe - 1)
            r_down = min(ROWS-1, r_safe + 1)
            
            dx = density_grid[c_right][r_safe] - density_grid[c_left][r_safe]
            dy = density_grid[c_safe][r_down] - density_grid[c_safe][r_up]
            
            angle = np.arctan2(dy, dx) + np.pi/2
            
            x += np.cos(angle)
            y += np.sin(angle)

            if x < -10 or x > pg.width+10 or y < -10 or y > pg.height+10:
                break
            
        pg.end_shape()

    label_peaks(pg)
    apply_vignette(pg)
    apply_grain(pg, intensity=0.03)

    pg.end_draw()
    
    print("Downsampling to final resolution...")
    final_img = py5.create_graphics(WIDTH, HEIGHT)
    final_img.begin_draw()
    final_img.image(pg, 0, 0, WIDTH, HEIGHT)
    final_img.end_draw()
    
    now = datetime.datetime.now()
    date_time = now.strftime("%m_%d_%Y_%I_%M_%p")
    filename = f"Map_{date_time}.png"
    final_img.save(filename)
    print(f"Saved: {filename}")
    py5.exit_sketch()

# --- Helpers ---

def draw_blueprint_grid(pg):
    pg.stroke(255, 25)
    pg.stroke_weight(2)
    step = int(128 * (pg.width / WIDTH))
    for x in range(0, pg.width, step):
        for y in range(0, pg.height, step):
            size = 5 * (pg.width / WIDTH)
            pg.line(x - size, y, x + size, y)
            pg.line(x, y - size, x, y + size)

def label_peaks(pg):
    threshold = np.max(density_grid) * THRESHOLD_COEF
    font_size = LABEL_FONT_SIZE * (pg.width / WIDTH)
    f = py5.create_font("Consolas Bold", font_size) 
    pg.text_font(f)
    
    placed_labels = []
    
    candidates = []
    for x in range(2, COLS-2):
        for y in range(2, ROWS-2):
            val = density_grid[x][y]
            if val > threshold:
                is_peak = True
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if density_grid[x+dx][y+dy] > val:
                            is_peak = False
                if is_peak:
                    candidates.append((x, y, val))
    
    candidates.sort(key=lambda item: item[2], reverse=True)
    
    for gx, gy, val in candidates:
        sx = remap(gx, 0, COLS-1, 0, pg.width)
        sy = remap(gy, 0, ROWS-1, 0, pg.height)
        
        raw_label = get_label_for_location(gx, gy)
        label = clean_text(raw_label)
        
        if not label or label == "Unknown": continue
        
        collision_dist = 150 * (pg.width / WIDTH)
        too_close = False
        for px, py, p_txt in placed_labels:
            dist = np.sqrt((sx - px)**2 + (sy - py)**2)
            if dist < collision_dist: 
                too_close = True
                break
        
        if not too_close:
            x_dir = 1 if sx < pg.width/2 else -1
            y_dir = 1 if sy < pg.height/2 else -1
            elbow_x = sx + (40 * x_dir * (pg.width / WIDTH))
            elbow_y = sy + (40 * y_dir * (pg.width / WIDTH))
            end_x = elbow_x + (120 * x_dir * (pg.width / WIDTH)) 
            
            pg.stroke_weight(1.5 * RESOLUTION_MULTIPLIER)
            pg.stroke(255, 200) 
            pg.no_fill()
            
            pg.begin_shape()
            pg.vertex(sx, sy)
            pg.vertex(elbow_x, elbow_y)
            pg.vertex(end_x, elbow_y)
            pg.end_shape()
            
            pg.fill(0)
            pg.stroke(255)
            pg.circle(sx, sy, 5)
            
            pg.fill(255)
            pg.no_stroke()
            
            if x_dir == 1:
                pg.text_align(py5.LEFT)
                pg.text(label, elbow_x + 5, elbow_y - 8)
            else:
                pg.text_align(py5.RIGHT)
                pg.text(label, elbow_x - 5, elbow_y - 8)
            
            placed_labels.append((sx, sy, label))

def clean_text(text):
    text = re.sub(r' - Google Chrome.*', '', text)
    text = re.sub(r' - Mozilla Firefox.*', '', text)
    text = re.sub(r' - Edge.*', '', text)
    text = re.sub(r' - Visual Studio Code', '', text)
    text = re.sub(r' \u2022 VS Code', '', text) 
    if "\\" in text:
        text = text.split("\\")[-1] 
    if len(text) > 50:
        text = text[:47] + "..."
    return text.strip()

def apply_vignette(pg):
    print("Applying Vignette...")
    pg.no_stroke()
    pg.rect_mode(py5.CORNER)
    for i in range(100):
        alpha = remap(i, 0, 100, 255, 0)
        pg.fill(0, alpha * 0.5)
        pg.rect(0, i, pg.width, 1) # Top
        pg.rect(0, pg.height-i, pg.width, 1) # Bottom
    for i in range(100):
        alpha = remap(i, 0, 100, 255, 0)
        pg.fill(0, alpha * 0.5)
        pg.rect(i, 0, 1, pg.height) # Left
        pg.rect(pg.width-i, 0, 1, pg.height) # Right

def apply_grain(pg, intensity=0.03):
    print("Applying Grain...")
    pg.stroke_weight(1)
    for _ in range(int(pg.width * pg.height * intensity)):
        x = py5.random(pg.width)
        y = py5.random(pg.height)
        pg.stroke(255 if py5.random(1) > 0.5 else 0, 20)
        pg.point(x, y)

def add_hill(cx, cy, radius, strength, color, intensity):
    r, g, b = color
    for x in range(cx - radius, cx + radius + 1):
        for y in range(cy - radius, cy + radius + 1):
            if 0 <= x < COLS and 0 <= y < ROWS:
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                if dist < radius:
                    factor = strength * (1 - dist/radius)
                    density_grid[x][y] += factor
                    color_grid[x][y][0] += r * factor
                    color_grid[x][y][1] += g * factor
                    color_grid[x][y][2] += b * factor
                    
                    # Accumulate the intensity (Keys + Mouse)
                    intensity_grid[x][y] += intensity * factor

def get_label_for_location(gx, gy):
    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]
    target_x = remap(gx, 0, COLS-1, min(xs), max(xs))
    target_y = remap(gy, 0, ROWS-1, min(ys), max(ys))
    best_dist = float('inf')
    best_label = "Unknown"
    for p in points:
        dist = (p['x'] - target_x)**2 + (p['y'] - target_y)**2
        if dist < best_dist:
            best_dist = dist
            best_label = p['window_title']
    return best_label

def remap(val, low, high, new_low, new_high):
    return new_low + (val - low) * (new_high - new_low) / (high - low)

if __name__ == "__main__":
    py5.run_sketch()