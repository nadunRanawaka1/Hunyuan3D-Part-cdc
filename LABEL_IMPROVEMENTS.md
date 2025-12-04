# Label & Visualization Improvements

## Summary of Changes

All requested improvements have been implemented! ✅

## 1. ✅ Sequential Label Numbering (0, 1, 2, ...)

**Before:** Labels used original part IDs (could be: 407, 817, 1228, etc.)
**After:** Labels are renumbered from 0 to N-1

### Implementation:
- Created `label_mapping` dictionary: `{old_id: new_id}`
- New IDs assigned sequentially: 0, 1, 2, 3, ...
- Part info now uses new IDs

### Code:
```python
label_mapping = {int(old_id): new_id for new_id, old_id in enumerate(valid_ids)}
```

---

## 2. ✅ Save Label Mapping as JSON

**Location:** `{output_dir}/label_mapping.json`

### Example JSON:
```json
{
  "407": 0,
  "817": 1,
  "1228": 2,
  "388": 3,
  "514": 4
}
```

### Usage:
```python
import json
with open('exploded_renders/label_mapping.json') as f:
    mapping = json.load(f)
    
# Convert new label back to original
old_label = mapping[str(new_label)]
```

---

## 3. ✅ Closer Camera Position

**Before:** `camera_distance = scene_scale * 2.5`
**After:** `camera_distance = scene_scale * 2.0`

**Result:** Camera is 20% closer to the object

---

## 4. ✅ High-Contrast Colors

**Before:** Used matplotlib's `tab20` colormap (limited, repetitive)
**After:** Golden ratio-based HSV color distribution

### Implementation:
```python
for i in range(n_parts):
    hue = (i * 0.618033988749895) % 1.0  # Golden ratio
    saturation = 0.9  # High saturation
    value = 0.85      # Bright
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
```

### Benefits:
- **Maximally distinct colors** using golden ratio sequence
- **High saturation** (0.9) for vibrant colors
- **Bright values** (0.85) for visibility
- **No repetition** - unique color for each part
- **Perceptually uniform** spacing

---

## 5. ✅ Labels Same Color as Segments

**Before:** White background with black text for all labels
**After:** Each label has the same color as its segment

### Features:
- Background color matches segment color
- **Automatic contrast detection** for text:
  - Black text on bright segments
  - White text on dark segments
- **Contrasting outline**:
  - Black outline on bright backgrounds
  - White outline on dark backgrounds
- **Semi-transparent** background (alpha=200)

### Code:
```python
label_color = segment_color
brightness = (R * 0.299 + G * 0.587 + B * 0.114)
text_color = BLACK if brightness > 128 else WHITE
outline_color = BLACK if brightness > 128 else WHITE
```

---

## 6. ✅ Only Show Visible Labels

**Before:** All labels shown regardless of visibility
**After:** Only shows labels for segments facing the camera

### Visibility Check Algorithm:
1. **Calculate view direction** from segment center to camera
2. **Dot product** of segment normal with view direction
3. **Threshold check**: `dot_product > 0.1`
   - > 0: Segment faces camera
   - < 0: Segment faces away
   - 0.1 threshold avoids grazing angles

### Benefits:
- **Cleaner images** - no label clutter
- **Context-aware** - shows only relevant parts
- **Dynamic** - changes per view angle
- **Performance** - fewer labels to draw

### Debug Output:
```
Drawing 7 visible labels  # Out of 12 total parts
```

---

## Visual Comparison

### Before:
- ❌ Random label numbers (407, 817, 1228...)
- ❌ All labels same color (white/black)
- ❌ All labels shown (cluttered)
- ❌ Camera too far
- ❌ Similar/repetitive colors

### After:
- ✅ Sequential labels (0, 1, 2, 3...)
- ✅ Labels match segment colors
- ✅ Only visible labels shown
- ✅ Camera closer (better view)
- ✅ High-contrast distinct colors
- ✅ Larger, more readable labels (48pt font)

---

## Output Files

### Generated Files:
```
exploded_renders/
├── label_mapping.json          # OLD_ID → NEW_ID mapping
├── render_view00.png           # Rendered views with labels
├── render_view01.png
├── ...
└── exploded_mesh.glb          # 3D mesh file
```

### Label Mapping JSON:
```json
{
  "407": 0,
  "817": 1,
  "1228": 2,
  "388": 3,
  "514": 4,
  "677": 5,
  "789": 6,
  "520": 7,
  "829": 8,
  "314": 9,
  "286": 10,
  "527": 11
}
```

---

## Usage Examples

### Basic Usage:
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --renderer pyrender
```

### With Icosahedron Views (12 angles):
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode icosahedron \
    --renderer pyrender
```

### High Resolution:
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode icosahedron \
    --resolution 3840 2160 \
    --renderer pyrender
```

---

## Technical Details

### Color Generation (Golden Ratio):
```python
# Golden ratio provides maximal visual distinction
φ = 0.618033988749895  # (√5 - 1) / 2
hue_i = (i * φ) % 1.0   # Wraps around [0, 1)

# High saturation & brightness for visibility
saturation = 0.9
value = 0.85
```

### Visibility Calculation:
```python
view_direction = normalize(camera_pos - part_center)
part_normal = average_face_normals(part)
dot_product = dot(part_normal, view_direction)

is_visible = dot_product > 0.1
```

### Label Contrast:
```python
brightness = 0.299*R + 0.587*G + 0.114*B  # Perceived brightness
text_color = BLACK if brightness > 128 else WHITE
```

---

## Benefits Summary

1. **Sequential Labels (0-N)**
   - Easier to reference
   - More intuitive
   - Professional appearance

2. **Label Mapping JSON**
   - Traceability to original IDs
   - Easy integration with other tools
   - Programmatic access

3. **Closer Camera**
   - Better detail visibility
   - Fills frame better
   - More engaging views

4. **High-Contrast Colors**
   - Maximally distinct
   - Easy part differentiation
   - Perceptually uniform

5. **Color-Matched Labels**
   - Visual consistency
   - Easy association part↔label
   - Professional look

6. **Visibility Filtering**
   - Cleaner images
   - Less clutter
   - Context-appropriate

---

## Migration Notes

### Backward Compatibility:
- Old angle specification still works
- Default camera mode is `icosahedron` now
- All previous features maintained

### Breaking Changes:
- `create_scene_from_parts()` now returns 3 values instead of 2:
  ```python
  scene, part_info, label_mapping = create_scene_from_parts(mesh, face_ids)
  ```

---

## Examples

### Load Label Mapping:
```python
import json

# Load mapping
with open('exploded_renders/label_mapping.json') as f:
    mapping = json.load(f)

# Use mapping
print(f"Part 0 was originally part {mapping['0']}")

# Reverse mapping (new → old)
reverse = {v: k for k, v in mapping.items()}
```

### Check Which Labels Are Visible:
Check the console output when rendering:
```
Rendering view 1/12
  Drawing 7 visible labels  # 7 out of 12 parts visible
```

---

## Testing

Try different views to see visibility filtering in action:

```bash
# Front view - sees different parts than back view
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode angles \
    --angles "0,30 180,30" \
    --renderer pyrender
```

Compare the two views - you'll see different labels!

---

## Future Enhancements

Possible future improvements:
- [ ] Adjustable visibility threshold
- [ ] Label size based on part size
- [ ] Connecting lines from labels to parts
- [ ] Export label positions to JSON
- [ ] Custom label text (not just numbers)
- [ ] Label grouping/hierarchies

---

## Credits

Color generation uses the **golden ratio** method for optimal perceptual distinction.

Visibility check uses standard **backface culling** technique from computer graphics.

---

Enjoy your improved labeled renders! 🎨

