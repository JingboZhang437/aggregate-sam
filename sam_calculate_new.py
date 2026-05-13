import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import math
import pickle

# Display image with matplotlib
def plt_show(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Global variables for point selection and masks
src_points = []
masks = []
selected_masks = []

# Merge overlapping masks (keep larger masks)
def click_merge(raw_image, image, masks):
    mask_list = masks.copy()
    mask_list.sort(key=lambda m: m['area'])
    
    for i in range(len(mask_list)):
        if mask_list[i] is None:
            continue
        mask_a = mask_list[i]
        
        for j in range(i + 1, len(mask_list)):
            if mask_list[j] is None:
                continue
            mask_b = mask_list[j]
            
            if np.logical_and(mask_a['segmentation'], mask_b['segmentation']).any():
                if mask_a['area'] < mask_b['area']:
                    mask_list[i] = None
                else:
                    mask_list[j] = None

    new_masks = [m for m in mask_list if m is not None]
    result_img = raw_image.copy()
    
    for mask in new_masks:
        mask_img = mask['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        result_img = cv2.drawContours(result_img, contours, -1, (0, 0, 255), 2)
    
    return new_masks, result_img

# Remove incorrect masks by mouse clicking
def click_remove(raw_image, image, masks):
    global selected_masks
    selected_masks = []
    click_points = []

    fig_interact = plt.figure(figsize=(8, 8))
    plt.imshow(image, aspect="equal")
    plt.title('Click wrong segmentation masks to remove (ESC to finish)', fontsize=10, pad=10)
    plt.axis('off')

    points = plt.ginput(n=-1, timeout=0)
    img_before = image.copy()
    
    for point in points:
        x, y = int(point[0]), int(point[1])
        click_points.append((x, y))
        print(f"Point clicked: ({x}, {y})")
        mask_indices = []
        
        for i, mask in enumerate(masks):
            if mask['segmentation'][y, x]:
                mask_indices.append(i)
        
        if len(mask_indices) != 0:
            for mask_to_remove in sorted(mask_indices, reverse=True):
                masks.pop(mask_to_remove)
    
    plt.close(fig_interact)

    for i in range(len(masks)):
        mask_i = masks[i]['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(mask_i, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        raw_image = cv2.drawContours(raw_image, contours, -1, (0, 0, 255), 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.5))
    plt.subplots_adjust(wspace=0.2)
    ax1.imshow(img_before)
    ax1.set_title("Before Removal", fontsize=10)
    ax1.axis('off')
    
    for (x, y) in click_points:
        ax1.scatter(x, y, color='red', s=50)
    
    ax2.imshow(raw_image)
    ax2.set_title("After Removal", fontsize=10)
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
    
    return masks, raw_image

# Legacy corner selection function (basic version)
def select_corner_points0(image):
    global src_points
    src_points = []

    fig_interact = plt.figure(figsize=(12, 12))
    plt.imshow(image, aspect="equal")
    plt.title('Select 4 calibration corners (ESC to finish)', fontsize=10, pad=10)
    plt.axis('off')

    points = plt.ginput(n=-1, timeout=0)
    plt.close(fig_interact)

    for p in points[:4]:
        src_points.append([int(p[0]), int(p[1])])
        print(f"Point added: ({int(p[0])}, {int(p[1])})")

# Enhanced corner selection: click + arrow key adjustment + green cross marker
def select_corner_points(image):
    global src_points
    src_points = []
    selected_points = []

    # Match window style with click_remove for automatic centering
    fig_interact = plt.figure(figsize=(12, 12))
    ax = plt.gca()
    plt.imshow(image, aspect="equal")
    plt.title('Select 4 corners | Use arrow keys to fine‑tune the point | ESC to finish', fontsize=10)
    plt.axis('off')

    # Light green cross marker (large size, thin line)
    point_markers = ax.plot([], [], '+', color='#39ff14', markersize=14, markeredgewidth=1.2)[0]

    # Keyboard event: adjust last point / close window with ESC
    def on_key_event(event):
        if event.key == 'escape':
            plt.close(fig_interact)
            return
        if not selected_points:
            return
        
        step = 1
        x, y = selected_points[-1]
        if event.key == 'up':
            y -= step
        elif event.key == 'down':
            y += step
        elif event.key == 'left':
            x -= step
        elif event.key == 'right':
            x += step
        
        selected_points[-1] = (x, y)
        point_markers.set_data(list(zip(*selected_points)))
        fig_interact.canvas.draw()

    # Mouse event: add point on click
    def add_point(event):
        if event.xdata is not None and event.ydata is not None:
            selected_points.append((event.xdata, event.ydata))
            point_markers.set_data(list(zip(*selected_points)))
            fig_interact.canvas.draw()

    # Bind mouse and keyboard events
    fig_interact.canvas.mpl_connect('button_press_event', add_point)
    fig_interact.canvas.mpl_connect('key_press_event', on_key_event)

    # Blocking input for window centering
    points = plt.ginput(n=-1, timeout=0)
    plt.close(fig_interact)

    # Save final 4 points
    src_points = [[int(round(x)), int(round(y))] for x, y in selected_points[:4]]
    print(f"Final selected points: {src_points}")

# Check if two bounding boxes overlap
def bbox_overlap(bbox1, bbox2):
    x1_min, y1_min, w1, h1 = bbox1
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_min, y2_min, w2, h2 = bbox2
    x2_max, y2_max = x2_min + w2, y2_min + h2
    return x1_min < x2_max and x1_max > x2_min and y1_min < y2_max and y1_max > y2_min

# Calculate Intersection over Union of two masks
def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

# Remove highly overlapping masks
def remove_overlapping_masks(masks):
    masks = sorted(masks, key=lambda x: x["area"])
    kept_masks = []
    
    for i, mask_i in enumerate(masks):
        overlap_flag = False
        for j in kept_masks:
            mask_j = masks[j]
            if bbox_overlap(mask_i["bbox"], mask_j["bbox"]) and iou(mask_i["segmentation"], mask_j["segmentation"]) > 0.5:
                overlap_flag = True
                break
        if not overlap_flag:
            kept_masks.append(i)
    
    return [masks[i] for i in kept_masks]

if __name__ == "__main__":
    # Configuration parameters
    points_per_side = 24
    image_path = r'1.jpg'
    real_length = 800
    model_type = "vit_b"
    device = "cuda"
    sam_checkpoint1 = r"sam_vit_b_01ec64.pth"

    # Output paths
    excel_path = r'output/res_excel.xlsx'
    mask_path = r'output/res_mask.pkl'
    img_folder_path = r'output/res_img'
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    os.makedirs(img_folder_path, exist_ok=True)

    img_path0 = os.path.join(img_folder_path, '0.png')
    img_path1 = os.path.join(img_folder_path, '1.png')
    img_path2 = os.path.join(img_folder_path, '2.png')
    img_path3 = os.path.join(img_folder_path, '3.png')
    img_path4 = os.path.join(img_folder_path, '4.png')
    img_path5 = os.path.join(img_folder_path, '5.png')

    # Read and preprocess image
    image = cv2.imread(image_path, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale_percent = 30
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Select 4 calibration corners
    select_corner_points(image)

    # Perspective transformation
    if len(src_points) == 4:
        src = np.float32(src_points)
        tgt_size = min(image.shape[:2])
        tgt = np.float32([[0, 0], [0, tgt_size], [tgt_size, tgt_size], [tgt_size, 0]])
        M = cv2.getPerspectiveTransform(src, tgt)
        result = cv2.warpPerspective(image, M, (tgt_size, tgt_size))
        plt_show(result)

    cv2.destroyAllWindows()
    cv2.imwrite(img_path0, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    result_h, result_w = result.shape[:2]
    factor = real_length / tgt_size

    # Load SAM model
    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint1)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=points_per_side)
    masks = mask_generator.generate(result)

    # Remove overly large masks
    max_area_threshold = 0.5 * result_w * result_h
    for i in reversed(range(len(masks))):
        if masks[i]['area'] > max_area_threshold:
            masks.pop(i)
            break

    # Process mask contours
    img_tmp0 = result.copy()
    for i in range(len(masks)):
        mask_img = masks[i]['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        max_contour = max(contours, key=cv2.contourArea)
        new_mask = np.zeros_like(mask_img)
        new_mask = np.ascontiguousarray(new_mask)
        cv2.drawContours(new_mask, [max_contour], -1, 255, cv2.FILLED)
        masks[i]['segmentation'] = new_mask.astype(bool)
        cv2.drawContours(img_tmp0, [max_contour], -1, (0, 0, 255), 2)

    cv2.imwrite(img_path1, cv2.cvtColor(img_tmp0, cv2.COLOR_RGB2BGR))

    # Merge masks
    img_tmp1 = result.copy()
    masks_after_merge1, image_after_merge1 = click_merge(img_tmp1, img_tmp0, masks)
    cv2.imwrite(img_path2, cv2.cvtColor(image_after_merge1, cv2.COLOR_RGB2BGR))

    # Remove incorrect masks
    img_tmp2 = result.copy()
    masks_after_remove1, image_after_remove1 = click_remove(img_tmp2, image_after_merge1, masks_after_merge1)
    cv2.imwrite(img_path3, cv2.cvtColor(image_after_remove1, cv2.COLOR_RGB2BGR))

    # Deduplicate overlapping masks
    print(f"Number of masks before deduplication: {len(masks_after_remove1)}")
    filtered_masks = remove_overlapping_masks(masks_after_remove1)
    print(f"Number of masks after deduplication: {len(filtered_masks)}")

    # Save masks
    with open(mask_path, 'wb') as file:
        pickle.dump(filtered_masks, file)

    # Draw final masks
    img_tmp9 = result.copy()
    for mask in filtered_masks:
        mask_img = mask['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img_tmp9 = cv2.drawContours(img_tmp9, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(img_path4, cv2.cvtColor(img_tmp9, cv2.COLOR_RGB2BGR))

    # Calculate aggregate parameters (ellipse fitting)
    major_axis = []
    minor_axis = []
    mask_area = []
    img_tmp99 = result.copy()
    
    for mask in filtered_masks:
        mask_img = mask['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) >= 5:
                retval = cv2.fitEllipse(contour)
                img_tmp99 = cv2.ellipse(img_tmp99, retval, (0, 0, 255), 2)
                major_axis.append(retval[1][1])
                minor_axis.append(retval[1][0])
                mask_area.append(mask['area'])
    
    cv2.imwrite(img_path5, cv2.cvtColor(img_tmp99, cv2.COLOR_RGB2BGR))

    # Export results to Excel
    df = pd.DataFrame()
    df["a"] = np.array(major_axis) * factor
    df["b"] = np.array(minor_axis) * factor
    df["area"] = np.array(mask_area) * factor * factor
    df['volume'] = 4 / 3 * math.pi * 0.125 * df['a'] * df['b'] * 0.7 * df['b']
    df.to_excel(excel_path, index=False)