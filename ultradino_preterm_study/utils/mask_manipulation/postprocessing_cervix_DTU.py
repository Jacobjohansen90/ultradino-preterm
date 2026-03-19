import cv2
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion
from PIL import Image

from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix



from scipy import ndimage

def clean_small_components(mask, min_size=100):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):  # Ignorer le fond (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    return cleaned


def clean_mask(mask, min_size=50):
    
    # === Masque de sortie (même taille, valeurs 0 par défaut) ===
    final_mask = np.zeros_like(mask, dtype=np.uint8)

    # === Traitement classe par classe ===
    for class_id in [1, 2, 3]:
        #print(f"Nettoyage de la classe {class_id}")
        class_mask = (mask == class_id).astype(np.uint8) * 255
        cleaned = clean_small_components(class_mask, min_size=min_size)
        final_mask[cleaned == 255] = class_id  # réintégration dans le masque final

    # === Résultat prêt ===
    return final_mask

from skimage.morphology import skeletonize

def skeletonize_mask(binary_mask):
    # Doit être booléen pour skimage
    skeleton = skeletonize(binary_mask > 0)
    return skeleton.astype(np.uint8)

from scipy.ndimage import convolve

def find_skeleton_endpoints(skeleton):
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = convolve(skeleton, kernel, mode='constant')
    endpoints = np.logical_and(skeleton == 1, neighbor_count == 1)
    yx = np.argwhere(endpoints)
    return [tuple(reversed(pt)) for pt in yx]  # (x, y)

def get_points_skel(mask, value):
    
    canal_mask = (mask == value).astype(np.uint8) * 255
    skeleton = skeletonize_mask(canal_mask)
    return find_skeleton_endpoints(skeleton)

    
def get_points(mask, value):

    canal_mask = (mask == value).astype(np.uint8) * 255
    contours, _ = cv2.findContours(canal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # On suppose qu'on garde le plus grand contour
    canal_contour = max(contours, key=len).squeeze()

    # Extraire les extrémités gauche et droite (par x)
    canal_sorted = sorted(canal_contour, key=lambda pt: pt[0])
    pt_left = canal_sorted[0]
    pt_right = canal_sorted[-1]

    return pt_left, pt_right

def get_middle(mask, value):

    canal_mask = (mask == value).astype(np.uint8) * 255
    contours, _ = cv2.findContours(canal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # On suppose qu'on garde le plus grand contour
    canal_contour = max(contours, key=len).squeeze()

    # Extraire les extrémités gauche et droite (par x)
    canal_sorted = sorted(canal_contour, key=lambda pt: pt[0])
    pt_middle = canal_sorted[len(canal_sorted)//2]

    return pt_middle
    

def check_two_components(mask, class_id):
    binary = (mask == class_id).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary)
    return num_labels - 1  # soustraire le fond

def get_extremities_per_component(mask, class_id):
    binary = (mask == class_id).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary)
    extremities = []
    for i in range(1, num_labels):  # ignorer le fond
        
        component = (labels == i).astype(np.uint8) 
        component = binary_dilation(skeletonize(binary_dilation(component, iterations=3)), iterations=2).astype(np.uint8)
        extremities.append(get_points_nearest(component, 1))
        
        #component = binary_dilation(component).astype(np.uint8) * 255
        #contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #contour = max(contours, key=len).squeeze()
        #sorted_by_x = sorted(contour, key=lambda pt: pt[0])
        #extremities.append((sorted_by_x[0], sorted_by_x[-1]))  # (leftmost, rightmost)
    return extremities  # liste de tuples : [ (ptL1, ptR1), (ptL2, ptR2) ]

def get_extremities_per_component_skel(mask, class_id):
    binary = (mask == class_id).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary)
    extremities = []
    for i in range(1, num_labels):  # ignorer le fond
        
        extremities.append(get_points_skel(labels, i))  # (leftmost, rightmost)
    return extremities  # liste de tuples : [ (ptL1, ptR1), (ptL2, ptR2) ]

def draw_line(mask, pt1, pt2, value=1, thickness=2):
    """
    Trace une ligne entre pt1 et pt2 sur le masque donné.
    
    - pt1, pt2 : tuples (x, y)
    - value : valeur à inscrire dans le masque
    """
    mask_out = mask.copy()
    cv2.line(mask_out, pt1, pt2, color=value, thickness=thickness)
    return mask_out


def erase_joints(mask_clean, indice_min, rayon=4):
        # Créer un masque temporaire du même type et même taille, rempli de zéros
        temp_mask = np.zeros_like(mask_clean)

        # Dessiner le cercle dans ce masque temporaire
        cv2.circle(temp_mask, (indice_min[1], indice_min[0]), rayon, 1, thickness=-1)

        # N'appliquer le masque que là où les valeurs ne sont pas égales à 1 dans mask_clean
        mask_clean[(temp_mask == 1) & (mask_clean != 1)] = 0
        
        return mask_clean

def join_cervix(mask, rayon=4, join_outer=False):
    
    mask_clean = clean_mask(mask, 20)
    pt_left, pt_right = get_points_nearest(mask_clean, 1)

    # Comment savoir lequel est à gauche ?
    # Plus proche de la bordure extérieure
    # Transformée de distance par rapport aux deux points

    dist_transf = {} # distance transforms wrt cervical canal extremities; useful for finding OB and IB landmarks
    distances = [] # Plus petite distance entre IB et extrémités CC: distingue gauche et droite CC
    middle_points = {} # Points vers le milieu IB et OB pour orienter les extrémités.
    points_ref = {} # Points de référence finaux.


    for point in [pt_left, pt_right]:

        mask_test = mask_clean.copy()
        mask_test[point[1], point[0]] = 10
        masque = (mask_test != 10).astype(np.uint8)
        distance = distance_transform_edt(masque)
        dist_transf[str(point)] = distance.copy()

        distance[mask_clean!=3] = 100

        distances.append(distance.min())

    points_ref['C_L'] = [pt_left, pt_right][np.argmin(np.array(distances))]
    points_ref['C_R'] = [pt_left, pt_right][np.argmax(np.array(distances))]

    # Couper en deux les bordures si ce n'est déjà fait
    ## Bordure extérieure
    distance = dist_transf[str(points_ref['C_L'])].copy()
    distance[mask_clean!=3] = 100
    indice_min = np.unravel_index(np.argmin(distance), distance.shape)
    middle_points['IB'] = (indice_min[1], indice_min[0])
    
    #clean_bord_ext = cv2.circle(mask_clean, (indice_min[1], indice_min[0]), rayon, 0, thickness=-1)
    mask_clean = erase_joints(mask_clean, indice_min, rayon=rayon)
    
    
    ## Bordure Intérieure
    distance = dist_transf[str(points_ref['C_R'])].copy()
    distance[mask_clean!=2] = 100
    indice_min = np.unravel_index(np.argmin(distance), distance.shape)
    middle_points['OB'] = (indice_min[1], indice_min[0])
    
    mask_clean = erase_joints(mask_clean, indice_min, rayon=rayon)
    #clean_bord_int = cv2.circle(mask_clean, (indice_min[1], indice_min[0]), rayon, 0, thickness=-1)

    mask_clean = clean_mask(mask_clean, min_size=20)
    #mask_clean = clean_mask(mask_clean,min_size=10)
    #print("Composants Classe 2 :", check_two_components(mask_clean, 2))  # Doit être 2
    #print("Composants Classe 3 :", check_two_components(mask_clean, 3))  # Doit être 2


    
    
    # Nommer les points 
    extremities = get_extremities_per_component_skel(mask_clean, 2)
    dist_middle = np.array([euclidean(ext, middle_points['OB']) for ext in extremities[0]])
    points_ref['OB_UL'], points_ref['OB_UR'] = extremities[0][np.argmax(dist_middle)], extremities[0][np.argmin(dist_middle)]

    dist_middle = np.array([euclidean(ext, middle_points['OB']) for ext in extremities[1]])
    points_ref['OB_DL'], points_ref['OB_DR'] = extremities[1][np.argmax(dist_middle)], extremities[1][np.argmin(dist_middle)]

    # Nommer les points 
    extremities = get_extremities_per_component_skel(mask_clean, 3)
    dist_middle = np.array([euclidean(ext, middle_points['IB']) for ext in extremities[0]])
    points_ref['IB_UL'], points_ref['IB_UR'] = extremities[0][np.argmax(dist_middle)], extremities[0][np.argmin(dist_middle)]

    dist_middle = np.array([euclidean(ext, middle_points['IB']) for ext in extremities[1]])
    points_ref['IB_DL'], points_ref['IB_DR'] = extremities[1][np.argmax(dist_middle)], extremities[1][np.argmin(dist_middle)]
    
    points_ref = correct_points_ref(points_ref)
    
    if not join_outer:
        a = euclidean(points_ref['C_L'], points_ref['IB_UR'])
        b = euclidean(points_ref['C_L'], points_ref['IB_DR'])
        c = euclidean(points_ref['OB_UL'], points_ref['IB_UL'])
        d = euclidean(points_ref['OB_DL'], points_ref['IB_DL'])
        g = euclidean(points_ref['C_R'], points_ref['OB_UR'])
        h = euclidean(points_ref['C_R'], points_ref['OB_DR'])

        for dist in [a,b,g,h]:
            if dist > 70:
                return 0,0,0
                
        for dist in [c,d]:
            if dist > 150: #previous 130
                return 0,0,0    
        mask_clean = draw_line(mask_clean, points_ref['C_L'], points_ref['IB_UR'], value=4)
        mask_clean = draw_line(mask_clean, points_ref['C_L'], points_ref['IB_DR'], value=4)

        mask_clean = draw_line(mask_clean, points_ref['OB_UL'], points_ref['IB_UL'], value=4)
        mask_clean = draw_line(mask_clean, points_ref['OB_DL'], points_ref['IB_DL'], value=4)
    else:
         mask_clean = draw_line(mask_clean, points_ref['C_L'], points_ref['OB_UL'], value=4)
         mask_clean = draw_line(mask_clean, points_ref['C_L'], points_ref['OB_DL'], value=4)
             
    mask_clean = draw_line(mask_clean, points_ref['C_R'], points_ref['OB_UR'], value=4)
    mask_clean = draw_line(mask_clean, points_ref['C_R'], points_ref['OB_DR'], value=4)



    
    # Remplissement
    
    return mask_clean, points_ref, dist_transf
    
def correct_points_ref(points_ref):
    
    if points_ref['OB_UL'][0] > points_ref['IB_UL'][0]: # Orientation normale
        # Dans ce cas, UL[1] < DL[1]
        if points_ref['OB_UL'][1] > points_ref['OB_DL'][1]:
            temp1, temp2 = points_ref['OB_UL'], points_ref['OB_UR']
            points_ref['OB_UL'], points_ref['OB_UR'] = points_ref['OB_DL'], points_ref['OB_DR']
            points_ref['OB_DL'], points_ref['OB_DR'] = temp1, temp2
        
        if points_ref['IB_UL'][1] > points_ref['IB_DL'][1]:
            temp1, temp2 = points_ref['IB_UL'], points_ref['IB_UR']
            points_ref['IB_UL'], points_ref['IB_UR'] = points_ref['IB_DL'], points_ref['IB_DR']
            points_ref['IB_DL'], points_ref['IB_DR'] = temp1, temp2
            
    else:
        if points_ref['OB_UL'][1] < points_ref['OB_DL'][1]:
            temp1, temp2 = points_ref['OB_UL'], points_ref['OB_UR']
            points_ref['OB_UL'], points_ref['OB_UR'] = points_ref['OB_DL'], points_ref['OB_DR']
            points_ref['OB_DL'], points_ref['OB_DR'] = temp1, temp2
        
        if points_ref['IB_UL'][1] < points_ref['IB_DL'][1]:
            temp1, temp2 = points_ref['IB_UL'], points_ref['IB_UR']
            points_ref['IB_UL'], points_ref['IB_UR'] = points_ref['IB_DL'], points_ref['IB_DR']
            points_ref['IB_DL'], points_ref['IB_DR'] = temp1, temp2
    
    return points_ref    
    
from scipy.interpolate import splprep, splev

def order_points_by_nearest(points, start_idx=0, max_step=40):
    """
    Trie les points par proximité en formant un chemin,
    mais arrête si le prochain point est trop loin (max_step).
    """
    

    points = points.copy()
    ordered = [points[start_idx]]
    used = set([start_idx])

    dist_mat = distance_matrix(points, points)
    current_idx = start_idx

    for _ in range(len(points) - 1):
        dist_mat[current_idx, list(used)] = np.inf
        next_idx = np.argmin(dist_mat[current_idx])
        next_dist = dist_mat[current_idx, next_idx]

        if next_dist > max_step:
            #print(f"Arrêt du chemin: saut trop grand ({next_dist:.2f})")
            break  # Ne pas relier au-delà de la distance seuil
        
        ordered.append(points[next_idx])
        used.add(next_idx)
        current_idx = next_idx
      
 
    return np.array(ordered)

def fit_smooth_curve_through_line(binary_line_mask, num_points=300):
    #points = np.argwhere(binary_line_mask > 0)  # (y, x)
    points = get_initial_coords(line_mask)
    
    if len(points) < 2:
        return []

    # Réorganiser les points dans l'ordre
    ordered_points = order_points_by_nearest(points, start_idx=0)
    ordered_points = detect_reverse_and_reorder(ordered_points)
   
    y = ordered_points[:, 0]
    x = ordered_points[:, 1]
    tck, _ = splprep([x, y], s=0.10)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = splev(u_fine, tck)

    curve_points = [(int(round(x)), int(round(y))) for x, y in zip(x_fine, y_fine)]
    return curve_points

def correct_line(line_mask):

    curve_pts = fit_smooth_curve_through_line(line_mask)
    mask_with_curve = draw_curve_on_mask(line_mask, curve_pts, value=2, thickness=2)
    
    return mask_with_curve

def draw_curve_on_mask(mask, curve_points, value=5, thickness=1):
    mask_out = mask.copy()
    for i in range(len(curve_points) - 1):
        cv2.line(mask_out, curve_points[i], curve_points[i + 1], color=value, thickness=thickness)
    return mask_out

def randomly_break_line_varied(line_mask, num_breaks=5, break_range=(5, 50)):
    """
    Coupe une ligne en plusieurs morceaux de longueur variable.
    
    - num_breaks : nombre de coupures
    - break_range : tuple (min_len, max_len) pour la taille aléatoire des coupures
    """
    coords = np.argwhere(line_mask > 0)
    if len(coords) < break_range[0] * num_breaks:
        print("⚠️ Trop peu de points pour autant de coupures.")
        return line_mask

    mask_out = line_mask.copy()
    available = np.arange(len(coords))
    np.random.shuffle(available)

    cuts_done = 0
    i = 0
    while cuts_done < num_breaks and i < len(available):
        idx = available[i]
        center = coords[idx]

        break_len = np.random.randint(break_range[0], break_range[1] + 1)
        dists = np.linalg.norm(coords - center, axis=1)
        close_idxs = np.argsort(dists)[:break_len]
        to_remove = coords[close_idxs]

        for y, x in to_remove:
            mask_out[y, x] = 0

        cuts_done += 1
        i += 1

    return mask_out

import numpy as np

def smooth_direction(points, idx, window=5):
    """Calcule la direction moyenne (normalisée) des `window` points avant idx."""
    if idx < window + 1:
        return None  # pas assez de points

    diffs = points[idx - window:idx] - points[idx - window - 1:idx - 1]
    mean_dir = np.mean(diffs, axis=0)
    norm = np.linalg.norm(mean_dir)
    if norm == 0:
        return None
    return mean_dir / norm

def compute_direction(a, b):
    vec = b - a
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else None

def detect_reverse_and_reorder(points, window=5, angle_threshold=150):
    """
    Détecte un retournement brutal par rapport à la direction lissée (momentum),
    puis réorganise les points pour commencer au bon endroit.
    """
    points = np.array(points)

    for i in range(window + 1, len(points) - 1):
        avg_dir = smooth_direction(points, i, window)
        if avg_dir is None:
            continue

        next_dir = compute_direction(points[i], points[i + 1])
        if next_dir is None:
            continue
        
        cos_angle = np.clip(np.dot(avg_dir, next_dir), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        if angle > angle_threshold:
            
            # On a détecté un "retournement" brutal
            head = points[i + 1:]       # on continue depuis là
            #print(head)
            tail = points[:i + 1][::-1] # on remonte l'autre bout
            
            reordered = np.concatenate([tail, head], axis=0)
            return reordered

    return points  # pas de retournement détecté
    
 
def get_initial_coords(a):
    skel = skeletonize(a, method='lee')
    
    
    coords = np.transpose(np.nonzero(skel))
    return coords
    
       
def join_lines(line):
    
    mask_out = np.zeros(line.shape)
    points = get_initial_coords(line)
    points = [(p[1], p[0]) for p in points]

    points_1 = order_points_by_nearest(points, start_idx=0, max_step=35)
    points_2 = order_points_by_nearest(points, start_idx=-1, max_step=35)
    points_3 = order_points_by_nearest(points, start_idx=len(points)//2, max_step=35)

    curve_1 = draw_curve_on_mask(mask_out, points_1, value=1, thickness=1)
    curve_2 = draw_curve_on_mask(mask_out, points_2, value=1, thickness=1)
    curve_3 = draw_curve_on_mask(mask_out, points_3, value=1, thickness=1)

    curve_final = curve_1 + curve_2 + curve_3
    curve_final[curve_final != 0] = 1
    curve_final = binary_dilation(curve_final)
    
    return curve_final
    
    
def join_lines(line, max_step=35, angle=170):
    
    mask_out = np.zeros(line.shape)
    points = get_initial_coords(line)
    points = [(p[1], p[0]) for p in points]

    points_1 = order_points_by_nearest(points, start_idx=0, max_step=max_step)
    points_2 = order_points_by_nearest(points, start_idx=-1, max_step=max_step)
    points_3 = order_points_by_nearest(points, start_idx=len(points)//2, max_step=max_step)
    
    points_1 = detect_reverse_and_reorder(points_1, angle_threshold=angle)
    points_2 = detect_reverse_and_reorder(points_2, angle_threshold=angle)
    points_3 = detect_reverse_and_reorder(points_3, angle_threshold=angle)
    
    curve_1 = draw_curve_on_mask(mask_out, points_1, value=1, thickness=1)
    curve_2 = draw_curve_on_mask(mask_out, points_2, value=1, thickness=1)
    curve_3 = draw_curve_on_mask(mask_out, points_3, value=1, thickness=1)

    curve_final = curve_1 + curve_2 + curve_3
    curve_final[curve_final != 0] = 1
    curve_final = binary_dilation(curve_final)
    
    return curve_final
        
def preprocess_cervix(mask_clean, method = "dist_matrix", break_randomly=False):
    
    final_mask = np.zeros(mask_clean.shape)
    corrupted_mask = np.zeros(mask_clean.shape)
    
    for value in [1,2,3]:
        
        line_ori = (mask_clean == value).astype(np.uint8)  # canal cervical
        
        
        if break_randomly:
            
            line_ori = randomly_break_line_varied(line_ori)
            corrupted_mask[line_ori == 1] = value
        
        line_ori = binary_dilation(skeletonize(binary_dilation(line_ori, iterations=3)), iterations=2).astype(np.uint8)
        
        n_comp = check_two_components(line_ori, 1) 
        if n_comp == 1:
           line = line_ori
        # Nouvelle logique: itérer
        
        if method == "bouts":
             line = join_line_bouts(line_ori.copy())
        
        elif method == "dist_matrix":       
                max_step = 35
                while n_comp != 1:
                   line = join_lines(line_ori.copy(), max_step=max_step)
                                   
                   line = binary_dilation(skeletonize(binary_dilation(line, iterations=3)), iterations=2).astype(np.uint8)
                   n_comp = check_two_components(line, 1)
                   #print(n_comp)
                   max_step += 10
        
        line = binary_erosion(line)   
        final_mask[line == 1] = value
    
    return final_mask, corrupted_mask
    
    

import cv2
import numpy as np

def fill_region(mask, seed_point, newVal=10):
    # Copie de l'image d'origine
    filled = mask.copy()

    # Préparation du masque pour floodFill (doit être 2 pixels plus grand)
    h, w = mask.shape
    mask_ff = np.zeros((h+2, w+2), np.uint8)

    # Remplissage
    cv2.floodFill(filled, mask_ff, seedPoint=seed_point, newVal=newVal)

    return filled


def fill_cervix(mask_linked, points_ref):
    # Barycentre de 'C_L', 'C_R', 'IB_UL', 'OB_UL'
    
    
    barycentre = [0,0]
    for point in ['C_L', 'IB_UL', 'OB_UL']:
        barycentre[0] += points_ref[point][0]
        barycentre[1] += points_ref[point][1]
    barycentre = (np.array(barycentre)/3).astype(np.uint8)

    upper_cervix = fill_region(mask_linked, barycentre, 5).astype(np.uint8)
    # Barycentre de 'C_L', 'C_R', 'IB_DL', 'OB_DL'

    barycentre = [0,0]
    for point in ['C_L', 'IB_DL', 'OB_DL']:
        barycentre[0] += points_ref[point][0]
        barycentre[1] += points_ref[point][1]
    barycentre = (np.array(barycentre)/3).astype(np.uint8)

    lower_cervix = fill_region(mask_linked, barycentre).astype(np.uint8)
    lower_cervix[lower_cervix!= 10] = 0
    lower_cervix[lower_cervix == 10] = 6
    final_cervix = upper_cervix + lower_cervix
    
    return final_cervix
    
def fill_cervix(mask_linked, points_ref):
    # Barycentre de 'C_L', 'C_R', 'IB_UL', 'OB_UL'
    
    C = points_ref['C_R']#get_middle(mask_linked, 1)
    #print(C)
    vect_CU = (points_ref['OB_UL'][0] - C[0], points_ref['OB_UL'][1] - C[1])
    PU_x = (C[0] + 0.2 * vect_CU[0]).astype(np.uint8)
    PU_y = (C[1] + 0.2 * vect_CU[1]).astype(np.uint8)
    #print(PU_x, PU_y)
    upper_cervix = fill_region(mask_linked, (PU_x, PU_y), 5).astype(np.uint8)
    
    vect_CD = (points_ref['OB_DL'][0] - C[0], points_ref['OB_DL'][1] - C[1])
    PD_x = (C[0] + 0.2 * vect_CD[0]).astype(np.uint8)
    PD_y = (C[1] + 0.2 * vect_CD[1]).astype(np.uint8)
    
    lower_cervix = fill_region(mask_linked, (PD_x, PD_y)).astype(np.uint8)
    lower_cervix[lower_cervix!= 10] = 0
    lower_cervix[lower_cervix == 10] = 6
    final_cervix = upper_cervix + lower_cervix
    
    return final_cervix
    
def fill_cervix(mask_linked, points_ref):
    # Barycentre de 'C_L', 'C_R', 'IB_UL', 'OB_UL'
    
    C = points_ref['C_L']#get_middle(mask_linked, 1)
    #print(C)
    vect_CU = (points_ref['OB_UL'][0] - C[0], points_ref['OB_UL'][1] - C[1])
    PU_x = (C[0] + 0.5 * vect_CU[0]).astype(np.uint8)
    PU_y = (C[1] + 0.5 * vect_CU[1]).astype(np.uint8)
    
    
    C = (PU_x, PU_y)
    vect_CU = (points_ref['C_R'][0] - C[0], points_ref['C_R'][1] - C[1])
    PU_x = (C[0] + 0.2 * vect_CU[0]).astype(np.uint8)
    PU_y = (C[1] + 0.2 * vect_CU[1]).astype(np.uint8)
    
    #mask_linked= draw_line(mask_linked, (PU_x, PU_y), points_ref['C_L'], value=1, thickness=2)
    upper_cervix = fill_region(mask_linked, (PU_x, PU_y), 5).astype(np.uint8)
    
    C = points_ref['C_L']
    
    vect_CD = (points_ref['OB_DL'][0] - C[0], points_ref['OB_DL'][1] - C[1])
    PD_x = (C[0] + 0.5 * vect_CD[0]).astype(np.uint8)
    PD_y = (C[1] + 0.5 * vect_CD[1]).astype(np.uint8)
    
    C = (PD_x, PD_y)
    vect_CD = (points_ref['C_R'][0] - C[0], points_ref['C_R'][1] - C[1])
    PD_x = (C[0] + 0.2 * vect_CD[0]).astype(np.uint8)
    PD_y = (C[1] + 0.2 * vect_CD[1]).astype(np.uint8)
    
    #mask_linked= draw_line(mask_linked, (PD_x, PD_y), points_ref['C_L'], value=1, thickness=2)
    lower_cervix = fill_region(mask_linked, (PD_x, PD_y)).astype(np.uint8)
    lower_cervix[lower_cervix!= 10] = 0
    lower_cervix[lower_cervix == 10] = 6
    final_cervix = upper_cervix + lower_cervix
    
    return final_cervix


def fill_cervix(mask_linked, points_ref):
    # Barycentre de 'C_L', 'C_R', 'IB_UL', 'OB_UL'
    
    C = get_middle_nearest(mask_linked, 1)
    D = (points_ref['OB_UL'] + points_ref['IB_UL']) / 2
    #print(C)
    vect_CU = (D[0] - C[0], D[1] - C[1])
    PU_x, PU_y = C
    delta = 1
    while mask_linked[PU_x, PU_y] != 0:
        PU_x = (C[0] + delta * vect_CU[0]).astype(np.uint8)
        PU_y = (C[1] + delta * vect_CU[1]).astype(np.uint8)
        delta += 1
        
    PU_x = (C[0] + (delta+5) * vect_CU[0]).astype(np.uint8)
    PU_y = (C[1] + (delta+5) * vect_CU[1]).astype(np.uint8)
    
    mask_linked= draw_line(mask_linked, (PU_x, PU_y), points_ref['C_L'], value=1, thickness=2)
    upper_cervix = fill_region(mask_linked, (PU_x, PU_y), 5).astype(np.uint8)
    
    D = (points_ref['OB_DL'] + points_ref['IB_DL']) / 2
    #print(C)
    vect_CU = (D[0] - C[0], D[1] - C[1])
    PD_x, PD_y = C
    delta = 1
    while mask_linked[PU_x, PU_y] != 0:
        PD_x = (C[0] + delta * vect_CU[0]).astype(np.uint8)
        PD_y = (C[1] + delta * vect_CU[1]).astype(np.uint8)
        delta += 1
    
    PD_x = (C[0] + (delta+5) * vect_CU[0]).astype(np.uint8)
    PD_y = (C[1] + (delta+5) * vect_CU[1]).astype(np.uint8)
    
    mask_linked= draw_line(mask_linked, (PD_x, PD_y), points_ref['C_L'], value=1, thickness=2)
    lower_cervix = fill_region(mask_linked, (PD_x, PD_y)).astype(np.uint8)
    lower_cervix[lower_cervix!= 10] = 0
    lower_cervix[lower_cervix == 10] = 6
    final_cervix = upper_cervix + lower_cervix
    
    return final_cervix 

def fill_cervix(mask_linked, points_ref):
    # Barycentre de 'C_L', 'C_R', 'IB_UL', 'OB_UL'
    mask_linked = fill_region(mask_linked, (0, 0), 7).astype(np.uint8)
    C = get_middle_nearest(mask_linked, 1)
    
    D = (np.array(points_ref['OB_UL']) + np.array(points_ref['IB_UL'])) // 2
    
    vect_CU = (D[0] - C[0], D[1] - C[1])
    
    PU_x, PU_y = C
    delta = 0.1
    PU_x = (C[0] + delta * vect_CU[0]).astype(np.uint8)
    PU_y = (C[1] + delta * vect_CU[1]).astype(np.uint8)
    while mask_linked[PU_y, PU_x] != 0:
      
        PU_x = (C[0] + delta * vect_CU[0]).astype(np.uint8)
        PU_y = (C[1] + delta * vect_CU[1]).astype(np.uint8)
        delta += 0.01
        
    PU_x = (C[0] + (delta+0.02) * vect_CU[0]).astype(np.uint8)
    PU_y = (C[1] + (delta+0.02) * vect_CU[1]).astype(np.uint8)
    
    #mask_linked= draw_line(mask_linked, (PU_x, PU_y), C, value=1, thickness=2)
    upper_cervix = fill_region(mask_linked, (PU_x, PU_y), 5).astype(np.uint8)
    
    
    D = (np.array(points_ref['OB_DL']) + np.array(points_ref['IB_DL'])) // 2
    
    vect_CU = (D[0] - C[0], D[1] - C[1])
    PD_x, PD_y = C
    delta = 0.1
    PD_x = (C[0] + delta * vect_CU[0]).astype(np.uint8)
    PD_y = (C[1] + delta * vect_CU[1]).astype(np.uint8)
    while upper_cervix[PD_y, PD_x] != 0:
        PD_x = (C[0] + delta * vect_CU[0]).astype(np.uint8)
        PD_y = (C[1] + delta * vect_CU[1]).astype(np.uint8)
        delta += 0.01
    
    PD_x = (C[0] + (delta+0.02) * vect_CU[0]).astype(np.uint8)
    PD_y = (C[1] + (delta+0.02) * vect_CU[1]).astype(np.uint8)
    #print(PD_x, PD_y)
    #mask_linked= draw_line(mask_linked, (PD_x, PD_y), C, value=1, thickness=2)
    
    lower_cervix = fill_region(mask_linked, (PD_x, PD_y)).astype(np.uint8)
    lower_cervix[lower_cervix!= 10] = 0
    lower_cervix[lower_cervix == 10] = 6
    final_cervix = upper_cervix + lower_cervix
    final_cervix[final_cervix == 7] = 0
    
    return final_cervix 
    
def smooth_direction(points, idx, window=5, avant=True, imprimer=False, seuil_norme=3):
    """
    Calcule la direction moyenne (normalisée) des `window` points avant ou après idx,
    en éliminant les déplacements dont la norme dépasse `seuil_norme`.
    """
    if idx < window + 1 and avant:
        return None  # pas assez de points avant
    if idx + window >= len(points) and not avant:
        return None  # pas assez de points après

    if avant:
        diffs = points[idx - window:idx] - points[idx - window - 1:idx - 1]
    else:
        diffs = points[idx:idx + window] - points[idx - 1:idx + window - 1]

    # Filtrer les déplacements trop grands
    
    norms = np.linalg.norm(diffs, axis=1)
    
    valid_diffs = diffs[norms <= seuil_norme]

    if imprimer:
        print(norms)
        print("Déplacements valides :", valid_diffs)

    if len(valid_diffs) == 0:
        return None

    mean_dir = np.mean(valid_diffs, axis=0)
    norm = np.linalg.norm(mean_dir)
    if norm == 0:
        return None
    return mean_dir / norm

def compute_direction(a, b):
    vec = b - a
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else None

def detect_reverse_and_reorder(points, angle_threshold=150):
    """
    Détecte un retournement brutal basé sur un changement de direction,
    et vérifie que ce changement est stable avant de réorganiser les points.
    """
    points = np.array(points)

    for i in range(1, len(points)-1):
        
        origin_dir = compute_direction(points[0], points[i])
        next_dir = compute_direction(points[i], points[i + 1])
        
        if next_dir is None or origin_dir is None:
            continue
        
        cos_angle = np.clip(np.dot(origin_dir, next_dir), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        if angle > angle_threshold:
                #print(points[i])
                
                
                head = points[i + 1:]
                tail = points[:i + 1][::-1]
                
                reordered = np.concatenate([tail, head], axis=0)
                return reordered

    return points # pas de retournement détecté

def reorder_again(points):
    
    dist_matrix = cdist(points, points)
    
    a,b = 0,0

    for n in range(1, len(dist_matrix[0])-1):

        if dist_matrix[0][n] - dist_matrix[0][n-1]< -10:
            print(dist_matrix[0][n])
            a = n

            print(n)
        if dist_matrix[0][n+1] - dist_matrix[0][n] > 10 and a > 0:
            print(dist_matrix[0][n])
            print(n)
            b=n
            break
    
    if a != 0 and b!=0:
        tail_1 = list(points[:a])
        tail_2 = list(points[b+1:])
        head = list(points[a:b+1][::-1])
        
        new = head + tail_1
        final = np.array(new[::-1] + tail_2)
        return final, True
    
    else:
        return points, False
        
        
def get_points_nearest(mask, value):

    line_ori = (mask == value).astype(np.uint8)
    points = get_initial_coords(line_ori)
    points_1 = order_points_by_nearest(points, start_idx=-1, max_step=60)
    points_2 = order_points_by_nearest(points, start_idx=0, max_step=60)
    
    if len(points_1) > len(points_2):
        
        bouts = [points_1[0][::-1], points_1[-1][::-1]]
    
    else:
        
        bouts = [points_2[0][::-1], points_2[-1][::-1]]
        
    return bouts

       
def get_middle_nearest(mask, value):

    line_ori = (mask == value).astype(np.uint8)
    points = get_initial_coords(line_ori)
    points_1 = order_points_by_nearest(points, start_idx=-1, max_step=60)
    points_2 = order_points_by_nearest(points, start_idx=0, max_step=60)
    
    l1 = len(points_1)
    l2 = len(points_2)
    
    if l1 > l2:
    
        middle = points_1[l1//2][::-1]
    
    else:
        
        middle = points_2[l2//2][::-1]
        
    return middle
    
def relier_bouts(line):
  
    labels, num_labels = ndimage.label(line)
    
    dic_bouts = {}
    dic_bords = {}
    
    M_DC = np.zeros((num_labels, num_labels))
    
    for value in range(1, num_labels +1):
        
        bouts = get_points_nearest(labels, value)
        dic_bouts[value] = bouts
    
    for val1 in range(1, num_labels +1):
        
        bouts1 = dic_bouts[val1]
        dic_bords[val1] = {}
                
        for val2 in range(1, num_labels +1):
            if val1 == val2:
                continue
            else:
                bouts2 = dic_bouts[val2]
                d_mat = distance_matrix(bouts1, bouts2)
                d_min = d_mat.min()
                arrete_min = np.unravel_index(np.argmin(d_mat), d_mat.shape)
                
                dic_bords[val1][val2] = [bouts1[arrete_min[0]], bouts2[arrete_min[1]]]
                M_DC[val1-1][val2-1] = d_min
                
    M_DC_sum = np.sum(M_DC, axis=0)
    
    if num_labels < 4:
        bloc_milieu = np.argmin(M_DC_sum) + 1
        
        for value in range(1, num_labels +1):
            if value == bloc_milieu:
                continue
            else:
                pt1, pt2 = dic_bords[bloc_milieu][value]
                #print(pt1)
                line = draw_line(line, pt1, pt2, value=1, thickness=2)
    else:
        
        bloc_ext = np.argmax(M_DC_sum)
        M_DC[bloc_ext, bloc_ext] = 1000
        next_bloc = np.argmin(M_DC[bloc_ext,:]) + 1
        pt1, pt2 = dic_bords[bloc_ext+1][next_bloc]
        line = draw_line(line, pt1, pt2, value=1, thickness=2)
    
    return line

def join_line_bouts(line_broken):
    
    n_comp = check_two_components(line_broken, 1) 
  
    if n_comp != 1:
        line = relier_bouts(line_broken)
        line = binary_dilation(skeletonize(binary_dilation(line, iterations=3)), iterations=2).astype(np.uint8)
    else:
        return line_broken
    
    while n_comp != 1:
        line = relier_bouts(line)
        line = binary_dilation(skeletonize(binary_dilation(line, iterations=3)), iterations=2).astype(np.uint8)
      
        n_comp = check_two_components(line, 1) 
    
    return line
    
def main_direction_filtered(binary, angle_threshold=np.pi/8, vertical_margin=10, min_fraction=0.8):
    labels, num_labels = ndimage.label(binary)

    sizes = ndimage.sum(binary, labels, range(1, num_labels + 1))
    largest_label = np.argmax(sizes) + 1
    largest_component = labels == largest_label

    # Direction principale
    p1, p2 = get_points_nearest(largest_component, 1)
    main_dir = compute_direction(p1, p2)  # vecteur (dx, dy) unitaire
    main_center = np.mean([p1, p2], axis=0)

    filtered = np.zeros_like(binary)

    for lbl in range(1, num_labels + 1):
        component_mask = labels == lbl

        if lbl == largest_label:
            filtered[component_mask] = 1
            continue

        try:
            p1_c, p2_c = get_points_nearest(component_mask, 1)
        except:
            continue  # trop petit ou échec

        local_dir = compute_direction(p1_c, p2_c)
        angle = np.arccos(np.clip(np.dot(main_dir, local_dir), -1.0, 1.0))

        if angle < angle_threshold or np.abs(angle - np.pi) < angle_threshold:
            # Vérifie si la majorité du segment est au-dessus ou au-dessous
            coords = np.column_stack(np.nonzero(component_mask))  # (y, x)
            y_coords = coords[:, 0]
            below = np.sum(y_coords > main_center[1]) / len(y_coords)
            above = np.sum(y_coords < main_center[1]) / len(y_coords)

            # Si la majorité est clairement au-dessus ou au-dessous → ignorer
            if below > min_fraction or above > min_fraction:
                continue

        # Sinon, on garde
        filtered[component_mask] = 1

    return filtered.astype(np.uint8)

    
def transform_mask(mask):
     
    a = main_direction_filtered((mask == 1))
    mask[mask==1] = 0
    mask[a == 1] = 1
    final_mask, corrupted_mask = preprocess_cervix(clean_mask(mask,35), "bouts", False)
    
    rayon=4
    max_rayon=15
    while True:
        try:
            
            mask_linked, points_ref, dist_transf = join_cervix(final_mask, rayon=rayon)
            break  # Succès, on sort de la boucle
        except Exception as e:
            rayon += 1
            if rayon > max_rayon:
                final_mask, corrupted_mask = preprocess_cervix(clean_mask(mask,15), "bouts", False)
                mask_linked, points_ref, dist_transf = join_cervix(final_mask, rayon=rayon, join_outer=True)
                #raise RuntimeError(f"Échec de join_cervix jusqu'à un rayon de {max_rayon}") from e

    if type(points_ref) == int:
        return 0,0,0
    final_cervix = fill_cervix(mask_linked, points_ref)
    
    return final_cervix, final_mask, points_ref, dist_transf
