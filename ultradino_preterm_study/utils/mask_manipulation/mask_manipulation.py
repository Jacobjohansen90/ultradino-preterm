   
import numpy as np
import cv2

from scipy.ndimage import binary_closing
from scipy.ndimage import binary_erosion, median_filter
from scipy.spatial.distance import euclidean

from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion

def compute_direction(a, b):
    vec = b - a
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else None
    
    
def get_MOIs(points_ref):

    
    MOIs = {}
    
    CL = euclidean(points_ref['C_R'], points_ref['C_L']) * 0.0112
    ant_ext = euclidean(points_ref['OB_UL'], points_ref['OB_UR']) * 0.0112
    post_ext = euclidean(points_ref['OB_DL'], points_ref['OB_DR']) * 0.0112
    ant_int = euclidean(points_ref['IB_UL'], points_ref['IB_UR']) * 0.0112
    post_int = euclidean(points_ref['IB_DL'], points_ref['IB_DR']) * 0.0112
    
    MOIs["CL"] = CL
    MOIs["CL/AE"] = CL/ant_ext
    MOIs["CL/PE"] = CL/post_ext
    MOIs["CL/AI"] = CL/ant_int
    MOIs["CL/PI"] = CL/post_int
    
    dir1 = compute_direction(np.array(points_ref['IB_UL']),np.array(points_ref['IB_UR']))
    dir2 = compute_direction(np.array(points_ref['IB_DL']), np.array(points_ref['IB_DR']))
    cos_angle = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    
    MOIs['Angle_Int'] = angle
    
    dir1 = compute_direction(np.array(points_ref['IB_UL']),np.array(points_ref['C_L']))
    dir2 = compute_direction(np.array(points_ref['C_L']), np.array(points_ref['C_R']))
    cos_angle = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    
    MOIs['Angle_Funnel'] = angle
    
    return MOIs
    
def get_ROIs_cervix(final_cervix, dist_transf, points_ref, D=40):
    
    ROIs = {}
    
    #rectangle = max_rectangle_in_mask((final_cervix == 6))
    zone_banos = median_filter(binary_erosion((final_cervix == 5), iterations=8), 3)
    ROIs["Ant_Banos"] = zone_banos.astype(np.uint8)
    
    proxy = final_cervix.copy()
    proxy[dist_transf[str(points_ref['C_R'])] > D] = 0
    rectangle = max_square_in_mask((proxy == 5))
    ant_ext = rectangle_to_mask(final_cervix.shape, rectangle)
    rectangle = max_square_in_mask((proxy == 6))
    post_ext = rectangle_to_mask(final_cervix.shape, rectangle)
    ROIs["Ant_Ext"] = ant_ext.astype(np.uint8)
    ROIs["Post_Ext"] = post_ext.astype(np.uint8)
    ROIs["Cervix"] = (final_cervix != 0).astype(np.uint8)
    
    return ROIs
 
import scipy.ndimage as snd

def edit_transf(mask, point, D, inf=True):

    mask_test = mask.copy()
    mask_test[point[1], point[0]] = 10
    masque = (mask_test != 10).astype(np.uint8)
    distance = distance_transform_edt(masque)

    if inf:
        mask[distance < D] = 0
    else:
        mask[distance > D] = 0
    
    return mask

def getLargestConnectedComponent_2D(img):
    c,n = snd.label(img)
    sizes = snd.sum(img, c, range(n+1))
    mask_size = sizes < (max(sizes))
    remove_voxels = mask_size[c]
    c[remove_voxels] = 0
    c[np.where(c!=0)]=1
    img[np.where(c==0)] = 0
    return img
    
def get_ROIs_cervix(final_cervix, dist_transf, points_ref, D=40):
    
    ROIs = {}
    
    #rectangle = max_rectangle_in_mask((final_cervix == 6))
    zone_banos = median_filter(binary_erosion((final_cervix == 5), iterations=8), 3)
    ROIs["Ant_Banos"] = zone_banos.astype('uint8')
    
    proxy = final_cervix.copy()
    proxy[dist_transf[str(points_ref['C_R'])] > D] = 0
    rectangle = max_square_in_mask((proxy == 5))
    ant_ext = rectangle_to_mask(final_cervix.shape, rectangle)
    rectangle = max_square_in_mask((proxy == 6))
    post_ext = rectangle_to_mask(final_cervix.shape, rectangle)

    proxy = final_cervix.copy()
    #proxy = binary_erosion(proxy, iterations=3)
    
    proxy[proxy == 0] = 8

    D = max(euclidean(points_ref['IB_UL'], points_ref['IB_UR']), euclidean(points_ref['IB_DL'], points_ref['IB_DR'])) 
    D /= 2
    point = (np.array(points_ref['IB_UR']) + np.array(points_ref['IB_DR'])) // 2
    mask_test = final_cervix.copy()
    mask_test[point[1], point[0]] = 10
    masque = (mask_test != 10).astype(np.uint8)
    distance = distance_transform_edt(masque)
   
    proxy[distance > D] = 0
    
    region_out = (proxy == 8)

    #1. Dilatation Postérieur

    proxy = final_cervix.copy()
    proxy[proxy != 6] = 0
    proxy[proxy == 6] = 1
    
    proxy_dil = binary_dilation(proxy, iterations=40).astype('float')
    proxy_dil[final_cervix != 0] = 0
    proxy_comp = proxy_dil - proxy.astype('float')
    
    #2. distance transform C_L
    point = points_ref['IB_DR']

    if euclidean(points_ref['IB_DL'], points_ref['OB_DL']) > euclidean(points_ref['OB_DR'], points_ref['OB_DL']):
        point_2 = (np.array(points_ref['IB_DL']) + np.array(points_ref['OB_DL'])) // 2
        D_2 = euclidean(points_ref['IB_DL'], points_ref['OB_DL']) / 2.2
    else:
        point_2 = (np.array(points_ref['OB_DR']) + np.array(points_ref['OB_DL'])) // 2
        D_2 = euclidean(points_ref['OB_DR'], points_ref['OB_DL']) / 2.2
    D = euclidean(point, point_2)
    proxy_comp = edit_transf(proxy_comp, point, D, inf=True)
    
    proxy_comp = edit_transf(proxy_comp, point_2, D_2, inf=False)
    # proxy_comp = edit_transf(proxy_comp, points_ref['OB_DR'], 30, inf=True)
    # proxy_comp = edit_transf(proxy_comp, points_ref['IB_DL'], 30, inf=True)
    # #3. D > C_L to milieu(OB_DR + OB_DL)
   
    
    ROIs['Under_Post'] = getLargestConnectedComponent_2D(proxy_comp).astype(np.uint8)
    ROIs['Region_Out'] = getLargestConnectedComponent_2D(region_out).astype(np.uint8)
    ROIs["Ant_Ext"] = ant_ext.astype(np.uint8)
    ROIs["Post_Ext"] = post_ext.astype(np.uint8)
    ROIs["Cervix"] = (final_cervix != 0).astype(np.uint8)
    
    return ROIs
    
       
def resize_to_img(img, mask, points_ref, dist_transf):

    shape_target = img.shape[1], img.shape[0]
    shape_basis = mask.shape[1], mask.shape[0]

    dist_transf_new = {}
    for key in points_ref.keys():

        point = points_ref[key]
        new_point = []
        for i in range(2):
            new_point.append(int(point[i] * shape_target[i] / shape_basis[i]))
        
        points_ref[key] = np.array(new_point)

        if str(point) in dist_transf.keys():
            dist_transf_new[str(np.array(new_point))] = dist_transf[str(point)]

    for key in dist_transf_new.keys():

        dist_transf_new[key] = cv2.resize(dist_transf_new[key], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)


    return mask, points_ref, dist_transf_new

      
       
def get_class(mask, classe):

    output = mask.copy()

    if classe == 'VD':
        output[output!=1] = 0
        return output

    elif classe == 'COND':
        output[output!=2] = 0
        output[output!=0] = 1
        return output

    elif classe == 'Lung':
        output[output!=3] = 0
        output[output!=0] = 1
        return output

    elif classe == 'Lesion':
        output[output==3] = 0
        output[output!=0] = 1
        return output

def get_complementary(mask, classe):

    output = mask.copy()

    if classe == 'VD':
        output[output==1] = 0
        output[output!=0] = 1
        return output

    elif classe == 'COND':
        output[output==2] = 0
        output[output!=0] = 1
        return output

    elif classe == 'Lung':
        output[output==3] = 0
        output[output!=0] = 1
        return output

    elif classe == 'Lesion':
        output[output==1] = 0
        output[output==2] = 0
        output[output!=0] = 1
        return output


def largest_rectangle_in_mask(mask):
    """
    Trouve les coordonnées du plus grand rectangle contenant uniquement des True dans un masque 2D.
    
    :param mask: np.ndarray booléen (True pour la région valide, False sinon)
    :return: (x1, y1, x2, y2) -> Coordonnées du coin supérieur gauche et inférieur droit du rectangle optimal
    """
    if not mask.any():
        return None  # Si le masque est vide, aucun rectangle possible
    
    rows, cols = mask.shape
    heights = np.zeros((rows, cols), dtype=int)

    # Construire la matrice des hauteurs cumulées
    for j in range(cols):
        for i in range(rows):
            if mask[i, j]:
                heights[i, j] = heights[i-1, j] + 1 if i > 0 else 1

    # Fonction pour trouver le plus grand rectangle dans un histogramme
    def max_histogram_area(heights_row):
        """
        Trouve la plus grande aire rectangulaire dans un histogramme.
        Retourne l'aire maximale et les indices (début, fin) de la base du rectangle.
        """
        stack = []
        max_area = 0
        best_x1, best_x2 = 0, 0
        h = np.append(heights_row, 0)  # Ajout d'une barrière pour vider la pile

        for i in range(len(h)):
            while stack and h[i] < h[stack[-1]]:
                h_idx = stack.pop()
                height = h[h_idx]
                width = i if not stack else i - stack[-1] - 1
                area = height * width
                if area > max_area:
                    max_area = area
                    best_x1 = stack[-1] + 1 if stack else 0
                    best_x2 = i - 1
            stack.append(i)

        return max_area, best_x1, best_x2

    # Parcourir chaque ligne et trouver le plus grand rectangle
    max_rectangle = (0, 0, 0, 0)
    max_area = 0

    for i in range(rows):
        area, x1, x2 = max_histogram_area(heights[i])
        if area > max_area:
            max_area = area
            max_rectangle = (x1, i - heights[i, x1] + 1, x2, i)

    return max_rectangle  # (x1, y1, x2, y2)

import numpy as np

def rectangle_to_mask(shape, rectangle):
    """
    Crée un masque binaire avec un rectangle de 1 dans une image de taille donnée.
    
    :param shape: (rows, cols) -> Dimensions du masque de sortie
    :param rectangle: (x1, y1, x2, y2) -> Coordonnées du plus grand rectangle
    :return: np.ndarray (masque binaire avec 1 dans le rectangle)
    """
    mask = np.zeros(shape, dtype=np.uint8)  # Initialise le masque vide (tout à 0)

    x1, y1, x2, y2 = rectangle  # Extraction des coordonnées du rectangle

    # Dessiner le rectangle (remplissage de 1)
    mask[y1:y2+1, x1:x2+1] = 1  

    return mask

def isolate_rectangle(image, rectangle):

    x1, y1, x2, y2 = rectangle  # Extraction des coordonnées du rectangle

    # Dessiner le rectangle (remplissage de 1)
    return image[y1:y2+1, x1:x2+1] 



def largest_component(mask):
    """
    Trouve la plus grande composante connexe dans un masque binaire.
    
    :param mask: np.ndarray booléen (1 = objet, 0 = fond)
    :return: np.ndarray booléen du même format, ne contenant que la plus grande composante
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    
    if num_labels <= 1:
        return mask  # Pas de composante détectée

    # Trouver l'indice de la plus grande composante (en ignorant le fond)
    largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Créer un nouveau masque contenant seulement la plus grande composante
    return (labels == largest_idx)

def max_rectangle_in_mask(mask):
    """
    Trouve le plus grand rectangle possible à l'intérieur d'un masque binaire.
    
    :param mask: np.ndarray booléen (1 = intérieur, 0 = extérieur)
    :return: (x1, y1, x2, y2) -> Coordonnées du plus grand rectangle
    """
    h, w = mask.shape
    heights = np.zeros((h, w), dtype=int)

    # Calcul des hauteurs continues de pixels 1
    for x in range(w):
        for y in range(h):
            if mask[y, x]:
                heights[y, x] = heights[y - 1, x] + 1 if y > 0 else 1

    max_area = 0
    best_rect = (0, 0, 0, 0)

    # Trouver le plus grand rectangle
    for y in range(h):
        stack = []
        for x in range(w + 1):
            while stack and (x == w or heights[y, x] < heights[y, stack[-1]]):
                h_rect = heights[y, stack.pop()]
                w_rect = x if not stack else x - stack[-1] - 1
                area = h_rect * w_rect
                if area > max_area:
                    max_area = area
                    best_rect = (stack[-1] + 1 if stack else 0, y - h_rect + 1, x - 1, y)

            stack.append(x)

    return best_rect

def max_square_in_mask(mask):
    """
    Trouve le plus grand carré possible à l'intérieur d'un masque binaire.

    :param mask: np.ndarray booléen (1 = intérieur, 0 = extérieur)
    :return: (x1, y1, x2, y2) -> Coordonnées du plus grand carré
    """
    h, w = mask.shape
    S = np.zeros((h, w), dtype=int)

    max_size = 0
    bottom_right = (0, 0)

    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                if y == 0 or x == 0:
                    S[y, x] = 1
                else:
                    S[y, x] = min(S[y-1, x], S[y, x-1], S[y-1, x-1]) + 1

                if S[y, x] > max_size:
                    max_size = S[y, x]
                    bottom_right = (x, y)

    if max_size == 0:
        return None  # Aucun carré trouvé

    x2, y2 = bottom_right
    x1 = x2 - max_size + 1
    y1 = y2 - max_size + 1

    return (x1, y1, x2, y2)
    
def expand_rectangle(rect, lung_mask, max_size):
    """
    Étend un rectangle dans la direction contenant le plus de tissu pulmonaire,
    en utilisant une zone d'exploration égale à la taille initiale du rectangle.

    :param rect: (x1, y1, x2, y2) -> Rectangle de base
    :param lung_mask: np.ndarray booléen du tissu pulmonaire
    :param max_size: (w_max, h_max) -> Dimensions maximales autorisées
    :return: (x1, y1, x2, y2) -> Rectangle étendu
    """
    x1, y1, x2, y2 = rect
    h, w = lung_mask.shape
    rect_width = x2 - x1 + 1
    rect_height = y2 - y1 + 1

    # Vérifier l’espace disponible dans chaque direction
    left_space = min(x1, rect_width)  
    right_space = min(w - x2 - 1, rect_width)  
    top_space = min(y1, rect_height)  
    bottom_space = min(h - y2 - 1, rect_height)  

    # Quantité de tissu pulmonaire explorée sur une aire égale à celle du rectangle de base
    lung_left = np.sum(lung_mask[y1:y2+1, max(0, x1-left_space):x1])
    lung_right = np.sum(lung_mask[y1:y2+1, x2+1:min(w, x2+1+right_space)])
    lung_top = np.sum(lung_mask[max(0, y1-top_space):y1, x1:x2+1])
    lung_bottom = np.sum(lung_mask[y2+1:min(h, y2+1+bottom_space), x1:x2+1])

    # Trouver la direction avec le plus de poumon
    lung_areas = {"left": lung_left, "right": lung_right, "top": lung_top, "bottom": lung_bottom}
    best_direction = max(lung_areas, key=lung_areas.get)

    # Étendre le rectangle dans cette direction
    if best_direction == "left":
        x1 = max(0, x1 - left_space)
    elif best_direction == "right":
        x2 = min(w - 1, x2 + right_space)
    elif best_direction == "top":
        y1 = max(0, y1 - top_space)
    elif best_direction == "bottom":
        y2 = min(h - 1, y2 + bottom_space)

    return x1, y1, x2, y2

def get_balanced_rectangle(lesion_mask, lung_mask):
    """
    Trouve un rectangle maximal à l'intérieur de la plus grande lésion et l'étend vers le tissu sain.
    
    :param lesion_mask: np.ndarray booléen (1 = lésion, 0 = ailleurs)
    :param lung_mask: np.ndarray booléen (1 = tissu sain, 0 = ailleurs)
    :return: (x1, y1, x2, y2) -> Coordonnées du rectangle final
    """
    # Étape 1 : Extraire la plus grande lésion
    largest_lesion = largest_component(lesion_mask)

    # Étape 2 : Trouver le plus grand rectangle interne
    rect = max_rectangle_in_mask(largest_lesion)

    # Étape 3 : Étendre le rectangle vers le tissu sain
    final_rect = expand_rectangle(rect, lung_mask, max_size=lesion_mask.shape)

    return final_rect
    
def normalize_points_ref(points_ref):

    points_ref_xy = {}
    
    shape_basis = [288,224]
    for key in points_ref.keys():

        point = points_ref[key]
        new_point = []
        for i in range(2):
            new_point.append(float(point[i] / shape_basis[i]))
        
        points_ref_xy[key+'_x'] = round(new_point[0], 4)
        points_ref_xy[key+'_y'] = round(new_point[1], 4)

    return points_ref_xy
