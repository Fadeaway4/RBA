import os
from evaluations.compute_idx_emb import compute_idx_embedding
from evaluations.ism_fdfr import matching_score_id

def matching_score_genimage_id(images_path, list_id_path, image_name_list):
    image_list = image_name_list
    fail_detection_count = 0
    ave_ism = 0
    avg_embedding = compute_idx_embedding(list_id_path)

    for image_name in image_list:
        image_path = os.path.join(images_path, image_name)
        ism = matching_score_id(image_path, avg_embedding)
        if ism is None:
            fail_detection_count += 1
        else:
            ave_ism += ism
    if fail_detection_count != len(image_list):
        return ave_ism/(len(image_list)-fail_detection_count), fail_detection_count/len(image_list)
    return None, 1

"""
from .evaluations.compute_idx_emb import compute_idx_embedding
from .evaluations.ism_fdfr import matching_score_id
def matching_score_genimage_id(images_path, list_id_path, image_name_list):
    image_list = image_name_list
    fail_detection_count = 0
    ave_ism = 0
    avg_embedding = compute_idx_embedding(list_id_path)

    for image_name in image_list:
        image_path = os.path.join(images_path, image_name)
        ism = matching_score_id(image_path, avg_embedding)
        if ism is None:
            fail_detection_count += 1
        else:
            ave_ism += ism
    if fail_detection_count != len(image_list):
        return ave_ism/(len(image_list)-fail_detection_count), fail_detection_count/len(image_list)
    return None, 1
    """