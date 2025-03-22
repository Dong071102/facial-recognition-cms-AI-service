import numpy as np  
import torch 

def string_to_np(embedding_string):
    face_embedding = embedding_string.strip("[]")  # Loại bỏ dấu ngoặc
    face_embedding_list = face_embedding.split(",")  # Tách chuỗi thành danh sách
    embedding_array = np.array([float(x) for x in face_embedding_list])  # Chuyển thành numpy array
    return embedding_array
    
def np_to_tensof(np_array):
    return torch.ten(np_array)

    