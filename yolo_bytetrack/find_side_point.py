import numpy as np

def find_side_point(A, B, distance):
    AB_vector = np.array(B) - np.array(A)
    mid_point = (np.array(A) + np.array(B)) / 2
    perpendicular_vector = np.array([-AB_vector[1], AB_vector[0]])
    normalized_perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
    print(normalized_perpendicular_vector)
    C = mid_point + distance * normalized_perpendicular_vector
    D = mid_point - distance * normalized_perpendicular_vector
    return C.astype(int), D.astype(int)

A = (1, 1)
B = (1, 5)
distance = 2
C, D = find_side_point(A, B, distance)
print("C:", C)
print("D:", D)

