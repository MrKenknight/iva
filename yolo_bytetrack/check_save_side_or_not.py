def vector_cross_product(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

def check_same_side(A, B, C, D):
    AB = (B[0] - A[0], B[1] - A[1])
    AC = (C[0] - A[0], C[1] - A[1])
    AD = (D[0] - A[0], D[1] - A[1])
    
    cross1 = vector_cross_product(AB, AC)
    cross2 = vector_cross_product(AB, AD)
    
    if cross1 * cross2 > 0:
        return True  # Cùng phía
    else:
        return False  # Khác phía

A = (1, 1)
B = (4, 4)
C = (2, 3)
D = (1, 1)

if check_same_side(A, B, C, D):
    print("Điểm D nằm cùng phía với điểm C so với đường thẳng AB.")
else:
    print("Điểm D nằm khác phía với điểm C so với đường thẳng AB.")
