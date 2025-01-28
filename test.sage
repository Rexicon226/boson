from sage.rings.polynomial.polynomial_gf2x import GF2X_BuildIrred_list

# Fp = GF(0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001)
Fp = GF(641)
Rp.<x> = PolynomialRing(Fp)

# points = [[1, -5], [2, 0], [3, 1], [4, 0]]

# Poly = Rp.lagrange_polynomial(points)
# print(f"final: {Poly}")

print((4*x**2 + 2*x + 1) + (3*x + 2))


# L = Matrix(Fp, [
#     [0,1,0,0,0,0],
#     [0,0,0,1,0,0],
#     [0,1,0,0,1,0],
#     [5,0,0,0,0,1]]
# )

# R = Matrix(Fp, [
#     [0, 1, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0]]
# )

# O = Matrix(Fp, [
#     [0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0, 0]]
# )

# M = [L, R, O]
# PolyM = []
# for m in M:
#     PolyList = []
#     for i in range(m.ncols()):
#         points = []
#         for j in range(m.nrows()):
#             points.append([j+1,m[j,i]])
#         Poly = Rp.lagrange_polynomial(points).coefficients(sparse=False)
#         if(len(Poly) < m.nrows()):
#             # if degree of the polynomial is less than 4
#             # we add zeroes to represent the missed out terms
#             dif = m.nrows() - len(Poly)
#             for c in range(dif):
#                 Poly.append(0);
#         PolyList.append(Poly)
#     print(Matrix(Fp, PolyList), end='\n\n')
#     PolyM.append(Matrix(Fp, PolyList))

# # We define the solution vector also in the field
# S = vector(Fp,[1, 3, 35, 9, 27, 30])

# # Create the Lx, Rx & Ox polynomial
# Lx = Rp(list(S*PolyM[0]))
# Rx = Rp(list(S*PolyM[1]))
# Ox = Rp(list(S*PolyM[2]))
# print("Lx = " + str(Lx))
# print("Rx = " + str(Rx))
# print("Ox = " + str(Ox))

# T = Lx * Rx - Ox
# print("T(X) = ", end="")
# print(T)

# Z = Rp((x-1)*(x-2)*(x-3)*(x-4))
# H = T.quo_rem(Z)
# print("Quotient of Z/T = ", end="")
# print(H[0])
# print("Remainder of Z/T = ", end="")
# print(H[1])