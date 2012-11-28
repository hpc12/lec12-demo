import numpy as np
import meshpy.triangle as triangle
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as sla
import matplotlib.pyplot as pt

# {{{ build US mesh

outline = np.loadtxt("us-outline.dat")

def round_trip_connect(start, end):
    result = []
    for i in range(start, end):
        result.append((i, i+1))
    result.append((end, start))
    return result

def needs_refinement(vertices, area):
    vert_origin, vert_destination, vert_apex = vertices
    bary_x = (vert_origin.x + vert_destination.x + vert_apex.x) / 3
    bary_y = (vert_origin.y + vert_destination.y + vert_apex.y) / 3

    dist_center = np.sqrt((bary_x-600)**2 + (750-bary_y)**2 )
    max_area = 1 + 0.8*dist_center
    return bool(area > max_area)


info = triangle.MeshInfo()
info.set_points(outline)
info.set_facets(round_trip_connect(0, len(outline)-1))

mesh = triangle.build(info, refinement_func=needs_refinement)

points = np.array(mesh.points)
elements = np.array(mesh.elements)

if 1:
    pt.triplot(points[:, 0], points[:, 1], elements, color="black")
    pt.show()

    import sys
    sys.exit()

# }}}

# {{{ find connectivity

neighbors = {}
for a, b, c in mesh.elements:
    for v1, v2 in [(a,b), (b,c), (c,a)]:
        for x, y in [(v1, v2), (v2, v1)]:
            neighbors.setdefault(v1, set()).add(v2)

# }}}

# {{{ make graph laplacian

row  = []
col  = []
data = []
for vnr, nb_nrs in neighbors.iteritems():
    row.append(vnr)
    col.append(vnr)
    data.append(len(nb_nrs))

    for nb in nb_nrs:
        row.append(vnr)
        col.append(nb)
        data.append(-1)

lap = coo_matrix((data, (row,col)), dtype=np.float64).tocsr()

# }}}

eigval, eigvec = sla.eigs(lap, 6, which="SM")
print eigval

for vec in eigvec.T[1:]:
    vec = vec.real
    pt.triplot(points[:, 0], points[:, 1], elements, color="black", lw=0.1)
    pt.tripcolor(points[:, 0], points[:, 1], elements, vec)
    pt.colorbar()
    pt.tricontour(points[:, 0], points[:, 1], elements, vec, colors="black", levels=[0])
    pt.show()

# vim: foldmethod=marker
