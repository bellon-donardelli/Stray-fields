import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import verde as vd
import choclo
import harmonica as hm
import math
from numba import njit, jit, prange
import pandas as pd
from scipy.interpolate import griddata

constant = choclo.constants.VACUUM_MAGNETIC_PERMEABILITY / (4 * np.pi)



def data_importer_paraview(
    file: str, 
    x_grid: int = 100, 
    y_grid: int = 100, 
    var_names: list[str] = ["Points:0", "Points:1", "Points:2", "Bz"]
):
    """
    Import and interpolate ParaView-exported CSV data onto a structured 2D grid.

    This function reads a CSV file exported from ParaView containing 3D point 
    coordinates and a scalar/vector field (e.g., magnetic field component `Bz`). 
    The data is interpolated onto a regular 2D mesh grid in the XY-plane using 
    cubic interpolation.

    Parameters
    ----------
    file : str or path-like
        Path to the CSV file containing ParaView-exported data.
    x_grid : int, optional, default=100
        Number of grid points along the X-axis for interpolation.
    y_grid : int, optional, default=100
        Number of grid points along the Y-axis for interpolation.
    var_names : list of str, optional
        List of column names in the CSV file. The order must be:
        
        - `var_names[0]` : X-coordinate column (default `"Points:0"`)
        - `var_names[1]` : Y-coordinate column (default `"Points:1"`)
        - `var_names[2]` : Z-coordinate column (default `"Points:2"`)
        - `var_names[3]` : Field component column to interpolate (default `"Bz"`)

    Returns
    -------
    xi : ndarray of shape (y_grid, x_grid)
        Meshgrid of X-coordinates for interpolation domain.
    yi : ndarray of shape (y_grid, x_grid)
        Meshgrid of Y-coordinates for interpolation domain.
    zi : ndarray of shape (y_grid, x_grid)
        Interpolated Z-values on the (xi, yi) grid.
    Bz : ndarray of shape (y_grid, x_grid)
        Interpolated field component values (from `var_names[3]`) on the grid.
    X : ndarray of shape (n,)
        Original scattered X-coordinate values from the CSV.
    Y : ndarray of shape (n,)
        Original scattered Y-coordinate values from the CSV.
    Z : ndarray of shape (n,)
        Original scattered Z-coordinate values from the CSV.
    bz_comp : ndarray of shape (n,)
        Original scattered field component values from the CSV.


    """

    # Load CSV data
    data = pd.read_csv(file, delimiter=',')

    # Extract coordinates and field
    X = data.get(var_names[0])
    Y = data.get(var_names[1])
    Z = data.get(var_names[2])
    bz_comp = np.asarray(data.get(var_names[3]))

    # Generate interpolation grid
    xi = np.linspace(min(X), max(X), x_grid)
    yi = np.linspace(min(Y), max(Y), y_grid)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate Z and Bz onto grid
    zi = griddata((X, Y), Z, (xi, yi), method='cubic')
    Bz = griddata((X, Y), bz_comp, (xi, yi), method='cubic')

    return xi, yi, zi, Bz, X, Y, Z, bz_comp


def read_tecplot_file(file_path):
    """
    Read and return the contents of a Tecplot file.

    This function opens a Tecplot file in text mode and reads the entire
    content into a string. It is useful for inspecting or parsing raw 
    Tecplot-formatted data.

    Parameters
    ----------
    file_path : str or path-like
        Path to the Tecplot file to be read.

    Returns
    -------
    str
        The full contents of the Tecplot file as a single string.

    """
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content



def read_tecplot_header(file_content):
    """
    Parse the header of a Tecplot file and extract metadata.

    This function processes the text content of a Tecplot file and extracts:
    - The dataset title
    - The list of variable names
    - The number of nodes (N)
    - The number of elements (E)

    Parameters
    ----------
    file_content : str
        Full text content of a Tecplot file, typically obtained by
        `read_tecplot_file`.

    Returns
    -------
    title : str
        The title of the Tecplot dataset.
    variables : list of str
        List of variable names defined in the Tecplot file.
    N : int
        Number of nodes defined in the ZONE specification.
    E : int
        Number of elements defined in the ZONE specification.
    """
    lines = file_content.strip().split('\n')
    title = lines[0].split('=')[1].strip().strip('"')
    variables = [v.strip().strip('"') for v in lines[1].split('=')[1].strip().split(',')]
    
    # Find the line containing ZONE information
    zone_line = [line for line in lines if 'ZONE' in line][0]
    N = int(zone_line.split('N=')[1].split(',')[0].strip())
    E = int(zone_line.split('E=')[1].strip())
    
    return title, variables, N, E



def organize_data(file_content, N, E):
    """
    Organize numerical data from a Tecplot file into structured arrays.

    Parameters
    ----------
    file_content : str
        Full text content of a Tecplot file.
    N : int
        Number of nodes specified in the Tecplot header.
    E : int
        Number of elements specified in the Tecplot header.

    Returns
    -------
    variables_values : dict
        Dictionary containing variables "X", "Y", "Z", "Mx", "My", "Mz" 
        with lists of float values.
    sd_values : list of float
        Scalar data values (length E).
    vertices_indexes_array : list of list of int
        Connectivity information, grouped in sets of 4 node indices per element 
        (zero-based indexing).
    """
    lines = file_content.strip().split('\n')
    
    # Initialize a large array to store all values
    all_values = []
    
    # Search from line 4 to 6*N + E
    start_index = 4
    end_index = start_index + 6 * N + E
    
    # Ensure end_index does not exceed the total number of lines
    if end_index > len(lines):
        end_index = len(lines)
    
    for i in range(start_index, end_index):
        values = lines[i].strip().split()
        all_values.extend(values)
    
    # Split values into variable arrays
    variables_values = {}
    variables = ["X", "Y", "Z", "Mx", "My", "Mz"]
    for i, var in enumerate(variables):
        variables_values[var] = [float(value) for value in all_values[i*N:(i+1)*N]]
    
    sd_values = [float(value) for value in all_values[6*N:6*N + E]]
    vertices_indexes = [int(value)-1 for value in all_values[6*N + E:]]  # -1 to make the first index be 0
    
    # Split vertices_indexes into groups of 4 values each
    vertices_indexes_array = [vertices_indexes[i:i+4] for i in range(0, len(vertices_indexes), 4)]
    
    return variables_values, sd_values, vertices_indexes_array


def calculate_normal(face):
    """
    Calculate the normalized normal vector of a face of a tetrahedral element.

    The face is defined by three vertices (points in 3D space), and the normal 
    is computed using the cross product of two edges of the face.

    Parameters
    ----------
    face : tuple or list of array-like
        A face of a tetrahedral element defined by three 3D points (v1, v2, v3),
        where each point is a sequence of three coordinates (x, y, z).

    Returns
    -------
    numpy.ndarray
        Normalized 3D normal vector of the face as a NumPy array [nx, ny, nz].
    """
    v1, v2, v3 = face
    
    # Vector subtraction
    vector1_x = v2[0] - v1[0]
    vector1_y = v2[1] - v1[1]
    vector1_z = v2[2] - v1[2]
    
    vector2_x = v3[0] - v1[0]
    vector2_y = v3[1] - v1[1]
    vector2_z = v3[2] - v1[2]
    
    # Cross product
    normal_x = vector1_y * vector2_z - vector1_z * vector2_y
    normal_y = vector1_z * vector2_x - vector1_x * vector2_z
    normal_z = vector1_x * vector2_y - vector1_y * vector2_x
    
    # Normalization
    norm = math.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm
    
    return np.asarray([normal_x, normal_y, normal_z])


def ensure_normal_outward(face, normal, point_inside, epsilon=1e-15):
    """
    Ensure that a face normal vector points outward from a tetrahedron.

    This function checks whether the given face normal points inward 
    (toward a reference point inside the tetrahedron). If so, the normal 
    is inverted to point outward.

    Parameters
    ----------
    face : list or tuple
        The triangular face of the tetrahedron, defined by three 3D vertices.
    normal : list or numpy.ndarray
        The normal vector of the face (not yet guaranteed to be outward).
    point_inside : list or numpy.ndarray
        A point known to be inside the tetrahedron.
    epsilon : float, optional, default=1e-15
        Numerical tolerance for dot product comparison.

    Returns
    -------
    list
        The corrected normal vector that points outward.
    """
    v1 = face[0]
    
    # Manual subtraction (vector from v1 to point_inside)
    sub_x = point_inside[0] - v1[0]
    sub_y = point_inside[1] - v1[1]
    sub_z = point_inside[2] - v1[2]
    
    # Manual dot product
    dot_product = normal[0] * sub_x + normal[1] * sub_y + normal[2] * sub_z
    
    # If dot product is positive, normal points inward → invert it
    if dot_product > epsilon:
        normal = [-normal[0], -normal[1], -normal[2]]
    
    return normal



def sort_tetrahedrons(tetrahedrons, tetrahedrons_centroid):
    """
    Sort tetrahedron faces to ensure outward-pointing normals.

    For each face of each tetrahedron, the normal vector is computed and
    adjusted (if necessary) so that it points outward relative to the 
    tetrahedron's centroid. If the normal was inverted, the vertex order 
    of the face is reversed to maintain consistent orientation.

    Parameters
    ----------
    tetrahedrons : numpy.ndarray
        Array of tetrahedrons, where each tetrahedron is defined by its faces,
        and each face is defined by its three vertices.
    tetrahedrons_centroid : numpy.ndarray
        Array of centroid coordinates, one for each tetrahedron.

    Returns
    -------
    numpy.ndarray
        A copy of the tetrahedrons array with consistently oriented faces, 
        ensuring outward-pointing normals.
    """
    tetrahedrons_sorted = np.copy(tetrahedrons)
    epsilon = 1e-15
    for i, (faces, centroid_tetrahedron) in enumerate(zip(tetrahedrons, tetrahedrons_centroid)):
       
        for j, face in enumerate(faces):
            n_f = calculate_normal(face)
            n_f_copy = n_f
            
            n_f = ensure_normal_outward(face, n_f, centroid_tetrahedron)
            
            # Check if the normal was inverted without using are_vectors_close
            if not (math.isclose(n_f_copy[0], n_f[0], abs_tol=epsilon) and
                    math.isclose(n_f_copy[1], n_f[1], abs_tol=epsilon) and
                    math.isclose(n_f_copy[2], n_f[2], abs_tol=epsilon)):
                r1, r2, r3 = face[::-1]
            else:
                r1, r2, r3 = face
            
            tetrahedrons_sorted[i, j, :, :] = r1, r2, r3
    return tetrahedrons_sorted


def define_tetrahedrons(X, Y, Z, vertices_indexes):
    """
    Build tetrahedral elements and their centroids from node coordinates.

    Each tetrahedron is defined by four vertex indices. The function converts
    the vertex coordinates from micrometers to meters, computes the centroid,
    and defines the four triangular faces of the tetrahedron.

    Parameters
    ----------
    X, Y, Z : list or numpy.ndarray
        Arrays containing the x, y, and z coordinates of all nodes.
    vertices_indexes : list of list of int
        Connectivity information. Each sublist contains four node indices 
        (zero-based) that define a tetrahedron.

    Returns
    -------
    tetrahedrons : list of list of list of float
        A list of tetrahedrons, where each tetrahedron is represented by its
        four triangular faces, and each face is a list of three 3D vertices.
    tetrahedrons_centroids : list of numpy.ndarray
        A list of centroid coordinates for each tetrahedron.
    """
    tetrahedrons = []
    tetrahedrons_centroids = []
    
    for vertices_array in vertices_indexes:
        vertices = np.array([
            [X[vertices_array[0]], Y[vertices_array[0]], Z[vertices_array[0]]],
            [X[vertices_array[1]], Y[vertices_array[1]], Z[vertices_array[1]]],
            [X[vertices_array[2]], Y[vertices_array[2]], Z[vertices_array[2]]],
            [X[vertices_array[3]], Y[vertices_array[3]], Z[vertices_array[3]]],
        ]) * 1.0e-6  # convert micrometers to meters

        centroid_tetrahedron = np.mean(vertices, axis=0)
        
        # Define the faces of the tetrahedron (4 triangular faces)
        faces = [
            [vertices[0], vertices[1], vertices[2]],
            [vertices[0], vertices[1], vertices[3]],
            [vertices[0], vertices[2], vertices[3]],
            [vertices[1], vertices[2], vertices[3]]
        ]
        
        tetrahedrons.append(faces)
        tetrahedrons_centroids.append(centroid_tetrahedron)
    
    return tetrahedrons, tetrahedrons_centroids


def define_tetrahedrons_magnetizations(Mx, My, Mz, vertices_indexes, Ms=480000):
    """
    Compute the average magnetization vector for each tetrahedral element.

    For each tetrahedron, the magnetization components are averaged across 
    its four vertices and then scaled by the saturation magnetization `Ms`.

    Parameters
    ----------
    Mx, My, Mz : list or numpy.ndarray
        Magnetization components at the mesh nodes.
    vertices_indexes : list of list of int
        Connectivity information. Each sublist contains four node indices 
        (zero-based) that define a tetrahedron.
    Ms : float, optional, default=480000
        Saturation magnetization used to scale the averaged components.

    Returns
    -------
    numpy.ndarray
        Array of shape (n, 3), where `n` is the number of tetrahedrons.
        Each row corresponds to the [Mx, My, Mz] magnetization vector 
        of a tetrahedron after averaging and scaling by Ms.
    """
    # Initialize array with the expected number of tetrahedrons
    n = len(vertices_indexes)
    tetrahedrons_magnetization = np.empty((n, 3))

    for i in range(n):
        # Access vertex indices directly
        mag_x = (Mx[vertices_indexes[i][0]] + Mx[vertices_indexes[i][1]] + 
                 Mx[vertices_indexes[i][2]] + Mx[vertices_indexes[i][3]]) / 4
        mag_y = (My[vertices_indexes[i][0]] + My[vertices_indexes[i][1]] + 
                 My[vertices_indexes[i][2]] + My[vertices_indexes[i][3]]) / 4
        mag_z = (Mz[vertices_indexes[i][0]] + Mz[vertices_indexes[i][1]] + 
                 Mz[vertices_indexes[i][2]] + Mz[vertices_indexes[i][3]]) / 4

        # Multiply by saturation magnetization Ms
        tetrahedrons_magnetization[i, 0] = mag_x * Ms
        tetrahedrons_magnetization[i, 1] = mag_y * Ms
        tetrahedrons_magnetization[i, 2] = mag_z * Ms

    return tetrahedrons_magnetization


def define_tetrahedrons_magnetizations_weighted(X, Y, Z, Mx, My, Mz, vertices_indexes, Ms=480000):
    """
    Compute the weighted average magnetization vector for each tetrahedral element.

    The magnetization is computed by weighting each vertex contribution inversely
    proportional to its distance from the tetrahedron centroid. This ensures that
    vertices closer to the centroid contribute more strongly to the averaged
    magnetization.

    Parameters
    ----------
    X, Y, Z : list or numpy.ndarray
        Node coordinates in micrometers.
    Mx, My, Mz : list or numpy.ndarray
        Magnetization components at the mesh nodes.
    vertices_indexes : list of list of int
        Connectivity information. Each sublist contains four node indices
        (zero-based) that define a tetrahedron.
    Ms : float, optional, default=480000
        Saturation magnetization used to scale the averaged components.

    Returns
    -------
    list of tuple
        A list of length n_tetrahedrons. Each element is a tuple (Mx, My, Mz)
        containing the weighted magnetization vector of the corresponding
        tetrahedron.
    """
    # Preallocate arrays to store tetrahedron magnetizations
    n_tetrahedrons = len(vertices_indexes)
    tetrahedrons_magnetization_x = [0.0] * n_tetrahedrons
    tetrahedrons_magnetization_y = [0.0] * n_tetrahedrons
    tetrahedrons_magnetization_z = [0.0] * n_tetrahedrons

    # Loop over all tetrahedrons
    for idx in range(n_tetrahedrons):
        vertices_array = vertices_indexes[idx]
        
        # Precompute vertex coordinates (convert micrometers to meters)
        v0_x, v0_y, v0_z = X[vertices_array[0]] * 1.0e-6, Y[vertices_array[0]] * 1.0e-6, Z[vertices_array[0]] * 1.0e-6
        v1_x, v1_y, v1_z = X[vertices_array[1]] * 1.0e-6, Y[vertices_array[1]] * 1.0e-6, Z[vertices_array[1]] * 1.0e-6
        v2_x, v2_y, v2_z = X[vertices_array[2]] * 1.0e-6, Y[vertices_array[2]] * 1.0e-6, Z[vertices_array[2]] * 1.0e-6
        v3_x, v3_y, v3_z = X[vertices_array[3]] * 1.0e-6, Y[vertices_array[3]] * 1.0e-6, Z[vertices_array[3]] * 1.0e-6
        
        # Compute centroid of the tetrahedron
        centroid_x = (v0_x + v1_x + v2_x + v3_x) / 4.0
        centroid_y = (v0_y + v1_y + v2_y + v3_y) / 4.0
        centroid_z = (v0_z + v1_z + v2_z + v3_z) / 4.0
        
        # Initialize magnetization sums and total weight
        mag_tetrahedron_x, mag_tetrahedron_y, mag_tetrahedron_z = 0.0, 0.0, 0.0
        total_weight = 0.0

        # Compute distances, weights, and accumulate weighted contributions
        for i, (vx, vy, vz) in enumerate([(v0_x, v0_y, v0_z), (v1_x, v1_y, v1_z), (v2_x, v2_y, v2_z), (v3_x, v3_y, v3_z)]):
            distance = math.sqrt((vx - centroid_x) ** 2 + (vy - centroid_y) ** 2 + (vz - centroid_z) ** 2)
            weight = 1 / distance
            total_weight += weight

            # Add weighted magnetization components
            mag_tetrahedron_x += Mx[vertices_array[i]] * Ms * weight
            mag_tetrahedron_y += My[vertices_array[i]] * Ms * weight
            mag_tetrahedron_z += Mz[vertices_array[i]] * Ms * weight

        # Normalize weighted contributions and store them
        tetrahedrons_magnetization_x[idx] = mag_tetrahedron_x / total_weight
        tetrahedrons_magnetization_y[idx] = mag_tetrahedron_y / total_weight
        tetrahedrons_magnetization_z[idx] = mag_tetrahedron_z / total_weight

    # Return results as list of tuples
    return list(zip(tetrahedrons_magnetization_x, tetrahedrons_magnetization_y, tetrahedrons_magnetization_z))


    


def calculate_grain_magnetic_moments(X, Y, Z, Mx, My, Mz, vertices_indexes, Ms=480000):
    """
    Calculate the magnetic moments of a grain based on tetrahedral decomposition.

    Parameters
    ----------
    X, Y, Z : array-like
        Arrays containing the x, y, z coordinates of all vertices (in micrometers).
    Mx, My, Mz : array-like
        Arrays containing the x, y, z components of the magnetization vectors
        defined at each vertex.
    vertices_indexes : list of lists/arrays
        Each element is an array of 4 indices that define a tetrahedron by
        referencing the vertices in X, Y, Z, Mx, My, Mz.
    Ms : float, optional (default=480000)
        Saturation magnetization in A/m.

    Returns
    -------
    np.ndarray
        A 3-element array with the total magnetic moment vector of the grain:
        [Mx_total, My_total, Mz_total].
    """

    # Pre-allocate arrays to store the magnetizations of the tetrahedra
    n_tetrahedrons = len(vertices_indexes)
    tetrahedrons_magnetization_x = [0.0] * n_tetrahedrons
    tetrahedrons_magnetization_y = [0.0] * n_tetrahedrons
    tetrahedrons_magnetization_z = [0.0] * n_tetrahedrons

    # Parallelized loop (conceptually could use prange if numba.jit applied)
    for idx in range(n_tetrahedrons):
        vertices_array = vertices_indexes[idx]
        
        # Pré-alocar coordenadas dos vértices e converter de microm para metro
        v0_x, v0_y, v0_z = X[vertices_array[0]] * 1.0e-6, Y[vertices_array[0]] * 1.0e-6, Z[vertices_array[0]] * 1.0e-6
        v1_x, v1_y, v1_z = X[vertices_array[1]] * 1.0e-6, Y[vertices_array[1]] * 1.0e-6, Z[vertices_array[1]] * 1.0e-6
        v2_x, v2_y, v2_z = X[vertices_array[2]] * 1.0e-6, Y[vertices_array[2]] * 1.0e-6, Z[vertices_array[2]] * 1.0e-6
        v3_x, v3_y, v3_z = X[vertices_array[3]] * 1.0e-6, Y[vertices_array[3]] * 1.0e-6, Z[vertices_array[3]] * 1.0e-6

        ## Calcular volume do tetraedro
        # Calcular os vetores v1_v0, v2_v0 e v3_v0
        v1_v0_x, v1_v0_y, v1_v0_z = v1_x - v0_x, v1_y - v0_y, v1_z - v0_z
        v2_v0_x, v2_v0_y, v2_v0_z = v2_x - v0_x, v2_y - v0_y, v2_z - v0_z
        v3_v0_x, v3_v0_y, v3_v0_z = v3_x - v0_x, v3_y - v0_y, v3_z - v0_z
        # Calcular o produto vetorial de v2_v0 e v3_v0 (termo a termo)
        cross_x = v2_v0_y * v3_v0_z - v2_v0_z * v3_v0_y
        cross_y = v2_v0_z * v3_v0_x - v2_v0_x * v3_v0_z
        cross_z = v2_v0_x * v3_v0_y - v2_v0_y * v3_v0_x
        # Calcular o produto escalar entre v1_v0 e o vetor cruzado (cross_x, cross_y, cross_z)
        dot_product = v1_v0_x * cross_x + v1_v0_y * cross_y + v1_v0_z * cross_z
        # Calcular o volume do tetraedro
        volume = abs(dot_product) / 6
        
        # Calculate the centroid of the tetrahedron
        centroid_x = (v0_x + v1_x + v2_x + v3_x) / 4.0
        centroid_y = (v0_y + v1_y + v2_y + v3_y) / 4.0
        centroid_z = (v0_z + v1_z + v2_z + v3_z) / 4.0
        
        # Initialize magnetization variables and total weight
        mag_tetrahedron_x, mag_tetrahedron_y, mag_tetrahedron_z = 0.0, 0.0, 0.0
        total_weight = 0.0

        # Calculate distances, weights, and accumulate weighted contributions of each vertex
        for i, (vx, vy, vz) in enumerate([(v0_x, v0_y, v0_z), (v1_x, v1_y, v1_z), (v2_x, v2_y, v2_z), (v3_x, v3_y, v3_z)]):
            distance = math.sqrt((vx - centroid_x) ** 2 + (vy - centroid_y) ** 2 + (vz - centroid_z) ** 2)
            weight = 1 / distance
            total_weight += weight

            # Accumulate weighted magnetization components
            mag_tetrahedron_x += Mx[vertices_array[i]] * Ms * weight
            mag_tetrahedron_y += My[vertices_array[i]] * Ms * weight
            mag_tetrahedron_z += Mz[vertices_array[i]] * Ms * weight

        # Normalize contributions and store in pre-allocated arrays
        tetrahedrons_magnetization_x[idx] = mag_tetrahedron_x / total_weight * volume
        tetrahedrons_magnetization_y[idx] = mag_tetrahedron_y / total_weight * volume
        tetrahedrons_magnetization_z[idx] = mag_tetrahedron_z / total_weight * volume

    # Return the summed total magnetic moment vector
    return np.array([
        np.sum(tetrahedrons_magnetization_x), 
        np.sum(tetrahedrons_magnetization_y), 
        np.sum(tetrahedrons_magnetization_z)
    ])




def plot_maps(Bx_final, By_final, Bz_final, coordinates, height, title):
    """
    Plot 2D contour maps of the magnetic field components (Bx, By, Bz) 
    at a given sensor height.

    Parameters
    ----------
    Bx_final, By_final, Bz_final : 2D array-like
        Gridded values of the magnetic field components in Tesla.
    coordinates : tuple or list of arrays
        (X, Y) coordinates of the grid points in meters.
    height : float
        Sensor height above the reference plane, in micrometers.
    title : str
        Title for the figure.

    Returns
    -------
    None
        Displays the 1x3 subplot figure with contour maps.
    """

    # Cria as variáveis x e y (converter para micrometros)
    x = coordinates[0] * 1.0e6
    y = coordinates[1] * 1.0e6
    cmap = 'seismic'
    levels = 100
    
    # Configuração do subplot 1x3
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Escala de cores baseada no campo máximo de Bz
    scale = np.max(abs(Bz_final.ravel()))
    scale *= 1.0e9
    
    # ---- Plot Bx_final ----
    ax = axes[0]
    contour = ax.contourf(x, y, Bx_final * 1.0e9, cmap=cmap, levels=levels, vmax=scale, vmin=-scale)
    fig.colorbar(contour, ax=ax, label='nT')
    ax.set_xlabel(r'X $\mu m$')
    ax.set_ylabel(r'Y $\mu m$')
    ax.set_title(f'$B_x$ at height = {height} $\mu m$')
    
    # ---- Plot By_final ----
    ax = axes[1]
    contour = ax.contourf(x, y, By_final * 1.0e9, cmap=cmap, levels=levels, vmax=scale, vmin=-scale)
    fig.colorbar(contour, ax=ax, label='nT')
    ax.set_xlabel(r'X $\mu m$')
    ax.set_ylabel(r'Y $\mu m$')
    ax.set_title(f'$B_y$ at height = {height} $\mu m$')
    
    # ---- Plot Bz_final ----
    ax = axes[2]
    contour = ax.contourf(x, y, Bz_final * 1.0e9, cmap=cmap, levels=levels, vmax=scale, vmin=-scale)
    fig.colorbar(contour, ax=ax, label='nT')
    ax.set_xlabel(r'X $\mu m$')
    ax.set_ylabel(r'Y $\mu m$')
    ax.set_title(f'$B_z$ at height = {height} $\mu m$')

    # Título geral e ajuste de layout
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()




@jit(nopython=True, parallel=True)
def calculate_Omega_f_and_Delta_W_f(tetrahedrons, tetrahedrons_mag, coordinates, Bx, By, Bz):
    """
    Compute the magnetic field contribution from tetrahedral elements using 
    face-based solid angle and edge integral formulations.

    For each observation point, this function iterates over all tetrahedrons 
    and their faces, computing:
    - The outward normal of each triangular face
    - The solid angle (Omega_f) subtended by the face
    - Edge contributions (W_f) along each of the three face edges
    - The combined correction term (Delta_W_f)
    - The magnetic field contribution (Bx, By, Bz) from the tetrahedron’s 
      magnetization vector

    The computation is fully vectorized inside `numba` JIT with parallelization 
    across observation points.

    Parameters
    ----------
    tetrahedrons : list
        List of tetrahedrons, where each tetrahedron is defined by 4 triangular faces.
        Each face is represented by 3 vertices, and each vertex is a 3D coordinate.
    tetrahedrons_mag : numpy.ndarray
        Array of shape (n, 3) containing the magnetization vector [Mx, My, Mz]
        for each tetrahedron.
    coordinates : tuple of numpy.ndarray
        Tuple (X, Y, Z), each flattened, representing observation point coordinates.
    Bx, By, Bz : numpy.ndarray
        Arrays to accumulate the magnetic field contributions at each observation point.

    Returns
    -------
    None
        The function updates Bx, By, Bz in place. At the end, they are scaled by `-constant`.

    Notes
    -----
    - `Omega_f` is computed using the solid angle formula for a triangle.
    - `W_f` contributions are calculated edge by edge using logarithmic expressions.
    - The right-hand rule (vertex order) determines the orientation of normals.
    - Assumes `constant` is defined globally before calling this function.
    """
    
    easting = coordinates[0].ravel() 
    northing = coordinates[1].ravel()
    upward = coordinates[2].ravel()

    for i in prange(len(easting)):
        r_x, r_y, r_z = easting[i], northing[i], upward[i]
 
        for j in range(len(tetrahedrons_mag)):
            faces = tetrahedrons[j]
            M = tetrahedrons_mag[j]

            for k in range(len(faces)):
                # ------------------- Calculate face normal -------------------
                r1_x, r1_y, r1_z = faces[k][0][0], faces[k][0][1], faces[k][0][2]
                r2_x, r2_y, r2_z = faces[k][1][0], faces[k][1][1], faces[k][1][2]
                r3_x, r3_y, r3_z = faces[k][2][0], faces[k][2][1], faces[k][2][2]
                
                v1_x, v1_y, v1_z = r2_x - r1_x, r2_y - r1_y, r2_z - r1_z 
                v2_x, v2_y, v2_z = r3_x - r1_x, r3_y - r1_y, r3_z - r1_z

                n_f_x = v1_y * v2_z - v1_z * v2_y
                n_f_y = v1_z * v2_x - v1_x * v2_z
                n_f_z = v1_x * v2_y - v1_y * v2_x

                norm = math.sqrt(n_f_x**2 + n_f_y**2 + n_f_z**2)
                n_f_x, n_f_y, n_f_z = n_f_x / norm, n_f_y / norm, n_f_z / norm
                # ------------------- Calculate face normal -------------------
                
                # ------------------- Calculate Omega_f -----------------------
                v1_x, v1_y, v1_z = r1_x - r_x, r1_y - r_y, r1_z - r_z
                v2_x, v2_y, v2_z = r2_x - r_x, r2_y - r_y, r2_z - r_z
                v3_x, v3_y, v3_z = r3_x - r_x, r3_y - r_y, r3_z - r_z

                v1_norm = math.sqrt(v1_x**2 + v1_y**2 + v1_z**2)
                v2_norm = math.sqrt(v2_x**2 + v2_y**2 + v2_z**2)
                v3_norm = math.sqrt(v3_x**2 + v3_y**2 + v3_z**2)

                dot_v1_v2 = v1_x * v2_x + v1_y * v2_y + v1_z * v2_z
                dot_v1_v3 = v1_x * v3_x + v1_y * v3_y + v1_z * v3_z
                dot_v2_v3 = v2_x * v3_x + v2_y * v3_y + v2_z * v3_z

                cross_v2_v3_x = v2_y * v3_z - v2_z * v3_y
                cross_v2_v3_y = v2_z * v3_x - v2_x * v3_z
                cross_v2_v3_z = v2_x * v3_y - v2_y * v3_x

                D = (v1_norm * v2_norm * v3_norm +
                     v3_norm * dot_v1_v2 +
                     v2_norm * dot_v1_v3 +
                     v1_norm * dot_v2_v3)

                numerator = v1_x * cross_v2_v3_x + v1_y * cross_v2_v3_y + v1_z * cross_v2_v3_z
                Omega_f = 2 * math.atan2(numerator, D)
                # ------------------- Calculate Omega_f -----------------------
 
                # ------------------- Calculate W_f (edge contributions) ------
                W_e_acum_x, W_e_acum_y, W_e_acum_z = 0.0, 0.0, 0.0

                # Edge r1-r2
                v1_norm = math.sqrt((r1_x - r_x)**2 + (r1_y - r_y)**2 + (r1_z - r_z)**2)
                v2_norm = math.sqrt((r2_x - r_x)**2 + (r2_y - r_y)**2 + (r2_z - r_z)**2)
                v3_x, v3_y, v3_z = r2_x - r1_x, r2_y - r1_y, r2_z - r1_z
                v3_norm = math.sqrt(v3_x**2 + v3_y**2 + v3_z**2)
                u_e_x, u_e_y, u_e_z = v3_x / v3_norm, v3_y / v3_norm, v3_z / v3_norm
                W_e = math.log((v2_norm + v1_norm + v3_norm) / (v2_norm + v1_norm - v3_norm))
                cross_product_x = n_f_y * u_e_z - n_f_z * u_e_y
                cross_product_y = n_f_z * u_e_x - n_f_x * u_e_z
                cross_product_z = n_f_x * u_e_y - n_f_y * u_e_x
                W_e_acum_x += cross_product_x * W_e
                W_e_acum_y += cross_product_y * W_e
                W_e_acum_z += cross_product_z * W_e

                # Edge r2-r3
                v1_norm = math.sqrt((r2_x - r_x)**2 + (r2_y - r_y)**2 + (r2_z - r_z)**2)
                v2_norm = math.sqrt((r3_x - r_x)**2 + (r3_y - r_y)**2 + (r3_z - r_z)**2)
                v3_x, v3_y, v3_z = r3_x - r2_x, r3_y - r2_y, r3_z - r2_z
                v3_norm = math.sqrt(v3_x**2 + v3_y**2 + v3_z**2)
                u_e_x, u_e_y, u_e_z = v3_x / v3_norm, v3_y / v3_norm, v3_z / v3_norm
                W_e = math.log((v2_norm + v1_norm + v3_norm) / (v2_norm + v1_norm - v3_norm))
                cross_product_x = n_f_y * u_e_z - n_f_z * u_e_y
                cross_product_y = n_f_z * u_e_x - n_f_x * u_e_z
                cross_product_z = n_f_x * u_e_y - n_f_y * u_e_x
                W_e_acum_x += cross_product_x * W_e
                W_e_acum_y += cross_product_y * W_e
                W_e_acum_z += cross_product_z * W_e

                # Edge r3-r1
                v1_norm = math.sqrt((r3_x - r_x)**2 + (r3_y - r_y)**2 + (r3_z - r_z)**2)
                v2_norm = math.sqrt((r1_x - r_x)**2 + (r1_y - r_y)**2 + (r1_z - r_z)**2)
                v3_x, v3_y, v3_z = r1_x - r3_x, r1_y - r3_y, r1_z - r3_z
                v3_norm = math.sqrt(v3_x**2 + v3_y**2 + v3_z**2)
                u_e_x, u_e_y, u_e_z = v3_x / v3_norm, v3_y / v3_norm, v3_z / v3_norm
                W_e = math.log((v2_norm + v1_norm + v3_norm) / (v2_norm + v1_norm - v3_norm))
                cross_product_x = n_f_y * u_e_z - n_f_z * u_e_y
                cross_product_y = n_f_z * u_e_x - n_f_x * u_e_z
                cross_product_z = n_f_x * u_e_y - n_f_y * u_e_x
                W_e_acum_x += cross_product_x * W_e
                W_e_acum_y += cross_product_y * W_e
                W_e_acum_z += cross_product_z * W_e
                # ------------------- Calculate W_f ----------------------------

                # ------------------- Main formula -----------------------------
                Delta_W_f_x = W_e_acum_x + n_f_x * Omega_f
                Delta_W_f_y = W_e_acum_y + n_f_y * Omega_f
                Delta_W_f_z = W_e_acum_z + n_f_z * Omega_f

                cross_product_x = M[1] * n_f_z - M[2] * n_f_y
                cross_product_y = M[2] * n_f_x - M[0] * n_f_z
                cross_product_z = M[0] * n_f_y - M[1] * n_f_x

                bx = cross_product_y * Delta_W_f_z - cross_product_z * Delta_W_f_y
                by = cross_product_z * Delta_W_f_x - cross_product_x * Delta_W_f_z
                bz = cross_product_x * Delta_W_f_y - cross_product_y * Delta_W_f_x
                # ------------------- Main formula -----------------------------
                
                Bx[i] += bx
                By[i] += by
                Bz[i] += bz

    Bx *= -constant
    By *= -constant
    Bz *= -constant

