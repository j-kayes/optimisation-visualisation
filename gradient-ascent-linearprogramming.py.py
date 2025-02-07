import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# =============================================================================
# 1. USER-DEFINED PARAMETERS
# =============================================================================
# Objective function: maximize f(x,y) = c1*x + c2*y.
c1, c2 = 3, 2
objective_coeffs = (c1, c2)

# Define constraints in the form: a*x + b*y <= c.
# For example:
#   (1, 1, 4) represents: x + y <= 4
#   (1, 0, 2) represents: x <= 2
#   (0, 1, 3) represents: y <= 3
constraints = [
    (1, 1, 4),
    (1, 0, 2),
    (0, 1, 3)
]

# Automatically add non-negativity constraints: x >= 0, y >= 0.
# In inequality form, x >= 0 is equivalent to -1*x <= 0.
constraints += [(-1, 0, 0), (0, -1, 0)]

# =============================================================================
# 2. HELPER FUNCTIONS FOR THE FEASIBLE REGION (POLYGON) COMPUTATION
# =============================================================================
def compute_intersections(constraints):
    """
    Compute intersections of all pairs of constraint boundaries (lines defined by a*x+b*y=c).
    Returns an array of candidate intersection points.
    """
    pts = []
    n = len(constraints)
    tol = 1e-7
    for i in range(n):
        a1, b1, c1_val = constraints[i]
        for j in range(i+1, n):
            a2, b2, c2_val = constraints[j]
            det = a1 * b2 - a2 * b1
            if abs(det) < tol:
                continue  # Lines are parallel or nearly so.
            x = (c1_val * b2 - c2_val * b1) / det
            y = (a1 * c2_val - a2 * c1_val) / det
            pts.append((x, y))
    return np.array(pts)

def is_feasible(pt, constraints, tol=1e-7):
    """Check if point pt satisfies all the constraints."""
    x, y = pt
    for (a, b, c_val) in constraints:
        if a*x + b*y > c_val + tol:
            return False
    return True

def unique_points(points, tol=1e-7):
    """Remove duplicate points (within a given tolerance)."""
    unique = []
    for p in points:
        if not any(np.linalg.norm(np.array(p)-np.array(q)) < tol for q in unique):
            unique.append(p)
    return np.array(unique)

def sort_vertices(vertices):
    """
    Sort the vertices in counterclockwise order around their centroid.
    (Assumes the vertices form a convex polygon.)
    """
    centroid = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1]-centroid[1], vertices[:, 0]-centroid[0])
    sorted_indices = np.argsort(angles)
    return vertices[sorted_indices]

# -----------------------------------------------------------------------------
# Compute candidate intersection points.
all_points = compute_intersections(constraints)
# Keep only feasible points.
feasible_points = np.array([pt for pt in all_points if is_feasible(pt, constraints)])
feasible_points = unique_points(feasible_points)
# Sort vertices to form the convex polygon.
polygon = sort_vertices(feasible_points)

# =============================================================================
# 3. PROJECTION FUNCTIONS
# =============================================================================
def point_in_convex_polygon(P, polygon, tol=1e-7):
    """
    Check if point P is inside a convex polygon.
    This function computes the cross products for each edge.
    """
    num = len(polygon)
    sign = None
    for i in range(num):
        A = polygon[i]
        B = polygon[(i+1) % num]
        edge = B - A
        vp = P - A
        cross = edge[0]*vp[1] - edge[1]*vp[0]
        if abs(cross) < tol:
            continue  # On the edge is acceptable.
        current_sign = np.sign(cross)
        if sign is None:
            sign = current_sign
        elif current_sign != sign:
            return False
    return True

def project_point_to_segment(P, A, B):
    """
    Projects point P onto the line segment AB.
    Returns the projection point.
    """
    A, B, P = np.array(A), np.array(B), np.array(P)
    AB = B - A
    t = np.dot(P - A, AB) / np.dot(AB, AB)
    t_clamped = np.clip(t, 0, 1)
    projection = A + t_clamped * AB
    return projection

def project_onto_polygon(P, polygon):
    """
    Projects point P onto the convex polygon (defined by vertices in order).
    If P is inside the polygon, returns P.
    Otherwise, computes the projection onto each edge and returns the one
    closest (in Euclidean distance) to P.
    """
    P = np.array(P)
    if point_in_convex_polygon(P, polygon):
        return P
    proj_candidates = []
    num = len(polygon)
    for i in range(num):
        A = polygon[i]
        B = polygon[(i+1) % num]
        proj = project_point_to_segment(P, A, B)
        proj_candidates.append(proj)
    proj_candidates = np.array(proj_candidates)
    # Choose the candidate with minimum Euclidean distance to P.
    dists = np.linalg.norm(proj_candidates - P, axis=1)
    best = proj_candidates[np.argmin(dists)]
    return best

# =============================================================================
# 4. PROJECTED GRADIENT ASCENT ALGORITHM
# =============================================================================
def f(x, y):
    """Objective function: f(x,y) = c1*x + c2*y."""
    return c1 * x + c2 * y

def grad_f(x, y):
    """Gradient of the linear function (constant)."""
    return np.array([c1, c2])

# Gradient ascent parameters
alpha = 0.1            # Step size (learning rate)
tol = 1e-5             # Tolerance for stopping
max_iters = 50         # Maximum number of iterations

# Starting point (should be feasible)
current_point = np.array([0.0, 0.0])
path = [current_point.copy()]

for i in range(max_iters):
    # Unconstrained update: move in the gradient direction.
    step = alpha * grad_f(current_point[0], current_point[1])
    candidate = current_point + step
    # Project candidate onto the feasible region.
    projected = project_onto_polygon(candidate, polygon)
    # If the update is very small, assume convergence.
    if np.linalg.norm(projected - current_point) < tol:
        current_point = projected
        path.append(current_point.copy())
        break
    current_point = projected
    path.append(current_point.copy())

path = np.array(path)
# The optimum should be reached (or approximated) at the vertex with maximum f(x,y).

# =============================================================================
# 5. SET UP THE VISUALIZATION (2D and 3D)
# =============================================================================
# Create a figure with two subplots.
fig = plt.figure(figsize=(12, 5))

# ----- 2D Axes (Left) -----
ax2d = fig.add_subplot(121)
# Close the polygon by repeating the first vertex.
polygon_closed = np.vstack([polygon, polygon[0]])
ax2d.plot(polygon_closed[:, 0], polygon_closed[:, 1], 'k-', lw=2, label='Feasible Region')

# Artists for the projected gradient ascent path (red) and the level curve (blue dashed).
path_line_2d, = ax2d.plot([], [], 'ro-', lw=2, markersize=8, label='Ascent Path')
level_curve_2d, = ax2d.plot([], [], 'b--', lw=2, label='Level Curve')

ax2d.set_xlabel('x')
ax2d.set_ylabel('y')
ax2d.set_title('Projected Gradient Ascent (2D)')
ax2d.legend(loc='lower right')

# Set plot limits with some margin.
x_min, x_max = np.min(polygon[:,0]), np.max(polygon[:,0])
y_min, y_max = np.min(polygon[:,1]), np.max(polygon[:,1])
ax2d.set_xlim(x_min - 1, x_max + 1)
ax2d.set_ylim(y_min - 1, y_max + 1)

def compute_level_curve_linear(z_val, objective_coeffs, xlim, num_points=200):
    """
    For a linear function f(x,y)=c1*x+c2*y=z_val,
    solve for y: y = (z_val - c1*x)/c2.
    """
    c1, c2 = objective_coeffs
    x_vals = np.linspace(xlim[0], xlim[1], num_points)
    if abs(c2) > 1e-7:
        y_vals = (z_val - c1 * x_vals) / c2
    else:
        y_vals = np.full_like(x_vals, z_val)
    return x_vals, y_vals

# ----- 3D Axes (Right) -----
ax3d = fig.add_subplot(122, projection='3d')
# Create a mesh for the objective plane.
X_plane = np.linspace(x_min - 1, x_max + 1, 50)
Y_plane = np.linspace(y_min - 1, y_max + 1, 50)
X_plane, Y_plane = np.meshgrid(X_plane, Y_plane)
Z_plane = c1 * X_plane + c2 * Y_plane
# Plot the objective function plane.
ax3d.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.5, cmap='viridis', edgecolor='none')

# Plot the feasible polygon (lifted onto the objective plane).
polygon_z = np.array([f(pt[0], pt[1]) for pt in polygon_closed])
ax3d.plot(polygon_closed[:, 0], polygon_closed[:, 1], polygon_z, 'k-', lw=2, label='Feasible Region')

# Artists for the ascent path and current level curve in 3D.
path_line_3d, = ax3d.plot([], [], [], 'ro-', lw=2, markersize=8, label='Ascent Path')
level_curve_3d, = ax3d.plot([], [], [], 'b--', lw=2, label='Level Curve')

ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('f(x,y)')
ax3d.set_title('Objective Function Surface (3D)')
ax3d.legend(loc='lower right')

# =============================================================================
# 6. ANIMATION FUNCTIONS
# =============================================================================
def init():
    path_line_2d.set_data([], [])
    level_curve_2d.set_data([], [])
    path_line_3d.set_data([], [])
    path_line_3d.set_3d_properties([])
    level_curve_3d.set_data([], [])
    level_curve_3d.set_3d_properties([])
    return path_line_2d, level_curve_2d, path_line_3d, level_curve_3d

def animate(i):
    # Update the ascent path up to iteration i.
    current_path = path[:i+1]
    path_line_2d.set_data(current_path[:, 0], current_path[:, 1])
    # 3D: use the objective value to set z-coordinates.
    current_path_z = f(current_path[:, 0], current_path[:, 1])
    path_line_3d.set_data(current_path[:, 0], current_path[:, 1])
    path_line_3d.set_3d_properties(current_path_z)
    
    # For the current point, compute the objective value.
    current_point = current_path[-1]
    z_val = f(current_point[0], current_point[1])
    # Compute and update the level curve for f(x,y)=z_val.
    x_vals_curve, y_vals_curve = compute_level_curve_linear(z_val, objective_coeffs, ax2d.get_xlim())
    level_curve_2d.set_data(x_vals_curve, y_vals_curve)
    level_curve_3d.set_data(x_vals_curve, y_vals_curve)
    level_curve_3d.set_3d_properties(np.full_like(x_vals_curve, z_val))
    
    # Optionally, update the 2D title to show progress.
    ax2d.set_title(f'Projected Gradient Ascent (2D)\nStep {i}: f(x,y) = {z_val:.2f}')
    
    return path_line_2d, level_curve_2d, path_line_3d, level_curve_3d

# Create the animation (each frame lasts 1000 ms).
ani = FuncAnimation(fig, animate, frames=len(path),
                    init_func=init, interval=1000, blit=True, repeat=False)

plt.tight_layout()
plt.show()
