import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting

# =============================================================================
# 1. USER-DEFINED PARAMETERS
# =============================================================================
# Define the objective function:
#   maximize   z = c1*x + c2*y
# Change these coefficients as desired.
c1, c2 = 55, 1
objective_coeffs = (c1, c2)

# Define constraints in the form: a*x + b*y <= c.
# For example, the following constraints represent:
#   1*x + 1*y <= 4    (i.e. x + y <= 4)
#   1*x + 0*y <= 2    (i.e. x <= 2)
#   0*x + 1*y <= 3    (i.e. y <= 3)
constraints = [
    (1, 1, 4),
    (1, 0, 2),
    (0, 1, 3)
]

# Automatically add non-negativity constraints: x >= 0 and y >= 0.
# To express x >= 0 in "a*x + b*y <= c" form, we write -1*x <= 0.
constraints += [(-1, 0, 0), (0, -1, 0)]

# =============================================================================
# 2. HELPER FUNCTIONS TO COMPUTE THE FEASIBLE REGION
# =============================================================================
def compute_intersections(constraints):
    """
    For every pair of constraint boundaries (given by a*x+b*y=c),
    compute the intersection point.
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
                continue  # Lines are parallel (or nearly so)
            x = (c1_val * b2 - c2_val * b1) / det
            y = (a1 * c2_val - a2 * c1_val) / det
            pts.append((x, y))
    return np.array(pts)

def is_feasible(pt, constraints, tol=1e-7):
    """Check if point pt satisfies all constraints."""
    x, y = pt
    for (a, b, c_val) in constraints:
        if a*x + b*y > c_val + tol:
            return False
    return True

def unique_points(points, tol=1e-7):
    """Remove duplicate points (within a given tolerance)."""
    unique = []
    for p in points:
        if not any(np.linalg.norm(np.array(p) - np.array(q)) < tol for q in unique):
            unique.append(p)
    return np.array(unique)

def sort_vertices(vertices):
    """
    Given a list of vertices (as a 2D numpy array), sort them in
    counterclockwise order around their centroid.
    """
    centroid = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return vertices[sorted_indices]

def objective_value(pt, objective_coeffs):
    """Compute the objective function value at point pt."""
    return objective_coeffs[0]*pt[0] + objective_coeffs[1]*pt[1]

# -----------------------------------------------------------------------------
# Compute all candidate intersection points.
all_points = compute_intersections(constraints)

# Filter only those points that are feasible.
feasible_points = np.array([pt for pt in all_points if is_feasible(pt, constraints)])

# Remove near-duplicates.
feasible_points = unique_points(feasible_points)

# Sort the vertices to form the feasible region polygon.
polygon = sort_vertices(feasible_points)

# =============================================================================
# 3. DETERMINE A SIMPLEX-LIKE PATH ALONG THE POLYGON BOUNDARY
# =============================================================================
# Evaluate the objective at each vertex.
obj_values = np.array([objective_value(pt, objective_coeffs) for pt in polygon])

# Choose a starting vertex (here, the one with minimum objective value)
start_index = np.argmin(obj_values)
# Identify the optimal vertex (maximum objective value)
opt_index = np.argmax(obj_values)

def compute_simplex_path(polygon, obj_values, start_index, opt_index):
    """
    In a convex polygon, the objective values vary unimodally along the
    boundary. To simulate a simplex pivot, we choose the arc (sequence of
    adjacent vertices) from the starting vertex to the optimum vertex that
    is monotonic (non-decreasing) in the objective function.
    """
    n = len(polygon)
    # Arc 1: moving forward (in the order of the sorted polygon) from start to optimum.
    arc1 = []
    i = start_index
    while True:
        arc1.append(i)
        if i == opt_index:
            break
        i = (i + 1) % n
        if i == start_index:  # safety check
            break

    # Arc 2: moving backward from start to optimum.
    arc2 = []
    i = start_index
    while True:
        arc2.append(i)
        if i == opt_index:
            break
        i = (i - 1) % n
        if i == start_index:
            break

    def is_monotonic(arc):
        vals = obj_values[arc]
        # Allow for a tiny tolerance
        return np.all(np.diff(vals) >= -1e-7)

    monotonic1 = is_monotonic(arc1)
    monotonic2 = is_monotonic(arc2)

    # Choose the arc that is monotonic.
    if monotonic1 and monotonic2:
        chosen_arc = arc1 if len(arc1) <= len(arc2) else arc2
    elif monotonic1:
        chosen_arc = arc1
    elif monotonic2:
        chosen_arc = arc2
    else:
        # Fallback: choose arc1.
        chosen_arc = arc1

    return polygon[chosen_arc]

# Compute the simplex path (a sequence of vertices).
simplex_path = compute_simplex_path(polygon, obj_values, start_index, opt_index)

# =============================================================================
# 4. SET UP THE VISUALIZATION (2D and 3D ANIMATION)
# =============================================================================
# Create a figure with two subplots: one for 2D and one for 3D.
fig = plt.figure(figsize=(12, 5))

# ----- 2D Axes (Left) -----
ax2d = fig.add_subplot(121)
# Close the polygon by repeating the first vertex.
polygon_closed = np.vstack([polygon, polygon[0]])
ax2d.plot(polygon_closed[:, 0], polygon_closed[:, 1], 'k-', lw=2, label='Feasible Region')

# Artists for the simplex path (red line with markers),
# the objective function level curve (blue dashed line),
# and a text box for the current step.
path_line_2d, = ax2d.plot([], [], 'ro-', lw=2, markersize=8, label='Simplex Path')
objective_line_2d, = ax2d.plot([], [], 'b--', lw=2, label='Objective Level Curve')
text_obj = ax2d.text(0.05, 0.95, '', transform=ax2d.transAxes, fontsize=12,
                     verticalalignment='top')

ax2d.set_xlabel('x')
ax2d.set_ylabel('y')
ax2d.set_title('Simplex Algorithm Demonstration (2D)')
ax2d.legend(loc='lower right')

# Set plot limits based on the polygon (with a margin).
x_min, x_max = np.min(polygon[:, 0]), np.max(polygon[:, 0])
y_min, y_max = np.min(polygon[:, 1]), np.max(polygon[:, 1])
ax2d.set_xlim(x_min - 1, x_max + 1)
ax2d.set_ylim(y_min - 1, y_max + 1)

# ----- 3D Axes (Right) -----
ax3d = fig.add_subplot(122, projection='3d')

# Create a mesh grid for the objective function surface.
x_vals = np.linspace(x_min - 1, x_max + 1, 50)
y_vals = np.linspace(y_min - 1, y_max + 1, 50)
X, Y = np.meshgrid(x_vals, y_vals)
Z = c1 * X + c2 * Y
# Plot the objective function surface (with transparency).
ax3d.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis', edgecolor='none')

# Plot the feasible region polygon (its vertices lifted to the surface).
polygon_z = np.array([objective_value(pt, objective_coeffs) for pt in polygon_closed])
ax3d.plot(polygon_closed[:, 0], polygon_closed[:, 1], polygon_z, 'k-', lw=2, label='Feasible Region')

# Create 3D artists for the simplex path and the current level curve.
path_line_3d, = ax3d.plot([], [], [], 'ro-', lw=2, markersize=8, label='Simplex Path')
objective_line_3d, = ax3d.plot([], [], [], 'b--', lw=2, label='Objective Level Curve')

ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('z')
ax3d.set_title('Objective Function Surface (3D)')
ax3d.legend(loc='lower right')

# =============================================================================
# 5. ANIMATION FUNCTIONS
# =============================================================================
def compute_level_curve(z_val, objective_coeffs, xlim, ylim, num_points=200):
    """
    Given an objective value z_val and the objective coefficients,
    compute a set of (x, y) points for the level curve (i.e. all points
    satisfying c1*x + c2*y = z_val). Handles the case where c2 is zero.
    """
    c1, c2 = objective_coeffs
    if abs(c2) > 1e-7:
        x_vals = np.linspace(xlim[0], xlim[1], num_points)
        y_vals = (z_val - c1 * x_vals) / c2
    elif abs(c1) > 1e-7:
        # If c2 is nearly zero, the level curve is vertical: x = z_val/c1.
        x_vals = np.full(num_points, z_val / c1)
        y_vals = np.linspace(ylim[0], ylim[1], num_points)
    else:
        x_vals, y_vals = np.array([]), np.array([])
    return x_vals, y_vals

def init():
    """Initialize animated artists for both 2D and 3D views."""
    path_line_2d.set_data([], [])
    objective_line_2d.set_data([], [])
    path_line_3d.set_data([], [])
    path_line_3d.set_3d_properties([])
    objective_line_3d.set_data([], [])
    objective_line_3d.set_3d_properties([])
    text_obj.set_text('')
    return (path_line_2d, objective_line_2d, path_line_3d, objective_line_3d, text_obj)

def animate(i):
    """Update the animation for frame i in both 2D and 3D views."""
    # Update the simplex path so far.
    current_path = simplex_path[:i + 1]
    # 2D update.
    path_line_2d.set_data(current_path[:, 0], current_path[:, 1])
    # 3D update: compute z-coordinates from the objective function.
    current_path_z = np.array([objective_value(pt, objective_coeffs) for pt in current_path])
    path_line_3d.set_data(current_path[:, 0], current_path[:, 1])
    path_line_3d.set_3d_properties(current_path_z)

    # For the current vertex, compute the objective value.
    current_vertex = simplex_path[i]
    z_val = objective_value(current_vertex, objective_coeffs)

    # Compute and set the level curve for z = c1*x + c2*y.
    # 2D view.
    x_vals_curve, y_vals_curve = compute_level_curve(z_val, objective_coeffs, ax2d.get_xlim(), ax2d.get_ylim())
    objective_line_2d.set_data(x_vals_curve, y_vals_curve)
    # 3D view: level curve on the surface has constant z = z_val.
    objective_line_3d.set_data(x_vals_curve, y_vals_curve)
    objective_line_3d.set_3d_properties(np.full_like(x_vals_curve, z_val))

    # Update the text annotation.
    text_obj.set_text(f"Step {i}:\nCurrent vertex: ({current_vertex[0]:.2f}, {current_vertex[1]:.2f})\nObjective: z = {z_val:.2f}")

    return (path_line_2d, objective_line_2d, path_line_3d, objective_line_3d, text_obj)

# Create the animation (each frame lasts 1500 ms).
ani = FuncAnimation(fig, animate, frames=len(simplex_path),
                    init_func=init, interval=1500, blit=True, repeat=False)

plt.tight_layout()
plt.show()
