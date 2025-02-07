import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting

# =============================================================================
# 1. DEFINE THE FUNCTION AND ITS GRADIENT
# =============================================================================
# Here we use:
#    f(x,y) = -((x-1)**2 + (y-2)**2) + 10,
# which is concave with a unique maximum at (1,2)
def f(x, y):
    return -((x - 1)**2 + (y - 2)**2) + 10

def grad_f(x, y):
    # The gradient of f is: [df/dx, df/dy] = [-2*(x-1), -2*(y-2)]
    return np.array([-2*(x - 1), -2*(y - 2)])

# =============================================================================
# 2. GRADIENT ASCENT PARAMETERS & PATH COMPUTATION
# =============================================================================
alpha = 0.1            # step size (learning rate)
num_iterations = 20    # number of gradient ascent steps
x0, y0 = -2, -2        # starting point

# Compute the gradient ascent path
points = [(x0, y0)]
for i in range(num_iterations):
    current = np.array(points[-1])
    gradient = grad_f(current[0], current[1])
    next_point = current + alpha * gradient
    points.append(tuple(next_point))
points = np.array(points)

# =============================================================================
# 3. SET UP THE PLOTTING DOMAIN & MESH FOR CONTOUR/SURFACE
# =============================================================================
# Define a plotting domain
x_min, x_max = -3, 5
y_min, y_max = -3, 5

# Create a grid for contour and surface plots
x_vals = np.linspace(x_min, x_max, 100)
y_vals = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# =============================================================================
# 4. SET UP THE FIGURE WITH 2D AND 3D SUBPLOTS
# =============================================================================
fig = plt.figure(figsize=(12, 5))

# ----- 2D subplot: contour plot with gradient ascent path -----
ax2d = fig.add_subplot(121)
# Draw contour (level) curves of f(x,y)
contour_levels = np.linspace(np.min(Z), np.max(Z), 20)
ax2d.contour(X, Y, Z, levels=contour_levels, cmap='viridis')
ax2d.set_xlabel('x')
ax2d.set_ylabel('y')
ax2d.set_title('Gradient Ascent (2D)')

# Artists to be updated during animation:
# - The gradient ascent path (red line with markers)
# - The current level curve (blue dashed line)
path_line_2d, = ax2d.plot([], [], 'ro-', lw=2, markersize=6, label='Ascent Path')
level_curve_2d, = ax2d.plot([], [], 'b--', lw=2, label='Level Curve')
# (We will also show a gradient arrow that updates each frame.)
gradient_quiver = None  # global variable to hold the arrow

ax2d.legend()

# ----- 3D subplot: surface plot with gradient ascent path -----
ax3d = fig.add_subplot(122, projection='3d')
surf = ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('f(x,y)')
ax3d.set_title('Gradient Ascent (3D)')

# Artists for the 3D view: the ascent path and the level curve (on the surface)
path_line_3d, = ax3d.plot([], [], [], 'ro-', lw=2, markersize=6, label='Ascent Path')
level_curve_3d, = ax3d.plot([], [], [], 'b--', lw=2, label='Level Curve')
ax3d.legend()

# =============================================================================
# 5. HELPER FUNCTION: COMPUTE LEVEL CURVE POINTS
# =============================================================================
def compute_level_curve_points(current_value, num_points=200):
    """
    For our quadratic function, the level set f(x,y)=current_value satisfies
         -((x-1)**2+(y-2)**2)+10 = current_value,
    which rearranges to:
         (x-1)**2+(y-2)**2 = 10 - current_value.
    When 10 - current_value >= 0, the level set is a circle (an ellipse in the
    general case) centered at (1,2) with radius sqrt(10 - current_value).
    """
    radius_sq = 10 - current_value
    if radius_sq < 0:
        return np.array([]), np.array([])
    r = np.sqrt(radius_sq)
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_curve = 1 + r * np.cos(theta)
    y_curve = 2 + r * np.sin(theta)
    return x_curve, y_curve

# =============================================================================
# 6. ANIMATION FUNCTIONS
# =============================================================================
def init():
    global gradient_quiver
    path_line_2d.set_data([], [])
    level_curve_2d.set_data([], [])
    path_line_3d.set_data([], [])
    path_line_3d.set_3d_properties([])
    level_curve_3d.set_data([], [])
    level_curve_3d.set_3d_properties([])
    if gradient_quiver is not None:
        gradient_quiver.remove()
    return path_line_2d, level_curve_2d, path_line_3d, level_curve_3d

def animate(i):
    global gradient_quiver
    # ----- Update the gradient ascent path (both 2D and 3D) -----
    current_path = points[:i + 1]
    # Update 2D path
    path_line_2d.set_data(current_path[:, 0], current_path[:, 1])
    # Update 3D path (compute z values using f)
    current_path_z = f(current_path[:, 0], current_path[:, 1])
    path_line_3d.set_data(current_path[:, 0], current_path[:, 1])
    path_line_3d.set_3d_properties(current_path_z)
    
    # ----- Update the current level curve -----
    current_point = points[i]
    current_value = f(current_point[0], current_point[1])
    x_curve, y_curve = compute_level_curve_points(current_value)
    # 2D level curve
    level_curve_2d.set_data(x_curve, y_curve)
    # 3D level curve (at z = current_value)
    level_curve_3d.set_data(x_curve, y_curve)
    level_curve_3d.set_3d_properties(np.full_like(x_curve, current_value))
    
    # ----- Update the gradient arrow in the 2D plot -----
    # Remove the previous arrow (if any) and draw a new one
    if gradient_quiver is not None:
        gradient_quiver.remove()
    g = grad_f(current_point[0], current_point[1])
    gradient_quiver = ax2d.quiver(current_point[0], current_point[1], 
                                  g[0], g[1], color='orange', 
                                  angles='xy', scale_units='xy', scale=1, width=0.005)
    
    # Optionally update the title to display the current step and function value
    ax2d.set_title(f'Gradient Ascent (2D) - Step {i}, f(x,y) = {current_value:.2f}')
    
    return path_line_2d, level_curve_2d, path_line_3d, level_curve_3d, gradient_quiver

# Create the animation: each frame lasts 500 ms.
ani = FuncAnimation(fig, animate, frames=len(points),
                    init_func=init, interval=500, blit=False, repeat=False)

plt.tight_layout()
plt.show()
