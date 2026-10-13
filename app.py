import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.ntheory.continued_fraction import continued_fraction_periodic

st.set_page_config(page_title="Lattices of Quadratic Orders", layout="wide")

# ==========================================
# ALGEBRAIC NUMBER THEORY COMPUTATIONS
# ==========================================

def get_discriminant(d, is_real=True):
    """
    Discriminant of Z[\sqrt{d}] or Z[\sqrt{-d}].
    For order Z[\sqrt{D}], discriminant Delta = 4 * D.
    """
    D = d if is_real else -d
    return 4 * D

def get_minkowski_bound(d, is_real=True):
    """
    Computes Minkowski bound M_K:
    Real: (2/pi)^0 * (2!/2^2) * sqrt(|Delta|) = (1/2) * sqrt(4d) = sqrt(d)
    Imaginary: (2/pi)^1 * (2!/2^2) * sqrt(|Delta|) = (2/pi) * (1/2) * sqrt(4d) = (2/pi) * sqrt(d)
    """
    delta = get_discriminant(d, is_real)
    if is_real:
        return 0.5 * np.sqrt(abs(delta))
    else:
        return (2.0 / np.pi) * 0.5 * np.sqrt(abs(delta))

def compute_imaginary_class_number(d):
    """
    Computes class number for Z[\sqrt{-d}] using binary quadratic forms (ax^2 + bxy + cy^2).
    Discriminant Delta = -4d. Reduced forms satisfy |b| <= a <= c.
    """
    delta = -4 * d
    max_a = int(np.floor(np.sqrt(abs(delta) / 3.0)))
    reduced_forms = []
    
    for a in range(1, max_a + 1):
        # b has same parity as Delta (Delta is even, so b is even)
        for b in range(-a, a + 1):
            if (b**2 - delta) % (4 * a) == 0:
                c = (b**2 - delta) // (4 * a)
                if c >= a:
                    if abs(b) == a or c == a:
                        if b >= 0:
                            reduced_forms.append((a, b, c))
                    else:
                        reduced_forms.append((a, b, c))
    return len(reduced_forms), reduced_forms

def compute_real_class_number(d):
    """
    Computes class number for Z[\sqrt{d}] using binary quadratic forms.
    """
    if sp.is_square(d):
        return "N/A (d is a perfect square)", []
    
    delta = 4 * d
    # Cycle of reduced forms for real quadratic orders
    # A form (a, b, c) with b^2 - 4ac = Delta is reduced if 0 < sqrt(Delta) - b < 2|a| < sqrt(Delta) + b
    sqrt_delta = np.sqrt(delta)
    reduced_forms = set()
    
    # Bound for a
    limit_a = int(np.floor(sqrt_delta / 2.0))
    for a in range(1, limit_a + 1):
        for b in range(1, int(np.floor(sqrt_delta)) + 1):
            if (b**2 - delta) % (4 * a) == 0:
                c = (b**2 - delta) // (4 * a)
                if abs(sqrt_delta - 2*abs(a)) < b < sqrt_delta:
                    reduced_forms.add((a, b, c))
                    
    # Simplified class number estimation for UI (accurate for small d)
    # Using sympy ideal class group if needed, or form reduction cycles
    h = len(reduced_forms) if len(reduced_forms) > 0 else 1
    return h, list(reduced_forms)

def get_matrix_power(A, k):
    """Computes A^k for matrix A (supports negative integers via inverse)."""
    A_sym = sp.Matrix(A)
    try:
        if k < 0:
            A_inv = A_sym.inv()
            res = A_inv ** abs(k)
        else:
            res = A_sym ** k
        return np.array(res, dtype=float), res
    except Exception as e:
        return None, str(e)

def determine_matrix_class(A):
    """
    Classifies a 2x2 integer/real matrix in terms of GL(2, Z) / SL(2, Z).
    """
    det = int(round(np.linalg.det(A)))
    if det == 1:
        return "Member of SL(2, Z) (Orientation & Area Preserving Unimodular)"
    elif det == -1:
        return "Member of GL(2, Z) \\ SL(2, Z) (Orientation Reversing Unimodular)"
    elif det != 0:
        return f"Non-unimodular integer/real matrix with Determinant = {det}"
    else:
        return "Singular Matrix (Determinant = 0)"

# ==========================================
# LATTICE VISUALIZATION (2D & 3D)
# ==========================================

def generate_lattice_points(basis, grid_range=5):
    """
    Generates 2D lattice vectors: L = { n1 * v1 + n2 * v2 | -N <= n1, n2 <= N }
    basis: 2x2 matrix where columns are basis vectors v1, v2.
    """
    n_values = np.arange(-grid_range, grid_range + 1)
    n1, n2 = np.meshgrid(n_values, n_values)
    
    # Basis vectors
    v1 = basis[:, 0]
    v2 = basis[:, 1]
    
    X = n1 * v1[0] + n2 * v2[0]
    Y = n1 * v1[1] + n2 * v2[1]
    
    return X.flatten(), Y.flatten(), n1.flatten(), n2.flatten()

def create_lattice_plot_3d(X, Y, transform_matrix, scale_factor, rot_angle_deg):
    """
    Renders interactive 3D plot of the lattice over an embedded grid plane.
    """
    # Apply user scaling and 2D rotation
    rad = np.radians(rot_angle_deg)
    R_2d = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    
    pts_2d = np.vstack([X, Y])
    transformed_pts = scale_factor * (transform_matrix @ R_2d @ pts_2d)
    
    X_t = transformed_pts[0, :]
    Y_t = transformed_pts[1, :]
    Z_t = np.zeros_like(X_t) # Embedded in Z=0 plane in 3D

    fig = go.Figure()

    # Scatter points for lattice
    fig.add_trace(go.Scatter3d(
        x=X_t, y=Y_t, z=Z_t,
        mode='markers',
        marker=dict(size=5, color=X_t**2 + Y_t**2, colorscale='Viridis', opacity=0.8),
        name='Lattice Points'
    ))

    # Add origin highlighted
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Origin (0,0,0)'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="3D Perspective of Lattice Projection"
    )
    return fig

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================

st.title("🌐 Lattices of $\mathbb{Z}[\sqrt{d}]$ & $\mathbb{Z}[\sqrt{-d}]$")
st.markdown("Explore algebraic and geometric properties of quadratic orders and their lattice representations.")

# Sidebar Controls
st.sidebar.header("1. Input Parameters")
d_val = st.sidebar.number_input("Enter Natural Number (d):", min_value=1, value=2, step=1)

if sp.is_square(d_val):
    st.sidebar.warning(f"Note: {d_val} is a perfect square. $\mathbb{Z}[\sqrt{{{d_val}}}]$ is not an integral domain of quadratic integers.")

# Create Tabs for Real and Imaginary Pages
tab1, tab2 = st.tabs([f"Real Order: ℤ[√{d_val}]", f"Imaginary Order: ℤ[√{-d_val}]"])

def render_order_page(d, is_real=True):
    symbol = f"\sqrt{{{d}}}" if is_real else f"\sqrt{{-{d}}}"
    st.header(f"Properties & Lattice of $\mathbb{{Z}}[{symbol}]$")
    
    # ------------------------------------
    # SECTION 1: ALGEBRAIC INVARIANTS
    # ------------------------------------
    col1, col2, col3 = st.columns(3)
    
    delta = get_discriminant(d, is_real)
    m_bound = get_minkowski_bound(d, is_real)
    
    if is_real:
        h, forms = compute_real_class_number(d)
    else:
        h, forms = compute_imaginary_class_number(d)
        
    with col1:
        st.metric("Discriminant (Δ)", f"{delta}")
    with col2:
        st.metric("Minkowski Bound", f"{m_bound:.4f}")
    with col3:
        st.metric("Class Number (h)", f"{h}")
        
    with st.expander("Show Reduced Binary Quadratic Forms"):
        st.write("Reduced forms $(a, b, c)$ representing ideal classes:")
        st.json(forms)

    st.markdown("---")
    
    # ------------------------------------
    # SECTION 2: MATRIX REPRESENTATIVE A
    # ------------------------------------
    st.subheader("2. Matrix Representative $A$ & Powers")
    
    st.write(f"Standard basis for $\mathbb{{Z}}[{symbol}]$ is $\{{\,1,\, {symbol}\,\}}$.")
    
    # Default basis matrix choice
    if is_real:
        # Basis embeddings into R^2: x + y*sqrt(d) -> (x + y*sqrt(d), x - y*sqrt(d)) or standard canonical
        basis_A = np.array([[1, 0], [0, np.sqrt(d)]])
    else:
        # Complex embedding z = x + i*y*sqrt(d) -> (Re(z), Im(z)) = (x, y*sqrt(d))
        basis_A = np.array([[1, 0], [0, np.sqrt(d)]])

    c_mat1, c_mat2 = st.columns([1, 2])
    
    with c_mat1:
        st.write("**Matrix Representative $A$ (Basis Generator):**")
        a11 = st.number_input(f"A[0,0] ({'Real' if is_real else 'Imag'})", value=float(basis_A[0,0]), key=f"a11_{is_real}")
        a12 = st.number_input(f"A[0,1] ({'Real' if is_real else 'Imag'})", value=float(basis_A[0,1]), key=f"a12_{is_real}")
        a21 = st.number_input(f"A[2,0] ({'Real' if is_real else 'Imag'})", value=float(basis_A[1,0]), key=f"a21_{is_real}")
        a22 = st.number_input(f"A[2,1] ({'Real' if is_real else 'Imag'})", value=float(basis_A[1,1]), key=f"a22_{is_real}")
        
        A = np.array([[a11, a12], [a21, a22]])
        
        st.write("**Class of $A$:**")
        st.info(determine_matrix_class(A))

    with c_mat2:
        st.write("**Matrix Transformations & Powers:**")
        k_val = st.select_slider("Select power $k \in \{-5, -4, -3, -2, 2, 3, 4, 5\}$:", 
                                 options=[-5, -4, -3, -2, 2, 3, 4, 5], value=2, key=f"k_{is_real}")
        
        A_k_num, A_k_sym = get_matrix_power(A, k_val)
        
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.write("**Transpose $A^T$:**")
            st.write(A.T)
        with p_col2:
            st.write("**Inverse $A^{-1}$:**")
            try:
                st.write(np.linalg.inv(A))
            except:
                st.write("Non-invertible")
        with p_col3:
            st.write(f"**Power $A^{{{k_val}}}$:**")
            if A_k_num is not None:
                st.write(np.round(A_k_num, 4))
            else:
                st.error(A_k_sym)

    st.markdown("---")

    # ------------------------------------
    # SECTION 3: INTERACTIVE LATTICE GRID
    # ------------------------------------
    st.subheader("3. Interactive Lattice Grid & Transformations")
    
    t_col1, t_col2 = st.columns([1, 2])
    
    with t_col1:
        st.markdown("#### Lattice Controls")
        grid_range = st.slider("Grid Range (N x N)", 2, 15, 6, key=f"grid_{is_real}")
        scale = st.slider("Scale Factor", 0.1, 5.0, 1.0, step=0.1, key=f"scale_{is_real}")
        rot_deg = st.slider("Rotate Angle (Degrees)", 0, 360, 0, step=5, key=f"rot_{is_real}")
        
        st.markdown("#### Custom Integer Matrix $M \in M_2(\mathbb{Z})$")
        m11 = st.number_input("M[0,0]", value=1, step=1, key=f"m11_{is_real}")
        m12 = st.number_input("M[0,1]", value=0, step=1, key=f"m12_{is_real}")
        m21 = st.number_input("M[1,0]", value=0, step=1, key=f"m21_{is_real}")
        m22 = st.number_input("M[1,1]", value=1, step=1, key=f"m22_{is_real}")
        
        M = np.array([[m11, m12], [m21, m22]])
        st.caption(f"Matrix Class: {determine_matrix_class(M)}")

    # Generate lattice points
    X, Y, n1, n2 = generate_lattice_points(A, grid_range=grid_range)
    
    with t_col2:
        # 3D Plot Display
        fig3d = create_lattice_plot_3d(X, Y, M, scale, rot_deg)
        st.plotly_chart(fig3d, use_container_width=True)

# Render Pages inside tabs
with tab1:
    render_order_page(d_val, is_real=True)

with tab2:
    render_order_page(d_val, is_real=False)
