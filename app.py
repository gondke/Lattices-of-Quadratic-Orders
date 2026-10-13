import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp

st.set_page_config(page_title="Lattices of Quadratic Orders", layout="wide")

# ==========================================
# ALGEBRAIC & QUADRATIC FORM COMPUTATIONS
# ==========================================

def is_perfect_square(n: int) -> bool:
    """Checks if an integer n is a perfect square using SymPy."""
    if n < 0:
        return False
    root, exact = sp.integer_nthroot(int(n), 2)
    return exact

def get_discriminant(d: int, is_real: bool = True) -> int:
    """Discriminant of Z[\sqrt{d}] (Delta = 4d) or Z[\sqrt{-d}] (Delta = -4d)."""
    return 4 * d if is_real else -4 * d

def get_minkowski_bound_sym(d: int, is_real: bool = True):
    """Exact Symbolic Minkowski bound."""
    if is_real:
        return sp.sqrt(d)
    else:
        return (2 / sp.pi) * sp.sqrt(d)

def get_reduced_binary_forms(d: int, is_real: bool = True):
    """
    Computes reduced binary quadratic forms (a, b, c) of discriminant Delta.
    Categorizes them into Principal Class (Identity) and Non-Principal Classes.
    """
    delta = get_discriminant(d, is_real)
    forms = []
    
    if not is_real:
        # Imaginary case: |b| <= a <= c
        max_a = int(np.floor(np.sqrt(abs(delta) / 3.0)))
        for a in range(1, max_a + 1):
            for b in range(-a, a + 1):
                if (b**2 - delta) % (4 * a) == 0:
                    c = (b**2 - delta) // (4 * a)
                    if c >= a:
                        if abs(b) == a or c == a:
                            if b >= 0:
                                forms.append((a, b, c))
                        else:
                            forms.append((a, b, c))
    else:
        # Real case: 0 < sqrt(Delta) - b < 2|a| < sqrt(Delta) + b
        if is_perfect_square(d):
            return [(1, 0, -d)]
        
        sqrt_delta = np.sqrt(delta)
        limit_a = int(np.floor(sqrt_delta / 2.0))
        seen = set()
        for a in range(1, limit_a + 1):
            for b in range(1, int(np.floor(sqrt_delta)) + 1):
                if (b**2 - delta) % (4 * a) == 0:
                    c = (b**2 - delta) // (4 * a)
                    if abs(sqrt_delta - 2*abs(a)) < b < sqrt_delta:
                        if (a, b, c) not in seen:
                            seen.add((a, b, c))
                            forms.append((a, b, c))
        if not forms:
            forms = [(1, 0, -d)]
            
    return forms

def form_to_matrix_sym(a: int, b: int, c: int, d: int, is_real: bool = True) -> sp.Matrix:
    """
    Constructs the canonical 2x2 lattice basis matrix corresponding to form (a,b,c) / ideal class.
    """
    if a == 1:
        # Principal ideal basis matrix
        return sp.Matrix([[1, 0], [0, sp.sqrt(d) if is_real else sp.sqrt(-d)]])
    else:
        # Non-principal ideal lattice matrix
        # Basis: {a, (b + sqrt(Delta))/2}
        sqrt_part = sp.sqrt(d) if is_real else sp.sqrt(-d)
        return sp.Matrix([[a, sp.Rational(b, 2)], [0, sqrt_part]])

def get_symbolic_matrix_power(A_sym: sp.Matrix, k: int):
    """Computes exact symbolic A^k for SymPy matrix A."""
    try:
        if k < 0:
            A_inv = A_sym.inv()
            res = A_inv ** abs(k)
        else:
            res = A_sym ** k
        return res, None
    except Exception as e:
        return None, str(e)

def determine_matrix_class(A_sym: sp.Matrix) -> str:
    """Classifies a matrix in terms of GL(2, Z) / SL(2, Z)."""
    try:
        det = sp.simplify(A_sym.det())
        if det == 1:
            return "Member of SL(2, Z) (Orientation & Area Preserving Unimodular)"
        elif det == -1:
            return "Member of GL(2, Z) \\ SL(2, Z) (Orientation Reversing Unimodular)"
        elif det != 0:
            return f"Non-unimodular matrix with Determinant = ${sp.latex(det)}$"
        else:
            return "Singular Matrix (Determinant = 0)"
    except Exception:
        return "General Matrix"

# ==========================================
# LATTICE VISUALIZATION (3D)
# ==========================================

def generate_lattice_points(basis_num: np.ndarray, grid_range: int = 5):
    """Generates lattice vectors for standard Z^2 grid and transformed order lattice."""
    n_values = np.arange(-grid_range, grid_range + 1)
    n1, n2 = np.meshgrid(n_values, n_values)
    
    # Standard Z^2 integer grid
    X_grid, Y_grid = n1.flatten(), n2.flatten()
    
    # Transformed ideal lattice points
    v1 = basis_num[:, 0]
    v2 = basis_num[:, 1]
    
    X_lat = n1 * v1[0] + n2 * v2[0]
    Y_lat = n1 * v1[1] + n2 * v2[1]
    
    return X_grid, Y_grid, X_lat.flatten(), Y_lat.flatten()

def create_lattice_plot_3d(X_grid, Y_grid, X_lat, Y_lat, transform_matrix_num, scale_factor, rot_angle_deg, d, is_real, class_label):
    """Renders interactive 3D plot with distinct colors for Z^2 and Ideal Lattice."""
    rad = np.radians(rot_angle_deg)
    R_2d = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    
    # Transform Standard Grid Points (Z^2)
    pts_grid = np.vstack([X_grid, Y_grid])
    trans_grid = scale_factor * (transform_matrix_num @ R_2d @ pts_grid)
    X_g_t, Y_g_t = trans_grid[0, :], trans_grid[1, :]
    Z_g_t = np.zeros_like(X_g_t)

    # Transform Ideal Lattice Points
    pts_lat = np.vstack([X_lat, Y_lat])
    trans_lat = scale_factor * (transform_matrix_num @ R_2d @ pts_lat)
    X_l_t, Y_l_t = trans_lat[0, :], trans_lat[1, :]
    Z_l_t = np.zeros_like(X_l_t)

    fig = go.Figure()

    # Trace 1: Standard Integer Grid Z^2
    fig.add_trace(go.Scatter3d(
        x=X_g_t, y=Y_g_t, z=Z_g_t,
        mode='markers',
        marker=dict(size=4, color='#FF9900', opacity=0.4),
        name='Standard Grid (ℤ²)'
    ))

    # Trace 2: Ideal Class Lattice
    fig.add_trace(go.Scatter3d(
        x=X_l_t, y=Y_l_t, z=Z_l_t,
        mode='markers',
        marker=dict(size=6, color='#00E5FF', opacity=0.9),
        name=f'Lattice of {class_label}'
    ))

    # Highlight origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=9, color='red'),
        name='Origin (0,0,0)'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            aspectmode='data'
        ),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"3D Lattice Geometry: {class_label}"
    )
    return fig

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================

st.title("🌐 Lattices of $\mathbb{Z}[\sqrt{d}]$ & $\mathbb{Z}[\sqrt{-d}]$")
st.markdown("Explore algebraic invariants, class groups, and geometry of quadratic orders.")

# Sidebar Controls
st.sidebar.header("1. Input Parameters")
d_val = st.sidebar.number_input("Enter Natural Number (d):", min_value=1, value=5, step=1)

if is_perfect_square(d_val):
    st.sidebar.warning(f"Note: {d_val} is a perfect square. $\mathbb{{Z}}[\sqrt{{{d_val}}}]$ is not an integral domain of quadratic integers.")

# Create Tabs for Real and Imaginary Pages
tab1, tab2 = st.tabs([f"Real Order: ℤ[√{d_val}]", f"Imaginary Order: ℤ[√{-d_val}]"])

def render_order_page(d: int, is_real: bool = True):
    symbol = f"\sqrt{{{d}}}" if is_real else f"\sqrt{{-{d}}}"
    st.header(f"Properties & Lattice of $\mathbb{{Z}}[{symbol}]$")
    
    # ------------------------------------
    # SECTION 1: ALGEBRAIC INVARIANTS & CLASS GROUP
    # ------------------------------------
    delta = get_discriminant(d, is_real)
    m_bound_sym = get_minkowski_bound_sym(d, is_real)
    m_bound_val = float(m_bound_sym.evalf())
    forms = get_reduced_binary_forms(d, is_real)
    h = len(forms)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Discriminant (Δ)", f"{delta}")
    with col2:
        st.markdown("**Minkowski Bound ($M_K$):**")
        st.latex(f"M_K = {sp.latex(m_bound_sym)} \\approx {m_bound_val:.4f}")
    with col3:
        st.metric("Class Number (h)", f"{h}")

    st.markdown("---")
    
    # ------------------------------------
    # SECTION 2: CLASS GROUP & INTERACTIVE CLASS SELECTION
    # ------------------------------------
    st.subheader("2. Class Group $Cl(K)$ & Quadratic Forms")
    st.write(f"The order has **{h} ideal class(es)** corresponding to the reduced binary quadratic forms below:")
    
    # Display Quadratic Forms in Textbook Typesetting
    form_options = []
    for idx, (a, b, c) in enumerate(forms):
        is_principal = (a == 1)
        type_str = "Principal Class [Identity]" if is_principal else f"Non-Principal Class #{idx}"
        form_latex = f"Q_{{{idx+1}}}(x, y) = {a}x^2 + ({b})xy + ({c})y^2"
        form_options.append((idx, a, b, c, type_str, form_latex))

    # Interactive Radio / Button Selection for Ideal Classes
    selected_idx = st.radio(
        "Select an Ideal Class from $Cl(K)$ to inspect its Matrix Representative & Lattice:",
        options=[opt[0] for opt in form_options],
        format_func=lambda i: f"Class [{i+1}]: {form_options[i][4]}  ⟶  {form_options[i][2]**2 - 4*form_options[i][1]*form_options[i][3]} | Q(x,y) = {form_options[i][1]}x² + {form_options[i][2]}xy + {form_options[i][3]}y²",
        key=f"class_select_{is_real}_{d}"
    )

    sel_idx, sel_a, sel_b, sel_c, sel_type, sel_latex = form_options[selected_idx]
    
    st.markdown(f"**Selected Form Typesetting:**")
    st.latex(f"{sel_latex} \quad \\text{{({sel_type})}}")

    st.markdown("---")
    
    # ------------------------------------
    # SECTION 3: MATRIX REPRESENTATIVE A & POWERS
    # ------------------------------------
    st.subheader(f"3. Matrix Representative $A$ for {sel_type}")
    
    # Default basis matrix derived directly from selected ideal class
    A_default = form_to_matrix_sym(sel_a, sel_b, sel_c, d, is_real)

    c_mat1, c_mat2 = st.columns([1, 2])
    
    with c_mat1:
        st.write("**Matrix Representative $A$ (Basis Generator):**")
        st.caption("Customizable entry notation (e.g., `sqrt(d)`, `1`, `0`):")
        
        a11_str = st.text_input("A[0,0]", value=str(A_default[0,0]), key=f"a11_s_{is_real}_{selected_idx}")
        a12_str = st.text_input("A[0,1]", value=str(A_default[0,1]), key=f"a12_s_{is_real}_{selected_idx}")
        a21_str = st.text_input("A[1,0]", value=str(A_default[1,0]), key=f"a21_s_{is_real}_{selected_idx}")
        a22_str = st.text_input("A[1,1]", value=str(A_default[1,1]), key=f"a22_s_{is_real}_{selected_idx}")
        
        try:
            A_sym = sp.Matrix([
                [sp.sympify(a11_str), sp.sympify(a12_str)],
                [sp.sympify(a21_str), sp.sympify(a22_str)]
            ])
        except Exception as e:
            st.error(f"Invalid input: {e}")
            A_sym = A_default
            
        st.write("**Textbook Matrix $A$:**")
        st.latex(f"A = {sp.latex(A_sym)}")
        
        st.write("**Class of Matrix $A$:**")
        st.info(determine_matrix_class(A_sym))

    with c_mat2:
        st.write("**Matrix Transformations & Powers:**")
        k_val = st.select_slider("Select power $k \in \{-5, -4, -3, -2, 2, 3, 4, 5\}$:", 
                                 options=[-5, -4, -3, -2, 2, 3, 4, 5], value=2, key=f"k_s_{is_real}_{selected_idx}")
        
        A_k_sym, err = get_symbolic_matrix_power(A_sym, k_val)
        
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.write("**Transpose $A^T$:**")
            st.latex(f"A^T = {sp.latex(A_sym.T)}")
        with p_col2:
            st.write("**Inverse $A^{-1}$:**")
            try:
                st.latex(f"A^{{-1}} = {sp.latex(A_sym.inv())}")
            except Exception:
                st.write("Non-invertible")
        with p_col3:
            st.write(f"**Power $A^{{{k_val}}}$:**")
            if A_k_sym is not None:
                st.latex(f"A^{{{k_val}}} = {sp.latex(A_k_sym)}")
            else:
                st.error(err)

    st.markdown("---")

    # ------------------------------------
    # SECTION 4: LIVE LATTICE GRID & TRANSFORMATIONS
    # ------------------------------------
    st.subheader("4. Live Geometric Deformation of the Ideal Lattice")
    
    t_col1, t_col2 = st.columns([1, 2])
    
    with t_col1:
        st.markdown("#### Lattice Controls")
        grid_range = st.slider("Grid Range (N x N)", 2, 15, 6, key=f"grid_{is_real}_{selected_idx}")
        scale = st.slider("Scale Factor", 0.1, 5.0, 1.0, step=0.1, key=f"scale_{is_real}_{selected_idx}")
        rot_deg = st.slider("Rotate Angle (Degrees)", 0, 360, 0, step=5, key=f"rot_{is_real}_{selected_idx}")
        
        st.markdown("#### Custom Integer Matrix $M \in M_2(\mathbb{Z})$")
        m11 = st.number_input("M[0,0]", value=1, step=1, key=f"m11_{is_real}_{selected_idx}")
        m12 = st.number_input("M[0,1]", value=0, step=1, key=f"m12_{is_real}_{selected_idx}")
        m21 = st.number_input("M[1,0]", value=0, step=1, key=f"m21_{is_real}_{selected_idx}")
        m22 = st.number_input("M[1,1]", value=1, step=1, key=f"m22_{is_real}_{selected_idx}")
        
        M_sym = sp.Matrix([[m11, m12], [m21, m22]])
        M_num = np.array([[m11, m12], [m21, m22]], dtype=float)
        
        st.latex(f"M = {sp.latex(M_sym)}")
        st.caption(f"Matrix Class: {determine_matrix_class(M_sym)}")

    # Convert symbolic basis matrix A_sym to numerical array for plotting
    try:
        basis_num = np.array(A_sym.evalf(), dtype=float)
    except Exception:
        basis_num = np.array([[1, 0], [0, np.sqrt(d)]], dtype=float)

    # Generate lattice points for both Z^2 grid and chosen ideal lattice
    X_grid, Y_grid, X_lat, Y_lat = generate_lattice_points(basis_num, grid_range=grid_range)
    
    with t_col2:
        fig3d = create_lattice_plot_3d(X_grid, Y_grid, X_lat, Y_lat, M_num, scale, rot_deg, d, is_real, sel_type)
        st.plotly_chart(fig3d, use_container_width=True)

# Render Pages inside tabs
with tab1:
    render_order_page(d_val, is_real=True)

with tab2:
    render_order_page(d_val, is_real=False)
