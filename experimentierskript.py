d = 2
n = 3

matrix = BlockMatrix(d, n)
matrix.print_matrix_dense()
matrix.get_lu()

matrix.eval_sparsity_lu()

p, l, u = matrix.get_lu()
solve_lu(p, l, u, rhs(d, n, lambda x: np.sum(x)))

u_approx = solve_lu(p, l, u, rhs(d, n, lambda x: np.sum(x)))
compute_error(d, n, u_approx, lambda x: np.sum(x)**2)

def _f(x):
  return \
    -1*(2 * np.pi * x[1] * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]) - np.pi**2 * x[0] * x[1] * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + \
    2 * np.pi * x[0] * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]) - np.pi**2 * x[1] * x[0] * np.sin(np.pi * x[1]) * np.sin(np.pi * x[0]))

def u(z):
  x,y = z[0], z[1]
  return x * np.sin(np.pi * x) * y * np.sin(np.pi * y)


plot_error(u = u, f = _f , d = 2, n_from = 5, n_to = 50, step_size = 1)

def _u(z):
  x = z[0]
  return x * np.sin(np.pi* x)
def _f(z):
  x = z[0]
  return -np.pi * (2*np.cos(np.pi* x) - np.pi* x * np.sin(np.pi* x))
plot_error(u = _u, f = _f, d = 1, n_from = 5, n_to = 100, step_size = 1)

matrix = BlockMatrix(d = 3, n = 5)

matrix.eval_sparsity(), matrix.eval_sparsity_lu()

plot_2d_function(lambda x,y: x * np.sin(np.pi * x) * y * np.sin(np.pi * y))

d = 2
n = 4
def _f(x):
  return \
    -1*(2 * np.pi * x[1] * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]) - np.pi**2 * x[0] * x[1] * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + \
    2 * np.pi * x[0] * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]) - np.pi**2 * x[1] * x[0] * np.sin(np.pi * x[1]) * np.sin(np.pi * x[0]))


matrix = BlockMatrix(d, n)

p, l, u = matrix.get_lu()
u_approx = solve_lu(p, l, u, rhs(d, n, _f))
m = sol_vector_to_matrix(u_approx)

plot_2d_from_matrix(m)