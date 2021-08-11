import numpy as np
from numpy.testing import assert_almost_equal
from scipy.sparse import diags, issparse

#ħ = 1.05e-34  # planck's constant J s
ħ = 1  # planck's constant J s

class Operator(object):
    """
    Class for 6.S079 Quantum Circuits, designed to work with operators, and apply them correctly
    to Wavefunction objects.
    
    Here's a basic example that creates an identity operator, operates with it on a simple 1-D
    gaussian, and then prints the result.

    >>> I = Operator(lambda x: x)
    >>> wf = Ket.init_gaussian((0,1))
    >>> print(I(wf)(0))
    (0.6316187777460647+0j)

    """

    def __init__(self, ofunc):
        """
        Initializes an instance of the Operator class, given a function that returns a Wavefunction.
        """
        self.O  = ofunc

    def __add__(self, op2):
        """
        Adds two objects of class Operator together. Returns a new Operator
        """
        return Operator(lambda ψ: self(ψ) + op2(ψ))
    
    def __iadd__(self, op2):
        """
        Adds an object of class Operator to another in place (using +=)
        """
        return self + op2
        

    def __sub__(self, op2):
        """
        Subtracts one object of class Operator from another. Returns a new 
        Operator.
        """
        return self + (-1)*op2
    
    def __isub__(self, op2):
        """
        Subtracts one object of class Operator from another one in place (using -=)
        """
        return self - op2
        
    
    def __mul__(self, sc):
        """
        Multiplies an operator by a scalar (with the scalar on the right)
        """
        return Operator(lambda ψ: sc*self(ψ))
    
    def __rmul__(self, sc):
        """
        Multiplies an operator by a scalar (with the scalar on the left)
        """
        return Operator(lambda ψ: sc*self(ψ))
    
    def __imul__(self, sc):
        """
        Multiplies an operator in place by a scalar (using the *= command)
        """
        return self * sc
        
    def __matmul__(self, op2):
        """
        Multiplies an operator by another operator (using the matrix multiplication @ command)
        """
        if not isinstance(op2, Operator):
            return NotImplemented
        return Operator(lambda ψ: op2(self(ψ)))

    def __rmatmul__(self, bra):
        """Operators can operate either to the left or to the right, assuming
        the operator is Hermitian, i.e. it is its own conjugate
        transpose.  Any observable, i.e. any measurable physical
        quantity, such as charge Q or flux Φ, or the energy operator
        (i.e. the Hamiltonian), will be a Hermitian matrix.  

        As a result, the operator acting to the left or to the right is
        the same operation, i.e. <bra|oper is calculated the same way
        oper|ket> is calculated.
        """
        if not isinstance(bra, Bra):
            raise TypeError("""
            You tried to operate on a ket from the right, that is not allowed.
            """)
        return self(bra)

    def __imatmul__(self, op2):
        """
        Multiplies an operator in place by another operator (using the @= command)
        """
        return self @ op2

        
    def __truediv__(self, sc):
        """
        Divide an operator by a scalar.
        """
        return Operator(lambda ψ: self(ψ)/sc)
    
    def __itruediv__(self, sc):
        """
        Divide an operator by a scalar in place (using the /= command)
        """
        return self / sc

    def __call__(self, ψ):
        if not isinstance(ψ, Wavefunction):
            raise TypeError('Operator must operate on a Wavefunction or Ket instance.')
        return self.O(ψ)

class Op_matx(object):
    """
    class for discretized operators, to be used with wavevector class.

    FIXME: It might be nice to have this inherit from sparse

    >>> I = Op_matx(diags([1,1,1]))
    >>> print(I.matx.todense())
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]

    >>> KE = Op_matx.make_KE((-2, 2, 5, ħ))
    >>> potential = Op_matx.from_function(lambda x: x**2/2, (-2,2,5))
    >>> Hamiltonian = KE + potential
    >>> print(Hamiltonian.matx.todense())
    [[2. -1.j  0. +0.5j 0. +0.j  0. +0.j  0. +0.j ]
     [0. +0.5j 0.5-1.j  0. +0.5j 0. +0.j  0. +0.j ]
     [0. +0.j  0. +0.5j 0. -1.j  0. +0.5j 0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.5j 0.5-1.j  0. +0.5j]
     [0. +0.j  0. +0.j  0. +0.j  0. +0.5j 2. -1.j ]]
    """
    
    def __init__(self, sparse_matx):
        """
        create numpy array of operator matrix based on a sparse matrix
        """
        if issparse(sparse_matx):
            self.matx = sparse_matx
        else:
            raise TypeError("""
            Op_matx expects sparse matrices.  You can make a sparse matrix by taking
            any list-like object and passing it to scipy.sparse.diags or similar method
            """)

    @classmethod
    def from_function(cls, func, *args):
        """ from_function::create n-dimensional diagonal potential-energy operator matrix
        from a function that varies in the parameter space of the system.  The function is
        assumed to be local, i.e. to depend only on the local coordinate, and thus the
        resulting matrix will be diagonal.

        Arguments should be a list of tuples in the form (min, max, N) where N is the
        number of points along that dimension.

        In this example, we create an operator from the sum of the coordinate values
        of the row,column indices of each element.

        >>> I = Op_matx.from_function(lambda *x: sum(x), (0,2,3), (-2,0,3))
        >>> print(I.matx.todense())
        [[-2.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0. -1.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0. -1.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  1.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  1.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.  2.]]

        """
        axis_arrays = []
        for arg in args:
            axis_arrays.append(np.linspace(*arg))
        X = np.meshgrid(*axis_arrays)
        return cls(diags(func(*X).flatten()))

    @classmethod
    def make_KE(cls, *args):
        """create operator (Op_matx) corresponding to kinetic energy operator

        Create n-dimensional sparse kinetic energy operator matrix
        from the second derivative in the n-dimensional parameter
        space of the system,

        A multi-dimensional KE operator looks something like:

        KE(x,y,z,...) = ⅉ ħ/(2 m_x) ∂²/∂x² +  ⅉ ħ/(2 m_y) ∂²/∂y² +  ⅉ ħ/(2 m_z) ∂²/∂z² + ...
        arguments should be a list of tuples in the form (xmin, xmax, N, m_eff) where N is the
        number of points along that dimension and m_eff an effective mass.

        >>> op = Op_matx.make_KE((-2, 2, 5, 1.05e-34))
        >>> print(op.matx.todense())
        [[0.-1.j  0.+0.5j 0.+0.j  0.+0.j  0.+0.j ]
         [0.+0.5j 0.-1.j  0.+0.5j 0.+0.j  0.+0.j ]
         [0.+0.j  0.+0.5j 0.-1.j  0.+0.5j 0.+0.j ]
         [0.+0.j  0.+0.j  0.+0.5j 0.-1.j  0.+0.5j]
         [0.+0.j  0.+0.j  0.+0.j  0.+0.5j 0.-1.j ]]

        """
        coeffs = []
        for x_min, x_max, Nx, m_eff in args:
            Δx = (x_max - x_min + 1)/Nx
            coeffs.append(1j*ħ/2/m_eff/Δx**2)

        # We need to know how large the final (flattened) matrix is going to be
        KE_matx_len = np.prod([N for _, _,N,_ in args])

        #let's make an array for each diag/set of 2 diags
        diag_list, placement_list = [], []
        central = np.full(KE_matx_len, -2*sum(coeffs))
        diag_list.append(central)
        placement_list.append(0)
        dim_offset_factor = 1
        for (_,_,N,_),coeff in zip(args,coeffs):
            #make an off diag, put it at 1, N0, N0*N1, respectively. Subtract that many
            #from off diag's length. Then multiply by that dimension's coeff.
            #Append each off diag to diag_list twice.
            off_diag = np.full(KE_matx_len - dim_offset_factor, coeff)
            diag_list.append(off_diag)
            placement_list.append(dim_offset_factor)
            diag_list.append(off_diag)
            placement_list.append(-dim_offset_factor)
            dim_offset_factor *= N
        return Op_matx(diags(diag_list,placement_list)) #no periodic b.c.

    def __add__(self, arg):
        return self.__class__(self.matx + arg.matx)

    def __iadd__(self, arg):
        self.matx += arg.matx

    def __sub__(self, arg):
        return self.__class__(self.matx - arg.matx)

    def __isub__(self, arg):
        self.matx -= arg.matx

    def __matmul__(self, arg):
        return self.__class__(self.matx @ arg.matx)

    def __imatmul__(self, arg):
        self.matx @= arg.matx

    def dot(self, arg):
        """
        >>> I = Op_matx(diags([1,1,1]))
        >>> wf = Wavefunction.init_gaussian((0,1))
        >>> wv = Wavevector.from_wf(wf, (-1, 1, 3))
        >>> print(I.dot(wv))
        [0.4919052 +0.j 0.63161878+0.j 0.4919052 +0.j]
        """
        return self.matx.dot(arg)
    

def make_operator_at_time(t, func_0, *args, t_dep = True):
    """
    make_operator_at_time::create Op_matx based on func_0 at
                           the specified time.

    t::time.  If operator is not time dependent, 0 should be used
    func_0::initialization function.  If function is time dependent,
            should take arguments like V(t, x, y, ...).  If function
            is not time dependent, should take args like V(x, y, ...).
    *args::each subsequent arg should be a tuple of the form (x_min, x_max, Nx).
    t_dep::optional boolean argument, required if function is not 
           time dependent

    >>> V = lambda t, x, y: (x**2 + y**2)*t
    >>> new_op = make_operator_at_time(1, V, (-1, 1, 3), (-1, 1, 3))
    >>> print(new_op.matx.todense())
    [[2. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 2. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 2. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 2.]]
    """
    if t_dep:
        V_t = lambda *x: func_0(t, *x)
        return Op_matx.from_function(V_t, *args)
    else:
        return Op_matx.from_function(func_0, *args)  # FIXME MEMOIZE THIS
    
if __name__ == '__main__':
    from wavefunction import *
    from wavevector import *
    import doctest
    doctest.testmod()
    
    TEST_PLOTS = False
    def identity_operator(wf):
        return wf

    def phase_shift(wf):
        return 1j*wf

    I = Operator(identity_operator)
    J = Operator(phase_shift)

    wf1 = Ket.init_gaussian((0,1))
    assert I(wf1)(0) == wf1(0), "identity operator doesn't work well"

    if TEST_PLOTS:
        plot_params = {"x_range": (-4, 4), "N": 40,
                       "method": "pdf", "x_label": "Q"}
        plt.close()
        wf1.plot_wf(**plot_params)
        plt.show()
        (I+I)(wf1).plot_wf(**plot_params)
        plt.show()
    assert (I+I)(wf1)(0) == 2*wf1(0), "operator addition isn't working"
    assert (3*I)(wf1)(0) == 3*wf1(0), "operator rmul isn't working"
    assert (I*3)(wf1)(0) == 3*wf1(0), "operator mul isn't working"
    I *= 3
    assert I(wf1)(0) == 3*wf1(0), "operator imul isn't working"
    I /= 3
    assert I(wf1)(0) == wf1(0), "operator itruediv isn't working"
    # test matmul and imatmul
    assert (I @ J)(wf1)(0) == 1j*wf1(0), "operator matmul isn't working"
    I @= J
    assert I(wf1)(0) == 1j*wf1(0), "operator imatmul isn't working"
    I = Operator(identity_operator)
    #test subtraction
    I -= J
    assert I(wf1)(0) == (1-1j)*wf1(0), "operator isub or sub not working"
    I += J
    assert I(wf1)(0) == wf1(0), "iadd isn't working"
    assert I(wf1)(0) == wf1(0), "iadd isn't working"
    assert I(wf1)(0) == wf1(0), "iadd isn't working"

    #test operator @ Ket

    assert (J @ wf1)(0) == (1j * wf1)(0), "wavefunction rmatmul not working"

    #test bra @ Operator
    bra1 = Bra.init_gaussian((0,1))
    #assert (bra1 @ J)(0) == (1j * bra1)(0), "bra @ operator not working"

    #test bra @ Operator
    bra1 = Bra.init_gaussian((0,1))
    ket1 = Ket.init_gaussian((0,1))
    assert (bra1 @ J)(0) == (1j * bra1)(0), "bra @ operator not working"

    #test full expectation value
    plot_params = {"x_range": (-4, 4), "N": 40,
                   "method": "cartesian", "x_label": "Q"}
    #(bra1 @ J).plot_wf(**plot_params)
    #plt.show()
    assert_almost_equal(bra1 @ J @ ket1, 1j, err_msg = "Expectation value of phase shift operator not working")

    print("end")
