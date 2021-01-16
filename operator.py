from wavefunction import *
from numpy.testing import assert_almost_equal
from scipy.sparse import diags

class Operator(object):
    """
    Class for 6.S079 Quantum Circuits, designed to work with operators, and apply them correctly
    to Wavefunction objects.
    
    Here's a basic example:

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
        (i.e. the Hamiltonian), will be a Hermitian operator.  

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
    def __init__(self, sparse_matx)
        """
        create numpy array of operator matrix based on a sparse matrix
        """
        self.matx = sparse_matx

    @classmethod
    def from_function(cls, func, *args):
        """ from_function::create n-dimensional sparse potential energy operator matrix
        from a function that varies in the parameter space of the system.

        arguments should be a list of tuples in the form (min, max, N) where N is the
        number of points along that dimension.
        """
        axis_arrays = []
        for arg in args:
            axis_arrays.append(np.linspace(*arg))
        X = np.meshgrid(*axis_arrays)
        return cls(diags(func(*X).flatten()))

    @classmethod
    def make_KE_matx(cls, *args):
        """ d_by_dsquared:: create n-dimensional sparse kinetic energy operator matrix
        from the second derivative in the n-dimensional parameter space of the system, 

        arguments should be a list of tuples in the form (min, max, N, m_eff) where N is the
        number of points along that dimension and m_eff an effective mass.
        """
        coeffs = []
        for x_min, x_max, Nx, m_eff in args:
            Δx = (x_max - x_min + 1)/N
            coeffs.append(1j*ħ/2/m_eff/Δx**2)

        KE_matx_len = np.prod([N for _, _,N,_ in args])

        #let's make an array for each diag/set of 2 diags
        diag_list = []
        placement_list = []
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
            diag_list.append(off_diag)
            placement_list.append(dim_offset_factor)
            placement_list.append(-dim_offset_factor)
            dim_offset_factor *= N
        return sparse.diags(diag_list,placement_list) #no periodic b.c.


        
        
if __name__ == '__main__':
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
