from wavefunction import *

class Operator(object):
    """
    Class for 6.S079 Quantum Circuits, designed to work with operators, and apply them correctly
    to Wavefunction objects.
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

    def __rmatmul__(self, junk):
        print("stupid rmatmul")

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
        if (not (isinstance(ψ, Wavefunction) or isinstance(ψ, Ket))):
            raise TypeError('Operator must operate on a Wavefunction or Ket instance.')
        return self.O(ψ)
        
        
if __name__ == '__main__':
    TEST_PLOTS = False
    def identity_operator(wf):
        return wf

    def phase_shift(wf):
        return 1j*wf

    I = Operator(identity_operator)
    J = Operator(phase_shift)

    wf1 = Ket.init_gaussian((0,1))
    assert I(wf1) == wf1, "identity operator doesn't work well"

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
    print("end")
